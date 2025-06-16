import os
import pathlib
import logging
from typing import Dict
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, PrinterCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb

from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from config import ModelArguments, DataArguments, TrainingArguments
from utils import freeze_gpt2_bottom_layers, WandbArtifactCallback, LogSpacedCheckpointCallback, ActivationSavingCallback, EvalLoggingCallback

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class ContrastivePredictiveCodingLoss(torch.nn.Module):
    """
    InfoNCE-based CPC loss using cosine similarity.
    y_pred:   Tensor of shape (batch_size, hidden_dim)
    y_pos:    Tensor of shape (batch_size, hidden_dim)
    y_neg:    Tensor of shape (batch_size, n_negatives, hidden_dim)
    tau:      temperature scalar
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self,
                y_pred: torch.Tensor,
                y_pos: torch.Tensor,
                y_neg: torch.Tensor) -> torch.Tensor:
        # Positive score: [batch]
        s_pos = F.cosine_similarity(y_pred, y_pos, dim=-1) / self.tau
        # Negative scores: [batch, n_neg]
        batch, n_neg, dim = y_neg.shape
        y_pred_exp = y_pred.unsqueeze(1).expand(-1, n_neg, -1)
        s_neg = F.cosine_similarity(y_pred_exp, y_neg, dim=-1) / self.tau
        # Logits: [batch, 1 + n_neg]
        logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)
        # Labels: 0 for positive sample
        labels = torch.zeros(batch, dtype=torch.long, device=logits.device)
        # InfoNCE loss via cross-entropy
        loss = F.cross_entropy(logits, labels)
        return loss


class CPCTrainer(Trainer):
    """
    Custom Trainer that combines LM CrossEntropy and CPC loss.
    Predicts hidden states at layer k=8 for token at distance d=8.
    """
    def __init__(self,
                 *args,
                 cpc_layer: int = 8,
                 cpc_distance: int = 8,
                 tau: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cpc_loss_fct = ContrastivePredictiveCodingLoss(tau=tau)
        self.ce_loss_fct = CrossEntropyLoss()
        self.cpc_layer = cpc_layer
        self.cpc_distance = cpc_distance
        self.alpha_start = self.args.alpha
        self.negative_queue = torch.empty((0, self.model.config.n_embd)).to(self.args.device)
        self.negative_queue = self.negative_queue.cpu()
        self.queue_max_size = 2500
        self.negatives_per_batch = 256 # reduced from 2000 because of memory issues, 512 takes 8.2 gb per run 

        # Frozen model Nk (used for CPC targets)
        self.Nk = transformers.AutoModelForCausalLM.from_pretrained(
            'gpt2',
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        for p in self.Nk.parameters():
            p.requires_grad = False

        # Linear projection head h: on top of f (fine-tuned) layer → match CPC target
        self.projection_head = nn.Linear(self.model.config.n_embd, self.model.config.n_embd, bias=False).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = 0

        # # Update self.alpha every 100 gradient steps for balanced contribution of CE and CPC losses, idk if this is necessary
        # if self.state.global_step is not None:
        #     self.alpha = min(1.0, self.alpha_start + self.state.global_step // self.alpha_step_size * 0.01)
        
        ## Or linear interpolation from alpha 0 to 1
        # max_steps = 10**6
        # self.alpha = min(1.0, self.state.global_step / max_steps)

        # Forward pass for fine-tuned model f
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits  # [B, L, vocab]
        hidden = outputs.hidden_states[-1]  # h(f(x_t)), [B, L, D]

        # LM CrossEntropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        not_ignore = shift_labels.ne(IGNORE_TOKEN_ID)
        shift_labels = shift_labels[not_ignore]
        ce_loss = self.ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Prepare CPC positives and negatives
        if hidden.size(1) <= self.cpc_distance:
            cpc_loss = torch.tensor(0.0, device=ce_loss.device)
        else:
            # Projected prediction: h(f(x_t))
            y_pred = hidden[:, :-self.cpc_distance, :]  # [B, L-d, D]
            y_pred = self.projection_head(y_pred)
            y_pred_flat = y_pred.reshape(-1, y_pred.size(-1))  # [B*(L-d), D]

            # Target: N^k(x_{t+d}) from frozen model
            with torch.no_grad():
                nk_outputs = self.Nk(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                nk_hidden = nk_outputs.hidden_states[self.cpc_layer]  # [B, L, D]
                y_pos = nk_hidden[:, self.cpc_distance:, :]  # [B, L-d, D]
                y_pos_flat = y_pos.reshape(-1, y_pos.size(-1))  # [B*(L-d), D]

            # === Sample negatives from the queue ===
            if self.negative_queue.shape[0] >= self.negatives_per_batch:
                idx = torch.randint(
                    0, self.negative_queue.shape[0],
                    (y_pred_flat.size(0), self.negatives_per_batch),
                    device='cpu'
                )
                y_neg = self.negative_queue[idx].to(y_pred_flat.device)  # [N, K, D]
            else:
                y_neg = y_pred_flat.unsqueeze(1)  # [N, 1, D] dummy

            # === Update negative queue ===
            with torch.no_grad():
                new_negatives = torch.cat([y_pred_flat, y_pos_flat], dim=0).detach().cpu()
                self.negative_queue = torch.cat([self.negative_queue, new_negatives], dim=0)
                if self.negative_queue.shape[0] > self.queue_max_size:
                    self.negative_queue = self.negative_queue[-self.queue_max_size:]

            # CPC loss
            cpc_loss = self.cpc_loss_fct(y_pred_flat, y_pos_flat, y_neg)

        # Total loss
        alpha = self.alpha_start
        loss = (1 - alpha) * ce_loss + alpha * cpc_loss
        
        # for k in range(1, 2):
        #     _, topk = shift_logits.topk(k, dim=-1)           # [N, k]
        #     correct = topk.eq(shift_labels.unsqueeze(-1))    # [N, k]
        #     log[f"top{k}"] = correct.any(-1).float().mean().item()

        preds = shift_logits.argmax(dim=-1)

        # Mask predictions like labels
        preds = preds[not_ignore]
        correct = (preds == shift_labels).float().mean().item()

        if self.state.global_step % 100 == 0 and self.args.local_rank in (-1, 0):
            
            wandb.log({
                "train/ce_loss": ce_loss.item(),
                "train/cpc_loss": cpc_loss.item() if isinstance(cpc_loss, torch.Tensor) else 0.0,
                "train/total_loss": loss.item(),
                "train/cpc_accuracy": correct,
            }, step=self.state.global_step) 

        return (loss, outputs) if return_outputs else loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        ce_losses = []
        cpc_losses = []
        total_losses = []
        accuracies = []

        for inputs in eval_dataloader:
            with torch.no_grad():
                inputs = self._prepare_inputs(inputs)
                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                hidden = outputs.hidden_states[self.cpc_layer]

                # LM CrossEntropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["labels"][..., 1:].contiguous()
                not_ignore = shift_labels.ne(IGNORE_TOKEN_ID)
                shift_labels = shift_labels[not_ignore]

                ce_loss = self.ce_loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # CPC
                if hidden.size(1) <= self.cpc_distance:
                    cpc_loss = torch.tensor(0.0, device=ce_loss.device)
                else:
                    y_pred = hidden[:, :-self.cpc_distance, :]
                    y_pos  = hidden[:, self.cpc_distance:, :]
                    y_pred_flat = y_pred.reshape(-1, hidden.size(-1))
                    y_pos_flat  = y_pos.reshape(-1, hidden.size(-1))
                    cpc_loss = self.cpc_loss_fct(y_pred_flat, y_pos_flat, y_pred_flat.unsqueeze(1))  # dummy negs

                total_loss = (1 - self.alpha_start) * ce_loss + self.alpha_start * cpc_loss

                preds = shift_logits.argmax(dim=-1)[not_ignore]
                correct = (preds == shift_labels).float().mean().item()

                ce_losses.append(ce_loss.item())
                cpc_losses.append(cpc_loss.item())
                total_losses.append(total_loss.item())
                accuracies.append(correct)

        metrics = {
            "ce_loss": np.mean(ce_losses),
            "cpc_loss": np.mean(cpc_losses),
            "total_loss": np.mean(total_losses),
            "cpc_accuracy": np.mean(accuracies),
        }

        return metrics
    

class WikiDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.length = len(self.data) // block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.data[start:end].astype(np.int64))
        return {
            "input_ids": x,
            "labels": x.clone(),
            "attention_mask": torch.ones_like(x)
        }
    

def make_supervised_data_module(data_args) -> Dict:
    print("Loading data from .bin files...")

    train_dataset = WikiDataset(data_args.data_path, data_args.block_size)
    eval_dataset = WikiDataset(data_args.eval_data_path, data_args.block_size) if data_args.eval_data_path else None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    freeze_gpt2_bottom_layers(model, freeze_until_layer=8)

    print("Loading data...")
    data_module = make_supervised_data_module(data_args=data_args)
    print("Data loaded successfully.")

    # Setup W&B
    wandb.init(
        project="cpc-gpt2-finetune",
        name=training_args.run_name,
        group=f"alpha_{training_args.alpha}",
        config=training_args,
        resume="allow", 
        reinit=True  
    )

    eval_loader = DataLoader(data_module["eval_dataset"],
                            batch_size=training_args.per_device_eval_batch_size,
                            shuffle=False)
    
    trainer = CPCTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[
            WandbArtifactCallback(), 
            LogSpacedCheckpointCallback(total_steps=10**6, num_checkpoints=15),
            ActivationSavingCallback(eval_dataloader=eval_loader),
            EvalLoggingCallback()
            ],
        **data_module
    )
    trainer.remove_callback(PrinterCallback)

    # Resume from checkpoint if exists
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        print(f"Resuming from checkpoint: {checkpoints[-1]}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoint found, starting training from scratch.")
        trainer.train()

    print("Training complete. Saving model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    os.makedirs("train_data", exist_ok=True)
    transformers.logging.set_verbosity_error()
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    logging.getLogger("transformers.trainer").propagate = False
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    train()
