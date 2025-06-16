import torch
import numpy as np
import transformers
from transformers import TrainerCallback
import wandb
import os

def freeze_gpt2_bottom_layers(model, freeze_until_layer=8):
    """
    Freezes GPT-2 layers from 0 up to (but not including) `freeze_until_layer`.
    
    Args:
        model: HuggingFace GPT2LMHeadModel.
        freeze_until_layer (int): The index of the first layer to keep trainable.
    """
    for name, param in model.named_parameters():
        # Freeze embeddings
        if name.startswith("transformer.wte") or name.startswith("transformer.wpe"):
            param.requires_grad = False

        # Freeze encoder block layers
        elif name.startswith("transformer.h."):
            layer_num = int(name.split(".")[2])
            if layer_num < freeze_until_layer:
                param.requires_grad = False


class WandbArtifactCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        if args.local_rank in (-1, 0):
            model_artifact = wandb.Artifact("cpc-gpt2-model", type="model")
            model_artifact.add_dir(args.output_dir)
            wandb.log_artifact(model_artifact)


class LogSpacedCheckpointCallback(transformers.TrainerCallback):
    def __init__(self, total_steps=10**6, num_checkpoints=15):
        self.steps = set(np.unique(np.logspace(0, np.log10(total_steps), num=num_checkpoints, dtype=int)))
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True


def extract_acts(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        layers_to_use = [0, 4, 8, 12]
        logits = outputs.logits
        hidden_layers = [outputs.hidden_states[i] for i in layers_to_use]
        # shape: (batch, num_layers, seq_len, hidden_dim)
        hidden_states = torch.stack(hidden_layers, dim=1)
    return logits, hidden_states

class ActivationSavingCallback(TrainerCallback):
    def __init__(self, eval_dataloader, save_dir="activations", max_batches=10):
        self.eval_dataloader = eval_dataloader
        self.save_dir = save_dir
        self.max_batches = max_batches  # control how much data to extract

    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        all_logits = []
        all_acts = []

        for i, batch in enumerate(self.eval_dataloader):
            if i >= self.max_batches:
                break
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits, activations = extract_acts(model, batch)
            all_logits.append(logits.cpu())
            all_acts.append(activations.cpu())

        all_logits = torch.cat(all_logits, dim=0)     # [total_batch, seq_len, vocab]
        all_acts = torch.cat(all_acts, dim=0)         # [total_batch, num_layers, seq_len, hidden_dim]

        save_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", self.save_dir)
        os.makedirs(save_path, exist_ok=True)
        torch.save({"logits": all_logits, "activations": all_acts}, os.path.join(save_path, "eval_activations.pt"))

        print(f"[INFO] Saved activations at step {state.global_step} to {save_path}")


class EvalLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero and metrics:
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=state.global_step)