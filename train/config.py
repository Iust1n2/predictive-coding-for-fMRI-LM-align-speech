from dataclasses import  dataclass, field
from typing import Optional
import transformers

@dataclass
class DataArguments:
    data_path: str = field(
        default="train/train_data_bin/train.bin",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default= "train/train_data_bin/val.bin",
        metadata={"help": "Path to the evaluation data."}
    )
    block_size: int = field(
        default=256,
        metadata={
            "help": "Input block size for the model. Sequences will be truncated to this length."
        },
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = None
    alpha: float = field(default=0.0, metadata={"help": "Weighting factor for CPC loss."})
    evaluation_strategy: str = field(default="no", metadata={"help": "Evaluation strategy: steps, epoch, or no."})
    save_strategy: str = field(default="steps", metadata={"help": "Saving strategy: steps, epoch, or no."})
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=False)
    model_max_length: int = field(
        default=256,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    log_level: str = field(default="error")  # suppress info logs
    

