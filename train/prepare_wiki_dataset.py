import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

IGNORE_TOKEN_ID = -100

# Parameters
num_proc = 32
num_proc_load_dataset = num_proc
BATCH_SIZE = 10**6
save_dir = "train/train_data"
os.makedirs(save_dir, exist_ok=True)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    ids = tokenizer.encode(example["text"], truncation=True, max_length=tokenizer.model_max_length)
    return {"ids": ids, "len": len(ids)}

# Load Wikipedia dataset
print("Loading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split="train", num_proc=num_proc_load_dataset)
print(f"Dataset columns: {dataset.column_names}")

# Extract exactly 1,000 examples for validation
val_size = 1000
total_size = len(dataset)
test_ratio = val_size / total_size

split_dataset = dataset.train_test_split(test_size=test_ratio, seed=2357, shuffle=True)
split_dataset["val"] = split_dataset.pop("test")

print(f"Train size: {len(split_dataset['train'])}")
print(f"Validation size (1000 expected): {len(split_dataset['val'])}")

# Tokenize dataset
print("Tokenizing...")
tokenized = split_dataset.map(
    tokenize,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# Save to memory-mapped .bin
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(save_dir, f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    max_shards = len(dset)  # can't have more shards than rows
    estimated_batches = max(1, int(len(dset) * BATCH_SIZE // arr_len))
    total_batches = min(estimated_batches, max_shards)

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        shard = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(shard['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    arr.flush()
    print(f"Finished writing {split}.bin with {arr_len} tokens")