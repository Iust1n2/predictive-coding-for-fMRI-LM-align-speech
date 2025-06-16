# predictive-coding-for-fMRI-LM-align-speech

# Setup

After setting up the conda environment and installing the dependencies in `requirements.txt`, make sure to install and setup `git-annex` for downloading the fMRI data:

```bash
pip install datalad
conda install -c conda-forge git-annex 
git annex init
```

And then download the [Narratives](https://openneuro.org/datasets/ds002345/versions/1.1.4) dataset from repo: 

```bash
bash scripts/setup.sh
```

Or download the fMRI data for only a subset of tasks:

```bash
bash scripts/setup_small.sh
```

# Training

First tokenize the [English Wikipedia](https://huggingface.co/datasets/wikipedia) dataset from Huggingface using batch tokenization and sharding by running:

```bash
python train/prepare_wiki_dataset.py
```

Finetune GPT-2 on the Wikipedia corpus: 

```bash
python train/train_cpc.py \
    --model_name_or_path gpt2 \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --alpha $alpha \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 10000000 \
    --save_total_limit 15 \
    --model_max_length 256 \
    --report_to wandb \
    --disable_tqdm True \
```