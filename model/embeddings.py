import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model

def get_gpt2_embeddings(df_contextual, data_params):
    """
    Computes GPT-2 (Small) word-level contextual embeddings using HuggingFace Transformers.

    Args:
        df_contextual (pd.DataFrame): Must include ['word', 'word_idx', 'sentence_id'] columns.
        data_params (DataParams): Must include `embedding_layer` (int).

    Returns:
        np.ndarray: Array of shape (num_words, hidden_size) with word-level embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()

    word_embeddings = []

    # Assumes each sentence is grouped by a unique sentence_id
    for sentence_id, group in df_contextual.groupby("sentence_id"):
        words = group["word"].tolist()
        word_indices = group["word_idx"].tolist()

        # Tokenize the full sentence
        sentence_text = " ".join(words)
        inputs = tokenizer(sentence_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            hidden_states = outputs.hidden_states  # tuple of (13,) including embeddings + 12 layers
            layer_output = hidden_states[data_params.embedding_layer].squeeze(0)  # [seq_len, hidden_dim]

        # Align subwords to words
        aligned_word_embeddings = []
        subword_idx = 0
        for idx in range(len(word_indices)):
            word = words[idx]
            tokenized = tokenizer.tokenize(word)
            num_subtokens = len(tokenized)

            if subword_idx + num_subtokens > layer_output.shape[0]:
                break  # Edge case: tokenizer may produce more subtokens than model returned

            word_emb = layer_output[subword_idx:subword_idx + num_subtokens].mean(0)
            aligned_word_embeddings.append(word_emb.cpu().numpy())
            subword_idx += num_subtokens

        word_embeddings.extend(aligned_word_embeddings)

    return np.stack(word_embeddings)
