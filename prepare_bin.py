# Preparing corpus data with GPT-2 BPE tokenizer
import numpy as np
import pickle
import tiktoken

input_file = "childrens_stories.txt"
with open(input_file, "r", encoding="utf-8") as file:
    input_data = file.read()

# Creating GPT-2 BPE tokenizer and encoding input data into tokens
enc = tiktoken.get_encoding("gpt2")
input_tokens = enc.encode(input_data)
print("Total tokens:", len(input_tokens))

# 80-20 split  of input tokens into training and validation sets
num_tokens = len(input_tokens)
# Taking the first 80% of tokens for training set
train_set_ids = np.array(input_tokens[:int(num_tokens*0.8)], dtype=np.uint16)
# Taking the last 20% of tokens for validation set
val_set_ids = np.array(input_tokens[int(num_tokens*0.8):], dtype=np.uint16)

# Saving the token ids to binary files for train.py
train_set_ids.tofile("train.bin")
val_set_ids.tofile("val.bin")

# Meta.pkl information
"""
meta_file = {
    "vocab_size": enc.n_vocab, 
    "encoding": "gpt2-bpe"
}

with open("meta.pkl", "wb") as file:
    pickle.dump(meta_file, file)
"""

print("Successfully created train.bin, val.bin, meta.pkl (GPT-2 BPE mode)")
