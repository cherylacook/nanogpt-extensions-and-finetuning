# Prepares the Children's Stories corpus for train.py by using the GPT-2 BPE tokenizer to produce train.bin and val.bin.

import numpy as np
import tiktoken

# Configuration
input_file = "childrens_stories.txt" # raw text corpus
train_file = "train.bin"
val_file = "val.bin"
train_fraction = 0.8 # fraction of data for training set

# Loading the input text
with open(input_file, "r", encoding="utf-8") as file:
    input_data = file.read()

# Encoding the text using the GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
input_tokens = enc.encode(input_data)
print("Total tokens in corpus:", len(input_tokens))

# 80-20 split of input tokens into training and validation sets
num_tokens = len(input_tokens)
split_idx = int(num_tokens*train_fraction)

# Taking the first 80% of tokens for training set
train_tokens = np.array(input_tokens[:split_idx], dtype=np.uint16)
# Taking the last 20% of tokens for validation set
val_tokens = np.array(input_tokens[split_idx:], dtype=np.uint16)

# Saving the token ids to binary files for train.py
train_tokens.tofile(train_file)
val_tokens.tofile(val_file)

print(f"Saved {len(train_tokens)} tokens to {train_file}")
print(f"Saved {len(val_tokens)} tokens to {val_file}")
print("Preparation complete. Files are ready for train.py")
