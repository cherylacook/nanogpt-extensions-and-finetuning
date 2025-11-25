# Copied from sample.py
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
# Added for eval.py
import json

# Copied from sample.py
# -----------------------------------------------------------------------------
show_probs = False # for Task 1.1
fixed_response = "" # for Task 1.3
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# Device setup
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    # loading checkpoint to CPU to avoid issues with MPS
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Deviation from sample.py starts here

# Eliminated meta.pkl case because both 'init=resume' (my fine-tuned GPT2 model) and 'init=gpt2' use GPT-2 BPE tokens
print("Using GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def eval(eval_file="eval_data.json"):
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)
        #print(type(eval_pairs))
        #print(type(eval_pairs[0]))

    for pr_pair in eval_pairs:
        prompt = pr_pair["prompt"]
        response = pr_pair["response"]

        # Encoding the prompt and response into tokens
        prompt_tokens = encode(prompt)
        #print(type(prompt_tokens))
        response_tokens = encode(response)
        
        # Converting the prompt tokens to a tensor (proper input format for model.generate())
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Generating with response tokens set as fixed_response
        y, probs_list, chosen_tokens, response_prob = model.generate(x, max_new_tokens=len(response_tokens), temperature=temperature, top_k=top_k, fixed_response=response_tokens)       
        print("Prompt:", prompt)
        print("Response:", response)
        #print("Generated sequence:", decode(y[0].tolist()))
        print("Probability of fixed response:", response_prob)

# run generation
eval(eval_file="eval_data.json")


