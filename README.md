# NanoGPT Extensions and Fine-tuning

## Objective
Extend the NanoGPT codebase to support:
- Visualisation of token probabilities and sampling decisions during text generation.
- Computation of probabilities for entire generated sequences.
- Optional fixed responses.

Additionally, fine-tune GPT-2 on a domain-specific corpus to examine how computed probabilities for fixed prompt-response pairs change after fine-tuning.

## Data
- `eval_data.json` - 10 curated prompt-response pairs for probability evaluation.
- Children Stories Text Corpus - Domain-specific dataset used for fine-tuning GPT-2
  - childrens_stories.txt - Raw text corpus.
  - `prepare_bin.py` - Converts the corpus into `train.bin` and `val.bin` for training in `train.py`.
  - `train.bin` and `val.bin` - Preprocessed binaries included for convenience and reproducibility.
  - `ckpt.pt` - Fine-tuned model checkpoint; not included due to repository size constraints.
     - Download here: https://huggingface.co/datasets/cherac/finetuned_gpt2/resolve/main/ckpt.pt 
 
## Structure
- `model.py` - Modified `generate()` to compute sequence probabilities and support fixed responses.
- `sample.py` - Added token probability visualisations and ability to supply a fixed response.
- `eval.py` - Evaluates GPT-2 on the fixed prompt-response pairs and prints computed sequence probabilities.
- `childrens_stories.txt` - Raw text corpus.
- `prepare_bin.py` - Preprocesses the raw corpus into binary training files.
- `train.bin` and `val.bin` - Preprocessed training and validation sets.
- `train.py` - Modified hyperparameters for fine-tuning on a small dataset.
- `eval_data.json` - Curated evaluation prompt-response pairs.
- `results/` - Folder containing token probability plots and full fine-tuning comparison table.
- `requirements.txt` - Python dependencies.
**Note**: This project extends [NanoGPT](https://github.com/karpathy/nanoGPT). Clone NanoGPT first to ensure full compatibility, then copy over the modified scripts and data files.

## Methods
1. *Token Probability Analysis* - Visualises the top-10 token probabilities at each generation step (`--show_probs=True`), with the selected token highlighted.
2. *Sequence Probability Computation* - Computes full sequence probabilities using log probabilities for numerical stability.
3. *Fixed Response Evaluation* - Computes the probability of a specific continuation via `--fixed_response`
4. *Fine-tuning* - GPT-2 is fine-tuned on the Children Stories corpus using `train.py` with adjusted hyperparameters to accommodate a small dataset. Pre‑ and post‑tuning evaluations use the curated prompt–response set.

## Key Results
Probability changes after fine-tuning (snippet):
| Prompt                               | Response                        | Pre-tuning Prob | Post-tuning Prob | Change |
|--------------------------------------|---------------------------------|-----------------|------------------|--------|
| Once upon a time                     | there was a king                | 4.31e-11        | 7.72e-08         | ↑      |
| Long, long ago                       | two boys lived in a village     | 3.78e-16        | 8.38e-14         | ↑      |
| she was so pretty that they thought  | she must be some fairy princess | 8.84e-16        | 1.64e-11         | ↑      |

The full comparison table is available in `results/`.
Fine-tuning consistently increased probabilities for domain-appropriate narrative continuations, indicating alignment with the training corpus.

## How to Run:
Requirements: Python 3.10+, PyTorch, NumPy, Matplotlib
```bash
# Install dependencies
pip install -r requirements.txt
# Sample generation with probability visualisation
python sample.py --init_from=gpt2 --start "Once upon a time" --num_samples 1 --max_new_tokens 10 --show_probs True
# To train GPT-2 on the provided train.bin and val.bin (--device=[cpu or gpu or mps])
python train.py --device=cpu
# Evaluate fixed responses on base GPT-2 
python eval.py --init_from=gpt2 --device=cpu
# Evaluate fixed responses on fine-tuned GPT-2 (requires `ckpt.pt` in `out/`)
python eval.py --init_from=resume --device=cpu
```
**Notes:**
- Ensure `ckpt.pt` is placed in `out/` if using `--init_from=resume`.

## Summary:
This project demonstrates
- Practical extensions to GPT-2 for token-level and sequence-level probability inspection.
- Fine‑tuning on a domain‑specific corpus and evaluation of resulting probability shifts.
- Clear evidence that fine‑tuning encourages the model to assign higher probability to corpus‑consistent narrative continuations.

## Reproducibility / Notes
- `train.bin` and `val.bin` are included; no preprocessing is required to run train.py.
- Fine-tuned checkpoint (ckpt.pt) is external due to size constraints, but training can be fully reproduced with `train.py`.
- Code modifications are minimal and documented inline for clarity.
- This repo **depends on the NanoGPT repository structure**, so ensure NanoGPT is cloned before running.
