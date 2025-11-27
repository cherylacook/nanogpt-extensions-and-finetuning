# NanoGPT Extensions and Fine-tuning

**Note**: This project was completed as part of the AIML332 course at Te Herenga Waka — Victoria University of Wellington.

## Objective
Extend the NanoGPT codebase to support:
- Visualisation of token probabilities and sampling decisions during text generation.
- Computation of probabilities for entire generated sequences.
- Optional fixed responses.

Additionally, fine-tune GPT-2 on a domain-specific corpus to examine how computed probabilities for fixed prompt-response pairs change after fine-tuning.

## Data
- `childrens_stories/` - Folder containing the domain-specific dataset used for fine-tuning GPT-2. This folder should be placed under `nanoGPT/data/` when reproducing fine-tuning.
  - `childrens_stories.txt` - Raw text corpus.
  - `prepare_bin.py` - Converts the corpus into `train.bin` and `val.bin` for `train.py`.
  - `train.bin` and `val.bin` - Preprocessed binaries included for convenience and reproducibility.
- `eval_data.json` - 10 curated prompt-response pairs for probability evaluation.
- `ckpt.pt` - Fine-tuned model checkpoint; not included due to repository size limits
     - Download here: https://huggingface.co/datasets/cherac/finetuned_gpt2/resolve/main/ckpt.pt
     - Place in `nanoGPT/out/` *only* if you want to evaluate or resume from the fine-tuned model. Training from scratch with `train.py` does not require `ckpt.pt`.
 
## Structure
**Data**
- `childrens_stories/` - Raw and preprocessed files for fine-tuning.
- `eval_data.json` - Curated evaluation prompt-response pairs.

**Model & Training**
- `model.py` - Modified `generate()` computes sequence probabilities and supports fixed responses.
- `train.py` - Fine-tuning script; automatically reads data from `childrens_stories/`.

**Sampling & Evaluation**
- `sample.py` - Visualises token probabilities and supports fixed responses.
- `eval.py` - Evaluates GPT-2 on the fixed prompt-response pairs, printing computed sequence probabilities.

**Results**
- `results/` - Contains token probability plots and full fine-tuning comparison table.

**Experiments**
- `experiments/` - Contains PDF reports for additional analyses (temperatures and sequence length effects).

**Environment**
- `requirements.txt` - Python dependencies.

**Note**: This project extends [NanoGPT](https://github.com/karpathy/nanoGPT). Clone NanoGPT first to ensure full compatibility, then copy over the modified scripts. Place `childrens_stories/` in `nanoGPT/data/`.

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

**Additional Experiments (Summary):**
- *Effect of temperature*:
  - Temp 0.5 → very deterministic, mostly top tokens.
  - Temp 0.8 → more diversity while favoring likely tokens.
  - Temp 1.5 → high diversity, less grammatical consistency.

- *Effect of sequence length*:
  - Shorter sequences → higher overall sequence probability
  - Longer sequences → lower overall sequence probability 

Full experiment reports, including bar charts for token probabilities, are available in the `experiments/` folder.

## How to Run:
Requirements: Python 3.10+
```bash
# Clone the GitHub repo for nanoGPT
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT 
# Install dependencies
pip install -r requirements.txt
# Sample generation with probability visualisation
python sample.py --init_from=gpt2 --start "Once upon a time" --num_samples 1 --max_new_tokens 10 --show_probs True
# Train GPT-2 on the Children's Stories corpus (folder is auto-detected) (--device=[cpu or gpu or mps])
python train.py --device=cpu
# Evaluate fixed responses on base GPT-2
python eval.py --init_from=gpt2 --device=cpu
# Evaluate fixed responses on fine-tuned GPT-2 (requires `ckpt.pt` in `out/`)
python eval.py --init_from=resume --device=cpu
```
**Notes:**
- Ensure `childrens_stories/` is placed under `nanoGPT/data/` if reproducing fine-tuning with `train.py`.
- Ensure `ckpt.pt` is placed under `nanoGPT/out/` if using `--init_from=resume`. It is not required for training from scratch.

## Summary:
This project demonstrates
- Practical extensions to GPT-2 for token-level and sequence-level probability inspection.
- Fine‑tuning on a domain‑specific corpus and evaluation of resulting probability shifts.
- Clear evidence that fine‑tuning encourages the model to assign higher probability to corpus‑consistent narrative continuations.
- How temperature and sequence length affect sampling behaviour and sequence probability.

## Reproducibility / Notes
- `train.bin` and `val.bin` are included; no preprocessing is required to run `train.py`.
- The fine-tuned checkpoint (ckpt.pt) is external due to size constraints, but training can be fully reproduced with `train.py`.
- Code modifications are minimal and documented inline for clarity.
- This repo **depends on the NanoGPT repository structure**, so ensure NanoGPT is cloned before running.
