# NanoGPT Extensions and Fine-tuning

## Objective
Extend the NanoGPT codebase to:
- Visualise token probabilities and sampling decisions during text generation.
- Compute probabilities for entire generated sequences.
- Support optional fixed responses.

Additionally, fine-tune GPT-2 on a domain-specific corpus to examine how computed probabilities for fixed prompt-response pairs change after fine-tuning.

## Data
- `eval_data.json` - 10 curated prompt-response pairs for evaluation.
- Children Stories Text Corpus - Domain-specific dataset used for fine-tuning GPT-2
  - Raw text (childrens_stories.txt) not included, as it is too large for the repository; download here: _
  - `prepare_bin.py` - converts the corpus into `train.bin` and `val.bin` for training in `train.py`
  - `train.bin` and `val.bin` - preprocessed binaries included for convenience.
 
## Structure
- `model.py` - Modified `generate()` that supports sequence probabilities and fixed responses.
- `sample.py` - Modified to support token probability visualisations and supplying of fixed responses.
- `eval.py` - Evaluates GPT-2 on fixed prompt-response pairs and prints computed sequence probabilities.
- `prepare_bin.py` - Converts raw text corpus to binary train and val files.
- `train.bin` and `val.bin` - Preprocessed training and validation sets.
- `train.py` - Modified hyperparameters to support training on the small, domain-specific corpus.
- `eval_data.json` - Curated evaluation prompt-response pairs.
- `results/` - Folder containing token probability plots and fine-tuning comparison table.
- `requirements.txt` - Python dependencies.

## Methods
1. *Token Probability Analysis* - Visualise the top-10 token probabilities at each generation step (if `--show_probs=True`), with the selected token highlighted.
2. *Sequence Probability Computation* - Compute full sequence probabilities using log probabilities for numerical stability.
3. *Fixed Response Evaluation* - Compute the probability of a specific continuation by passing `--fixed_response`
4. *Fine-tuning* - GPT-2 fine-tuned on the Children Stories corpus using `train.py` with adjusted hyperparameters to accommodate a small dataset. Evaluation before and after fine-tuning uses the curated prompt-response pairs, with results available in `results/`.

## Key Results
Probability changes after fine-tuning (Snippet):
| Prompt                               | Response                        | Pre-tuning Prob | Post-tuning Prob | Change |
|--------------------------------------|---------------------------------|-----------------|------------------|--------|
| Once upon a time                     | there was a king                | 4.31e-11        | 7.72e-08         | ↑      |
| Long, long ago                       | two boys lived in a village     | 3.78e-16        | 8.38e-14         | ↑      |
| she was so pretty that they thought  | she must be some fairy princess | 8.84e-16        | 1.64e-11         | ↑      |

Full table available in `results/` for reference.
Fine-tuning consistently increased probabilities for domain-appropriate continuations, illustrating alignment with the corpus.

## How to Run:
Requirements: Python 3.10+, PyTorch, NumPy, Matplotlib
```bash
# Install dependencies
pip install -r requirements.txt
# Run sample generation with probability visualisation
python sample.py --init_from=gpt2 --start "Once upon a time" --num_samples 1 --max_new_tokens 10 --show_probs True
# Evaluate fixed responses on GPT-2
python eval.py --init_from=gpt_2
# Evaluate fixed responses on fine-tuned GPT-2
python eval.py --init_from=resume # utilises ckpt.pt
```

## Summary:
This project demonstrates
- Practical extensions to GPT-2 for token-level and sequence-level probability analysis.
- Domain adaptation through fine-tuning on a curated corpus.
- How probability assignments for fixed sequences change after fine-tuning, illustrating model alignment with human narrative expectations.

## Reproducibility / Notes

