# NanoGPT Extensions and Fine-tuning

## Objective:
Extend the NanoGPT codebase to 
- Visualise token probabilities and sampling decisions during text generation.
- Compute probabilities for entire generated sequences.
- Support optional fixed responses.

Additionally, fine-tune GPT-2 on a domain-specific corpus to evaluate how the model's probabilities for fixed prompt-response pairs change after fine-tuning.

## Data
- `eval_data.json` - Sample prompt-response pairs used to evaluate the fine-tuned model.
- Children Stories Text Corpus: a domain-specific corpus used for fine-tuning GPT-2
  - Not included in this repository due to size constraints. Download from: https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus

## Structure:
- `model.py` - Modified `generate()` to support sequence probability computation and optional fixed responses.
- `sample.py` - Added top-10 token probability visualisations, with the actually sampled token highlighted.
- `eval.py` - Iterates through `eval_data.json` and computes the probability assigned to each fixed response using `generate(fixed_response=...)`
- `eval_data.json` - Sample prompt-response pairs for evaluation.

## Methods:
- *Token Probability Analysis*: Modified `sample.py` to visualise the top-10 token probabilities at each generation step.
- *Sequence Probability Computation*: Modified `generate()` returns the probability of an entire token sequence, calculated using log probabilities for numerical stability.
- *Fixed Response Evaluation*: `generate()` optionally accepts the argument `fixed_response` to generate and compute the probability for a specified sequence.
- *Fine-tuning:* GPT-2 is fine-tuned on the Children Stories corpus using `train.py`. `eval.py` uses the same fixed prompt-response pairs to compare probabilities before and after fine-tuning.

## Key Results

## How to Run


