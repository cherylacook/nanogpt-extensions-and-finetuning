# NanoGPT Extensions and Fine-tuning

## Objective
Extend the NanoGPT codebase to:
- Visualise token probabilities and sampling decisions during text generation.
- Compute probabilities for entire generated sequences.
- Support optional fixed responses.

Additionally, fine-tune GPT-2 on a domain-specific corpus to examine how computed probabilities for fixed prompt-response paris change after fine-tuning.

## Data
- `eval_data.json` - 10 curated prompt-response pairs used to evaluate the fine-tuned model.
- Children Stories Text Corpus - Domain-specific dataset used for fine-tuning GPT-2
  - Not included in the repo due to size constraints
  - Download from: https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus
 
## Structure
- `model.py` - Modified `generate()` to support sequence probabilities and fixed responses.
- `sample.py` - Added flags to support fixed responses and creating top-10 token probability plots for each generation step.
- `eval.py` - Evaluates GPT-2 on fixed prompt-response pairs and prints computed sequence probabilities.
- `train.py` - Modified hyperparameters to support training on a small corpus.
- `eval_data.json` - Curated evaluation pairs.
- `results/` - Folder containing fine-tuning results.

## Methods
1. *Token Probability Analysis* - If `--show_probs=True`, the top-10 token probabilities at each generation step are visualised, with the selected token highlighted.
2. *Sequence Probability Computation* - Computes full sequence probabilities using log probabilities for numerical stability.
3. *Fixed Response Evaluation* - By passing an argument for `--fixed_response`, the probability of a specific continuation can be computed.
4. *Fine-tuning* - GPT-2 fine-tuned on the Children Stories corpus, with the `train.bin` and `val.bin` necessary for `train.py` generated in `prepare_bin.py`. Fine-tuned model's checkpoint is `ckpt.pt`.

## Key Results
Snippet of probability changes after fine-tuning:
| Prompt                               | Response                        | Pre-tuning Prob | Post-tuning Prob | Change |
|--------------------------------------|---------------------------------|-----------------|------------------|--------|
| Once upon a time                     | there was a king                | 4.31e-11        | 7.72e-08         | ↑      |
| Long, long ago                       | two boys lived in a village     | 3.78e-16        | 8.38e-14         | ↑      |
| she was so pretty that they thought  | she must be some fairy princess | 8.84e-16        | 1.64e-11         | ↑      |

Full table available in `results/` for reference.
Fine-tuning consistently incresed the model's probability assignments for domain-appropriate continuations, illustrating alignment with the corpus.

## How to Run:
Python 3.10+ recommended
```bash
pip install -r requirements.txt
# Run sample generation with probability visualisation
python sample.py --init_from=gpt2 --start "Once upon a time" --num_samples 1 --max_new_tokens 10 --show_probs True
# Evaluate fixed responses on GPT-2
python eval.py --init_from=gpt_2
# Evalute fixed responses on fine-tuned GPT-2
python eval.py --init_from=resume # utilises ckpt.pt
```

## Summary:
This project highlights:
- Practical modifications to GPT-2 for detailed probabilistic analysis.
- Domain adaptation through fine-tuning on a curated corpus.

## Reproducibility / Notes


