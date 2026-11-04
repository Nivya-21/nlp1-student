# NLP1 — Project 1

Classify WildGuardMix prompts as harmful or unharmful, excluding the agreed
`toxic_language_hate_speech` category.

Open the notebooks in this order:

1. [Setup](part1/setup.ipynb) — install dependencies and check Hugging Face access.
2. [Project 1](part1/project1.ipynb) — prepare data, train models and analyse results.

The project compares indicators, token counts and TF-IDF with logistic regression,
a one-hidden-layer MLP and a two-hidden-layer MLP. All nine systems use the same
five stratified folds. The notebook includes macro-F1, error analysis and paired
Bayesian comparisons with ROPE ±0.01. The official test remains held out.

## Run locally

Use Python 3.14. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab part1/setup.ipynb
```

Select the `.venv` kernel. Accept the dataset conditions on
[Hugging Face](https://huggingface.co/datasets/allenai/wildguardmix), then run
`hf auth login` in the activated environment. Each teammate uses their own login.

Run Project 1 from top to bottom. The settings cell controls sampling, BPE and
training. The current defaults are 2,000 prompts per class, 3,000 BPE tokens and
10 epochs. Set `use_toy_data = True` for a quick offline check; its artificial
topic labels do not measure harmfulness classification.

Each run saves metrics, hashed-ID predictions, settings and audit counts in a new
folder under `part1/results/`. These local outputs and caches are ignored by Git.
Clear notebook outputs before committing. Raw prompts and credentials should
not be committed.

## Code and attribution

The workflow stays in notebook code cells. Repeated project operations live in
[utils/p1/functions.py](utils/p1/functions.py); the Python modules under `part1/`
are reused course code.

This repository retains the history and Apache-2.0 license of
[nlp-illc/nlp1-student](https://github.com/nlp-illc/nlp1-student), based on commit
`6d93c951fa2bd0a92b6e90ca954865f1c1165eef`. The original
[tutorial](https://github.com/nlp-illc/nlp1-student/blob/6d93c951fa2bd0a92b6e90ca954865f1c1165eef/part1/tutorial.ipynb)
and [assignment](https://github.com/nlp-illc/nlp1-student/blob/6d93c951fa2bd0a92b6e90ca954865f1c1165eef/part1/project.ipynb)
remain available upstream. `main` contains the team version; exploratory notebooks
are maintained separately.
