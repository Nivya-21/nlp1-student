# NLP1 — Project 1

Open these notebooks in order:

1. [part1/setup.ipynb](part1/setup.ipynb): install dependencies and check Hugging Face access. Run this once with the project's Python kernel.
2. [part1/project1.ipynb](part1/project1.ipynb): the full experiment, explained step by step like the course tutorial.
3. [part1/project1_experiments.ipynb](part1/project1_experiments.ipynb): a separate exploratory copy comparing 1,500, 4,000 and 8,000 BPE tokens, plus single-setting MLP controls. It explains tokenization versus the three representations and preserves the original notebook/results.

To learn by making your own choices, open
[part1/project1_learning.ipynb](part1/project1_learning.ipynb). Edit its control
panel, write a hypothesis, save the notebook, then restart the kernel and run all.
It starts with one model on one fold. Switch `run_all_folds` to `True` for CV;
each saved run gets a separate folder under `part1/results/learning/`.
The notebook includes blank reflection prompts, a history of comparable runs,
and an optional paired Bayesian comparison. It does not automatically tune models.

The experiment notebook contains data filtering, BPE, the three representations,
the three models, five-fold cross-validation, Bayesian ROPE and error analysis.
Reusable encoding and training helpers live in [utils/p1/functions.py](utils/p1/functions.py).
There is no separate runner or YAML configuration. Change the settings in the notebook.
The extension runs 150 cross-validation fits on the same sample and folds, saving
separate outputs under `part1/results/project1_experiments/`. Its rankings are
exploratory; the official test is not used to select a configuration.

The original [tutorial](part1/tutorial.ipynb), [assignment](part1/project.ipynb),
course modules and Apache-2.0 license are preserved. The Git history comes from
[nlp-illc/nlp1-student](https://github.com/nlp-illc/nlp1-student), starting at
`6d93c951fa2bd0a92b6e90ca954865f1c1165eef`.

Use Python 3.14 and select the `.venv` kernel. The environment on this Mac is
already set up. From the repository root:

```bash
source .venv/bin/activate
jupyter lab part1/project1.ipynb
```

Real-data access requires accepting the WildGuardMix terms and `hf auth login`.
The notebook uses the existing login without displaying its token. For a quick
offline check, set `use_toy_data = True` in its settings cell. Toy results are
labelled `SMOKE_NOT_SCIENTIFIC`. Normal results use only filtered training data;
the official test remains untouched.

Results are generated locally under `part1/results/`. Raw prompts, credentials
and notebook outputs should not be committed. AI assistance is disclosed in
the notebook. Nothing has been published to GitHub; the current `origin` is
still the course repository and must not be used for a push.
