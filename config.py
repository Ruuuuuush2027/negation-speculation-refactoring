"""Hyperparameters and run configuration.

Defaults follow the experimental protocol of the paper this repo replicates:

    Aditya Khandelwal & Benita Kathleen Britto (2020),
    "Multitask Learning of Negation and Speculation using Transformers",
    LOUHI @ EMNLP 2020.  https://doi.org/10.18653/v1/2020.louhi-1.9

See README "Replicating the paper" for the full experiment grid and for which
table each setting corresponds to.
"""
import os

import torch


def _env(name, default):
    """Read an override from the environment, falling back to `default`.

    data.py and model.py do `from config import X`, which binds the value at
    import time, so these knobs cannot be overridden by a command-line flag in
    main.py -- by the time argparse runs, the value is already baked in. The
    environment is read here, before any of that happens, so it works:

        MODEL=xlnet-base-cased SCOPE_METHOD=global python main.py

    Knobs that only main.py reads (subtask, train/test datasets, number of
    runs, early-stopping method) are command-line flags instead.
    """
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Paper protocol (Section 4, "Experimentation Details")
# ---------------------------------------------------------------------------
# "We use a 70-15-15 train-dev-test split."
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# "We perform an early stopping (with a patience of 6) on the validation F1
# Score."
PATIENCE = 6

# The three backbones the paper reports (the "Model" column of every table).
# A given table row uses ONE of these for whichever subtask it reports, so
# CUE_MODEL and SCOPE_MODEL should be the same when replicating a row.
PAPER_MODELS = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'xlnet': 'xlnet-base-cased',
}

# Every model is tested on all three corpora regardless of what it trained on
# (the "Test Dataset" rows of every table).
PAPER_TEST_DATASETS = ['bioscope_full_papers', 'bioscope_abstracts', 'sfu']


def runs_for(train_datasets):
    """How many runs the paper averages over, given the training corpora.

    "The results are reported as an average of 5 runs for training on a single
    dataset and an average of 3 runs for training on a combination of multiple
    datasets."
    """
    return 5 if len(train_datasets) == 1 else 3


# ---------------------------------------------------------------------------
# What this invocation runs
# ---------------------------------------------------------------------------
# Backbone: XLNet for both subtasks, the paper's best (Table 5, Tables 8/11,
# and Section 5.4: it "consistently outperforms RoBERTa and BERT"). Set MODEL
# to change both at once, or CUE_MODEL / SCOPE_MODEL individually.
#
# Scope resolution marks cue words with BERT's reserved [unused*] tokens. XLNet
# and RoBERTa vocabularies do not contain them, so under those backbones every
# marker maps to <unk> (for XLNet: id 0, which the attention mask then hides
# from the other tokens). This is exactly how the original notebook -- and so
# the paper -- ran XLNet scope resolution, and it is left as-is on purpose so
# that results stay comparable. See README "The two configurations".
MODEL = _env('MODEL', None)
CUE_MODEL = _env('CUE_MODEL', MODEL or PAPER_MODELS['xlnet'])
SCOPE_MODEL = _env('SCOPE_MODEL', MODEL or PAPER_MODELS['xlnet'])

# Cue-marker preprocessing scheme (paper Section 3.2). Section 5.4: "the global
# preprocessing method outperforms the local preprocessing method" (Table 12).
SCOPE_METHOD = _env('SCOPE_METHOD', 'global')  # Options: global, local

# These are the defaults main.py's --subtask / --train-datasets /
# --test-datasets / --early-stopping flags start from; override per run on the
# command line rather than editing them here.
SUBTASK = 'scope_resolution'  # Options: cue_detection, scope_resolution
# Section 5.4: "The Combined Early Stopping training method outperform[s] the
# Separate Early Stopping training method."
EARLY_STOPPING_METHOD = 'combined'  # Options: combined, separate
# BF+BA is the training combination behind the paper's best BioScope results
# for both subtasks (Tables 5 and 8).
TRAIN_DATASETS = ['bioscope_full_papers', 'bioscope_abstracts', 'sfu']
TEST_DATASETS = list(PAPER_TEST_DATASETS)
NUM_RUNS = runs_for(TRAIN_DATASETS)

ERROR_ANALYSIS_FOR_SCOPE = False  # Options: True, False -- not part of the paper

# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
# The paper does not state these; they are inherited from the NegBERT codebase
# (Khandelwal & Sawant, 2020) that this model code descends from.
MAX_LEN = int(_env('MAX_LEN', 128))
bs = int(_env('BS', 8))
INITIAL_LEARNING_RATE = float(_env('INITIAL_LEARNING_RATE', 3e-5))
# Upper bound only -- early stopping on validation F1 ends training well before
# this in practice.
EPOCHS = int(_env('EPOCHS', 60))

# Automatically use a GPU if one is available locally, otherwise fall back to
# CPU. The original notebook assumed a Colab GPU was always present.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
# Each run writes to <CHECKPOINT_DIR>/<subtask>/run<N>/. Scope models are plain
# HuggingFace models and are saved as HuggingFace folders; cue models (custom
# two-headed MultiHeadTokenClassifier) are saved as state_dict .pt files.
# See README "Checkpoints" for loading.
CHECKPOINT_DIR = 'check_pts'
# Save a checkpoint every N epochs (0 disables periodic checkpoints; the
# best-validation weights are always saved by early stopping).
CHECKPOINT_EVERY = 10

# Human-readable label names. SCOPE_LABELS is written into the saved scope
# model's config (id2label); CUE_LABELS is for decoding cue-model output.
# Cue labels follow the notebook's scheme; 4 is the padding label (weight 0).
CUE_LABELS = {0: "AFFIX_CUE", 1: "CUE", 2: "MULTIWORD_CUE", 3: "NOT_CUE", 4: "PAD"}
SCOPE_LABELS = {0: "OUT_OF_SCOPE", 1: "IN_SCOPE"}
