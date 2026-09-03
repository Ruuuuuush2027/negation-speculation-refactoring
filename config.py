"""Hyperparameters and run configuration."""
import torch

MAX_LEN = 128
bs = 8
EPOCHS = 60
PATIENCE = 6
INITIAL_LEARNING_RATE = 3e-5
NUM_RUNS = 3 #Number of times to run the training and evaluation code

CUE_MODEL = 'roberta-base'
SCOPE_MODEL = 'bert-base-uncased'
SCOPE_METHOD = 'local' # Options: global, local
EARLY_STOPPING_METHOD = 'combined' # Options: combined, separate
ERROR_ANALYSIS_FOR_SCOPE = False # Options: True, False
SUBTASK = 'scope_resolution' # Options: cue_detection, scope_resolution
TRAIN_DATASETS = ['bioscope_full_papers']
TEST_DATASETS = ['bioscope_full_papers','bioscope_abstracts','sfu']

# Automatically use a GPU if one is available locally, otherwise fall back to
# CPU. The original notebook assumed a Colab GPU was always present.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Where training writes checkpoints. Each run gets its own sub-directory
# (run1/, run2/, ...). Scope models are plain HuggingFace models and are saved
# as HuggingFace folders; cue models (custom two-headed MultiHeadTokenClassifier)
# are saved as state_dict .pt files. See README "Checkpoints" for loading.
CHECKPOINT_DIR = 'check_pts'
# Save a checkpoint every N epochs (0 disables periodic checkpoints; the
# best-validation weights are always saved by early stopping).
CHECKPOINT_EVERY = 10

# Human-readable label names. SCOPE_LABELS is written into the saved scope
# model's config (id2label); CUE_LABELS is for decoding cue-model output.
# Cue labels follow the notebook's scheme; 4 is the padding label (weight 0).
CUE_LABELS = {0: "AFFIX_CUE", 1: "CUE", 2: "MULTIWORD_CUE", 3: "NOT_CUE", 4: "PAD"}
SCOPE_LABELS = {0: "OUT_OF_SCOPE", 1: "IN_SCOPE"}
