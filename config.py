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
