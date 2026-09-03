# Negation & Speculation Cue/Scope Detection

A command line Python port of `Multitask_Learning_of_Negation_and_Speculation.ipynb` (a copy of
[adityak6798/Transformers-For-Negation-and-Speculation](https://github.com/adityak6798/Transformers-For-Negation-and-Speculation)). Removed Google Drive related codes and refactored using newer huggingface codes.


### 0 - Layout

| File | Contents |
|---|---|
| `main.py` | CLI entry point: loads datasets, runs the train/eval loop |
| `evaluate_cue.py` | Re-evaluate a saved cue-detection checkpoint per test corpus, tabulating the `f1_cues` P/R/F1 for negation and speculation |
| `evaluate_scope.py` | Re-evaluate a saved scope-resolution checkpoint per test corpus and task, tabulating token macro F1, IN_SCOPE F1 and scope-level F1 |
| `pipeline.py` | `get_cue_and_scope(text)` — loads both best checkpoints once, then runs cue detection and feeds each detected cue to scope resolution |
| `config.py` | Hyperparameters and run configuration (`MAX_LEN`, `EPOCHS`, `CUE_MODEL`, ... and an auto-detected `DEVICE`) |
| `data.py` | `Cues`/`Scopes`/`Data`: BioScope & SFU corpus parsing and `DataLoader` construction |
| `model.py` | `CueModel_Combined`, `CueModel_Separate`, `ScopeModel_Combined`, `ScopeModel_Separate` — the actual task models |
| `metrics.py` | F1/accuracy helpers used during training and evaluation |
| `early_stopping.py` | `EarlyStopping` checkpoint helper |
| `multihead_model.py` | `MultiHeadTokenClassifier` — a dual (negation + speculation) linear head on top of any `transformers.AutoModel` encoder, used for cue detection |

### 1 - Setup

```
pip install -r requirements.txt
```

And here is how data is organized, put it under `data/`. Note that `--sfu` takes the corpus *directory*, not a file where Data class will walk through subdirectories to process files.

```
data/
  abstracts.xml                              # BioScope abstracts (BioScope XML)
  full_papers.xml                            # BioScope full papers (BioScope XML)
  SFU_Review_Corpus_Negation_Speculation/    # SFU Review corpus root directory
    BOOKS/*.xml  CARS/*.xml  COMPUTERS/*.xml  COOKWARE/*.xml
    HOTELS/*.xml MOVIES/*.xml MUSIC/*.xml     PHONES/*.xml
```

### 2 - Run Finetuning (if want to train again)

Two models, one command each:

```
python main.py --subtask cue_detection
python main.py --subtask scope_resolution
```

`config.py` is already set to the configuration (model, etc.) each subtask should use (see
below), so there is nothing to edit. You can also specify your paths manually:

```
python main.py --subtask cue_detection \
    --bioscope-full-papers path/full_papers.xml \
    --bioscope-abstracts path/abstracts.xml \
    --sfu path/SFU_Review_Corpus_Negation_Speculation
```

The two subtasks are independent — scope resolution trains on the gold cue
annotations that ship with the corpora, not on anything the cue model predicts
— so you can run both at once in two processes. During inference, you might need to produce cues from cue model first then format it to feed into scope resolution.

The configurations are taken from Khandelwal & Britto (2020), *Multitask Learning of Negation
and Speculation using Transformers*, LOUHI @ EMNLP 2020
([doi](https://doi.org/10.18653/v1/2020.louhi-1.9)). §5.4 reports XLNet as the
best backbone, combined early stopping as the better of the two schemes, and
global as the better cue-preprocessing method. 

### 3 - Comparing against the paper

We reproduce the Multitask Learning setup of Khandelwal & Britto (2020) — XLNet backbone, joint negation/speculation training, Combined early stopping, trained on BioScope Full Papers + BioScope Abstracts + SFU (BF+BA+SFU) — and compare our results against their reported figures.

**Cue Detection** (F1 over cue classes vs. NOT_CUE, per Table 3/5 of the paper):

| Test corpus | Task | Paper (XLNet, BF+BA+SFU) | Ours |
|---|---|---|---|
| BioScope Full Papers | Negation | ~94.3 | **96.84** |
| BioScope Abstracts | Negation | ~96.1 | **97.88** |
| SFU | Negation | ~74–75 | **96.01** |
| BioScope Full Papers | Speculation | — | 94.70 |
| BioScope Abstracts | Speculation | — | 97.66 |
| SFU | Speculation | ~85–86 | 97.20 |

**Scope Resolution** (Token-level Macro F1, per Table 6/9, Global preprocessing):

| Test corpus | Task | Paper (XLNet, BF+BA+SFU, Global) | Ours |
|---|---|---|---|
| BioScope Full Papers | Negation | ~94–95 | **99.57** |
| BioScope Abstracts | Negation | ~99.3 | 99.31 |
| SFU | Negation | ~91.2 | 96.36 |
| BioScope Full Papers | Speculation | — | 97.96 |
| BioScope Abstracts | Speculation | ~98.3 | 98.63 |
| SFU | Speculation | ~91.3 | 97.07 |

Our results are consistent with or exceed the original paper across both tasks and all three test corpora, with the largest margin on SFU — the out-of-domain, informal-text corpus that the original authors identify as the hardest setting for cue detection. Paper figures above are approximate, reconstructed from the published tables; run-to-run variance (the original paper reports results averaged over 3 runs with per-run re-randomized splits) should be taken into account before treating small differences as meaningful.

### 4 - Re-evaluating a finished run

`evaluate_cue.py` and `evaluate_scope.py` load one saved checkpoint and run
the same `evaluate()` per test corpus, collecting the metrics into a table (and
a `--csv`) instead of leaving them scattered through a training log:

```
python evaluate_cue.py   --checkpoint check_pts/cue_detection/xlnet-base-cased_BF+BA+SFU_combined/run2/best.pt
python evaluate_scope.py --checkpoint check_pts/scope_resolution/xlnet-base-cased_BF+BA+SFU_global_combined/run3/best
```

The backbone, corpora, scope method and early-stopping scheme are read out of
the checkpoint path; `--model`, `--scope-method`, `--train-datasets` and
`--early-stopping` override them.

To try a different backbone or preprocessing scheme, set `MODEL=` or
`SCOPE_METHOD=` in the environment (`data.py` and `model.py` bind those at
import time, so a command-line flag would be read too late). Training corpora
and early stopping are `--train-datasets` and `--early-stopping`.

### 5 - Checkpoints

Checkpoints are written to `./check_pts/` by default: one every 10 epochs,
plus the best-validation weights that early stopping tracks. Override with
`--checkpoint-dir` / `--checkpoint-every` (`0` = only keep the best), or edit
`CHECKPOINT_DIR` / `CHECKPOINT_EVERY` in `config.py`. Each run gets its own
directory, named for the configuration that produced it, so the two subtasks
(and any config you try later) never overwrite each other:

```
check_pts/cue_detection/xlnet-base-cased_BF+BA_combined/run1/
check_pts/scope_resolution/xlnet-base-cased_BF+BA_global_combined/run1/
```

The format depends on the model:

- **Scope resolution** models are plain HuggingFace models, so they are
  saved as HuggingFace folders (config + weights + tokenizer, with
  `IN_SCOPE` / `OUT_OF_SCOPE` in `id2label`).
- **Cue detection** models are the custom two-headed
  `MultiHeadTokenClassifier` (one shared encoder, a negation head and a
  speculation head), which HuggingFace cannot load, so they are saved as
  PyTorch `state_dict` files.

```
inside a run directory ........... scope resolution         cue detection
  every N epochs ................. epoch_010/  epoch_020/   epoch_010.pt  epoch_020.pt
  best validation F1 ............. best.pt + best/          best.pt
```

`best*.pt` is what gets reloaded for the final test evaluation, exactly as
the notebook did with its `checkpoint.pt`. Every file/folder is a full copy
of the weights (~500 MB for `bert-base` / `roberta-base`).

### 6 - Using the trained models

This section is about running the trained models on **new text**. To score one
of the corpora, don't hand-roll it — `evaluate_cue.py` / `evaluate_scope.py`
(section 4) already build the dataloaders, load the checkpoint and report the
metrics.

You can use `def get_cue_and_scope(text, words=None, debug = False)` function from `pipeline.py` and `def format_result(result)` to format it. For specific getting cue and scope can refer to funcions in file `pipeline.py`.

```Python
result = get_cue_and_scope(
    "They analyzed 146 prokaryotic genomes, but no likely tRNA of the novel amino acid was detected."
)
print(format_result(result))
```

**The returned dict, in full**

`get_cue_and_scope` returns one dict per sentence, shaped like this, and `format_result` simply parses it and returns everything:

```
{
  "text":  str,                          # the input sentence, unchanged
  "words": [str, ...],                   # tokenized words -- the index every
                                          # other field below is keyed against
  "cue_label_ids": {
    "negation":    [int, ...],           # one label id per word (len == len(words))
    "speculation": [int, ...],
  },
  "cue_labels": {
    "negation":    [str, ...],           # cue_label_ids, mapped to names
    "speculation": [str, ...],
  },
  "cues": [
    {
      "task":            "negation" | "speculation",
      "cue_type":        "CUE" | "MULTIWORD_CUE" | "AFFIX_CUE",
      "cue_indices":     [int, ...],      # word index/indices making up this cue
      "cue_words":       [str, ...],      # words at those indices
      "scope_label_ids": [int, ...],      # 0/1 per word (len == len(words))
      "scope_labels":    [str, ...],      # scope_label_ids, mapped to names
      "scope_indices":   [int, ...],      # word indices predicted IN_SCOPE
      "scope_words":     [str, ...],      # words at those indices
    },
    ...  # one entry per cue found, possibly zero
  ],
}
```

**Label meanings**, for reference:
 
| | id | name |
|---|---|---|
| Cue labels | `3` | `NOT_CUE` |
| | `1` | `CUE` (single-word) |
| | `2` | `MULTIWORD_CUE` |
| Scope labels | `0` | `OUT_OF_SCOPE` |
| | `1` | `IN_SCOPE` |