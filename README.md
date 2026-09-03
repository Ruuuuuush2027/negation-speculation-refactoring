# Negation & Speculation Cue/Scope Detection

A command line Python port of `Multitask_Learning_of_Negation_and_Speculation.ipynb` (a copy of
[adityak6798/Transformers-For-Negation-and-Speculation](https://github.com/adityak6798/Transformers-For-Negation-and-Speculation)). Removed Google Drive related codes and refactored using newer huggingface codes.


### 0 - Layout

| File | Contents |
|---|---|
| `main.py` | CLI entry point: loads datasets, runs the train/eval loop |
| `evaluate_cue.py` | Re-evaluate a saved cue-detection checkpoint per test corpus, tabulating the `f1_cues` P/R/F1 for negation and speculation |
| `evaluate_scope.py` | Re-evaluate a saved scope-resolution checkpoint per test corpus and task, tabulating token macro F1, IN_SCOPE F1 and scope-level F1 |
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

Both models are token classifiers over a **word-by-word** encoding that
`data.py` builds by hand: each word is sub-tokenized on its own, **no**
`[CLS]`/`[SEP]` special tokens are added, and a word's prediction is the argmax
of the *mean* of its sub-token logits. A plain `pipeline(...)` call on raw text
will not give meaningful results. This one helper covers both models:

```python
import numpy as np
import torch


def encode(words, tokenizer, lower, cues=None, task=None, scope_method="global"):
    """Encode a pre-tokenized sentence exactly the way data.py does.

    cues: {word index: cue label} -- 0 affix, 1 single-word, 2 multiword.
          Scope only. A [unused*] marker is inserted in front of EVERY
          sub-token of a cue word (data.py's loop runs over sub-tokens, not
          words); all of them belong to that word's slot.
    task: "Negation" or "Speculation". Scope only. Under 'global' it is
          appended after a literal "[SEP]" and both tasks use
          [unused{label+1}]; under 'local' there is no suffix, negation uses
          [unused{label+1}] and speculation uses [unused{label+6}] for the
          first cue sub-token, then [unused{label+1}].
    Returns the input ids and, for each id, which word it belongs to.
    """
    cues = cues or {}
    seq = list(words)
    if task is not None and scope_method == "global":
        seq += ["[SEP]", task]
    ids, word_of = [], []
    first = True                         # first sub-token of the current cue run
    for w, word in enumerate(seq):
        subs = tokenizer.tokenize(word.lower() if lower else word)
        if w in cues:
            for sub in subs:
                off = 6 if (scope_method == "local" and task == "Speculation" and first) else 1
                ids.append(tokenizer.convert_tokens_to_ids(f"[unused{cues[w] + off}]"))
                word_of.append(w)
                ids.append(tokenizer.convert_tokens_to_ids(sub))
                word_of.append(w)
                first = False
        else:
            first = True
            for sub in subs:
                ids.append(tokenizer.convert_tokens_to_ids(sub))
                word_of.append(w)
    return ids, word_of


def per_word(logits, word_of, n_words):
    """Average each word's sub-token logits, as model.py does, then argmax."""
    return [
        int(np.argmax(np.mean([logits[i] for i, w in enumerate(word_of) if w == word], axis=0)))
        for word in range(n_words)
    ]
```

The appended `[SEP] <task>` is ordinary text: it goes through the same
lower-casing (for an `uncased` model) and sub-tokenization as the sentence, and
is not the tokenizer's special `[SEP]` token. The marker string is never
lower-cased. Build the attention mask as `(input_ids > 0)` — that is what
training used, and under XLNet it is what hides the `<unk>` markers. `encode()`
has been checked against `data.py` on every scope row of the BioScope
full-papers corpus under XLNet and BERT, `global` and `local`.

#### Cue detection (default config: XLNet, BF+BA, combined)

The cue model is the custom two-headed `MultiHeadTokenClassifier`, so it loads
from a `state_dict` rather than `from_pretrained`. It takes plain text — no cue
markers, no task suffix — and returns negation and speculation labels for every
word in one pass.

```python
from transformers import AutoTokenizer
from config import CUE_MODEL, CUE_LABELS
from multihead_model import MultiHeadTokenClassifier

CKPT = "check_pts/cue_detection/xlnet-base-cased_BF+BA_combined/run1/best.pt"

model = MultiHeadTokenClassifier(CUE_MODEL, num_labels=5)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(CUE_MODEL, use_fast=False)
lower = "uncased" in CUE_MODEL          # xlnet-base-cased -> False

words = "It should be noted that the degree distribution is not maintained .".split()
ids, word_of = encode(words, tokenizer, lower)

input_ids = torch.tensor([ids])
with torch.no_grad():
    logits_neg, logits_spec = model(input_ids, attention_mask=(input_ids > 0).long())[0]

neg = per_word(logits_neg[0].numpy(), word_of, len(words))
spec = per_word(logits_spec[0].numpy(), word_of, len(words))

for word, n, s in zip(words, neg, spec):
    if n != 3 or s != 3:                # 3 = NOT_CUE
        print(f"{word:14s} negation={CUE_LABELS[n]:14s} speculation={CUE_LABELS[s]}")
# a trained checkpoint should print something like:
# -> not            negation=CUE           speculation=NOT_CUE
```

`CUE_LABELS` is `{0: AFFIX_CUE, 1: CUE, 2: MULTIWORD_CUE, 3: NOT_CUE, 4: PAD}`.
Label 4 is the padding label and carries a class weight of 0 during training,
so the model is never trained to predict it and never trained not to — if it
turns up on a real word, treat it as "no cue" rather than as a prediction.

#### Scope resolution (default config: XLNet, BF+BA, global, combined)

The scope model is saved as a HuggingFace folder and loads with the stock
`AutoModelForTokenClassification` (XLNet is *trained* through a subclass that
only adds dropout — see "The two configurations" — so nothing changes at load
time). It resolves the scope of **one cue at a time**: you tell it
where the cue is and which phenomenon you are asking about, and it labels every
word `IN_SCOPE` / `OUT_OF_SCOPE`. Feed it the cues the model above found, or
gold cues.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from config import SCOPE_MODEL, SCOPE_METHOD

CKPT = "check_pts/scope_resolution/xlnet-base-cased_BF+BA_global_combined/run1/best"

model = AutoModelForTokenClassification.from_pretrained(CKPT).eval()
tokenizer = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
lower = "uncased" in SCOPE_MODEL         # xlnet-base-cased -> False

words = ("They analyzed 146 prokaryotic genomes , but no likely tRNA "
         "of the novel amino acid was detected .").split()
cues = {7: 1}                            # "no" is a single-word cue
task = "Negation"                        # or "Speculation"

ids, word_of = encode(words, tokenizer, lower, cues=cues, task=task, scope_method=SCOPE_METHOD)

input_ids = torch.tensor([ids])
with torch.no_grad():
    logits = model(input_ids, attention_mask=(input_ids > 0).long()).logits[0].numpy()

n_words = len(words) + (2 if SCOPE_METHOD == "global" else 0)
pred = per_word(logits, word_of, n_words)[:len(words)]   # drop the "[SEP] Negation" suffix

print(" ".join(w.upper() if p == 1 else w for w, p in zip(words, pred)))
# this sentence is from BioScope full papers; its gold scope is
# -> They analyzed 146 prokaryotic genomes , but no LIKELY TRNA OF THE NOVEL
#    AMINO ACID WAS DETECTED .
```

The `cues` labels match `CUE_LABELS`: `1` for a single-word cue, `2` on **every**
word of a multiword cue, `0` for an affix cue. `encode()` handles `global` and
`local` itself via `scope_method`; always pass `task=`.

For the `separate` early-stopping method there is no single `best/`; use
`best/negation/` or `best/speculation/` (scope) or `best_negation.pt` /
`best_speculation.pt` (cue).
