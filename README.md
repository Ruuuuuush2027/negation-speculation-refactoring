# Negation & Speculation Cue/Scope Detection

Command-line port of `original_file.ipynb` (a copy of
[adityak6798/Transformers-For-Negation-and-Speculation](https://github.com/adityak6798/Transformers-For-Negation-and-Speculation)).
The Google Drive/Colab mounting cell has been removed — everything now reads
local files.

## Setup

```
pip install -r requirements.txt
```

The corpora are used in their original, unmodified distribution formats —
no preprocessing step is needed. Expected layout under `data/`:

```
data/
  abstracts.xml                              # BioScope abstracts (BioScope XML)
  full_papers.xml                            # BioScope full papers (BioScope XML)
  SFU_Review_Corpus_Negation_Speculation/    # SFU Review corpus root directory
    BOOKS/*.xml  CARS/*.xml  COMPUTERS/*.xml  COOKWARE/*.xml
    HOTELS/*.xml MOVIES/*.xml MUSIC/*.xml     PHONES/*.xml
```

`--sfu` takes the corpus *directory*, not a file: `Data` walks every
sub-directory that has no `.` in its name and parses every file inside, so
the README/PDFs at the SFU root are skipped automatically and all 400
reviews across the 8 categories are loaded.

## Run

Two models, one command each:

```
python main.py --subtask cue_detection
python main.py --subtask scope_resolution
```

`config.py` is already set to the configuration each subtask should use (see
below), so there is nothing to edit. If your corpora live elsewhere:

```
python main.py --subtask cue_detection \
    --bioscope-full-papers path/full_papers.xml \
    --bioscope-abstracts path/abstracts.xml \
    --sfu path/SFU_Review_Corpus_Negation_Speculation
```

The two subtasks are independent — scope resolution trains on the gold cue
annotations that ship with the corpora, not on anything the cue model predicts
— so you can run both at once in two processes. They write to separate
checkpoint directories, but on a single GPU they will compete for VRAM.

## The two configurations

Both are taken from Khandelwal & Britto (2020), *Multitask Learning of Negation
and Speculation using Transformers*, LOUHI @ EMNLP 2020
([doi](https://doi.org/10.18653/v1/2020.louhi-1.9)). §5.4 reports XLNet as the
best backbone, combined early stopping as the better of the two schemes, and
global as the better cue-preprocessing method. Tables 5 and 8 show BF+BA
(BioScope full papers + abstracts) as the training set behind the paper's best
BioScope results.

| | Cue detection | Scope resolution |
|---|---|---|
| Backbone | `xlnet-base-cased` | `bert-base-uncased` |
| Training corpora | BF+BA | BF+BA |
| Early stopping | combined | combined |
| Cue preprocessing | — | global |

Also pinned to the paper's §4 protocol: a 70-15-15 train/dev/test split,
patience 6 on validation F1, testing on all three corpora, and 3 runs averaged
(its rule for a multi-corpus training set). Pass `--num-runs 1` for a quicker
first look.

**Scope resolution uses BERT rather than XLNet on purpose.** The paper's best
scope models are XLNet too, but scope resolution marks cue words with the
reserved WordPiece tokens `[unused1]`-`[unused8]`, which exist only in
BERT-family vocabularies. Under `xlnet-base-cased` all of them map to id 0,
which is both `<unk>` and the id the attention mask treats as padding, so the
markers are dropped entirely; under `roberta-base` they all collapse to
`<unk>`. BERT is the best backbone that actually works here, and BERT (BF+BA,
global) is a headline result in its own right — 97.40 in Table 8. Cue
detection injects no markers, so XLNet is fine there.

### Comparing against the paper

Metrics print at the end of each `Evaluate on <corpus>:` block; take the F1
from the `classification_report`. Targets for these two configurations, from
the paper's "Ours (Jointly Trained)" rows:

| Run | Test corpus | Paper | Source |
|---|---|---|---|
| Cue detection, negation | BioScope Abstracts | 97.01 | Table 5 |
| Cue detection, negation | BioScope Full Papers | 96.25 | Table 5 |
| Cue detection, speculation | BioScope Abstracts | 93.98 | Table 5 |
| Scope resolution, negation | BioScope Full Papers | 97.40 | Table 8 |

Two things to know before reading a gap as a failed replication:

- **SFU will score badly**, because it is not in BF+BA — those cells are
  cross-domain transfer, and the paper's own numbers for this configuration
  drop to the mid-30s on SFU. That is the split behaving, not a bug.
- **The reported metric is not literally the one the paper's prose describes.**
  §4 says "Macro F1 Average (Token-level)", but the code scores scope
  resolution as the F1 of the `IN_SCOPE` class alone and cue detection with
  `f1_cues`, which pools the three cue classes against `NOT_CUE`. This code is
  a port of the authors' own notebook, so it is probably what produced the
  tables — but the two definitions give different numbers.

To try a different backbone or preprocessing scheme, set `MODEL=` or
`SCOPE_METHOD=` in the environment (`data.py` and `model.py` bind those at
import time, so a command-line flag would be read too late). Training corpora
and early stopping are `--train-datasets` and `--early-stopping`.

## Checkpoints

Checkpoints are written to `./check_pts/` by default: one every 10 epochs,
plus the best-validation weights that early stopping tracks. Override with
`--checkpoint-dir` / `--checkpoint-every` (`0` = only keep the best), or edit
`CHECKPOINT_DIR` / `CHECKPOINT_EVERY` in `config.py`. Each run gets its own
directory, named for the configuration that produced it, so the two subtasks
(and any config you try later) never overwrite each other:

```
check_pts/cue_detection/xlnet-base-cased_BF+BA_combined/run1/
check_pts/scope_resolution/bert-base-uncased_BF+BA_global_combined/run1/
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

### Using the trained models

Both models are token classifiers over a **word-by-word** encoding that
`data.py` builds by hand: each word is sub-tokenized on its own, **no**
`[CLS]`/`[SEP]` special tokens are added, and a word's prediction is the argmax
of the *mean* of its sub-token logits. A plain `pipeline(...)` call on raw text
will not give meaningful results. This one helper covers both models:

```python
import numpy as np
import torch


def encode(words, tokenizer, lower, cues=None, task=None):
    """Encode a pre-tokenized sentence the way data.py does.

    cues: {word index: cue label} -- 0 affix, 1 single-word, 2 multiword.
          Only for scope resolution; a [unused{label+1}] marker is inserted
          in front of the cue word and takes over its word slot.
    task: "Negation" or "Speculation", appended after a literal "[SEP]".
          Only for scope resolution with SCOPE_METHOD='global'.
    Returns the input ids and, for each id, which word it belongs to.
    """
    seq = list(words) if task is None else list(words) + ["[SEP]", task]
    cues = cues or {}
    ids, word_of = [], []
    for w, word in enumerate(seq):
        if w in cues:
            ids.append(tokenizer.convert_tokens_to_ids(f"[unused{cues[w] + 1}]"))
            word_of.append(w)
        for sub in tokenizer.tokenize(word.lower() if lower else word):
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

Note the marker is inserted *after* lower-casing, so `[unused2]` keeps its case
while the appended `[SEP] Negation` is lower-cased to `[sep] negation` and
sub-tokenized as ordinary text — it is not the real `[SEP]` special token.
Both helpers above reproduce `data.py`'s token ids exactly.

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
    logits_neg, logits_spec = model(input_ids, attention_mask=torch.ones_like(input_ids))[0]

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

#### Scope resolution (default config: BERT, BF+BA, global, combined)

The scope model is a stock `AutoModelForTokenClassification`, saved as a
HuggingFace folder. It resolves the scope of **one cue at a time**: you tell it
where the cue is and which phenomenon you are asking about, and it labels every
word `IN_SCOPE` / `OUT_OF_SCOPE`. Feed it the cues the model above found, or
gold cues.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from config import SCOPE_MODEL, SCOPE_METHOD

CKPT = "check_pts/scope_resolution/bert-base-uncased_BF+BA_global_combined/run1/best"

model = AutoModelForTokenClassification.from_pretrained(CKPT).eval()
tokenizer = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
lower = "uncased" in SCOPE_MODEL         # bert-base-uncased -> True

words = ("They analyzed 146 prokaryotic genomes , but no likely tRNA "
         "of the novel amino acid was detected .").split()
cues = {7: 1}                            # "no" is a single-word cue
task = "Negation"                        # or "Speculation"

ids, word_of = encode(words, tokenizer, lower, cues=cues,
                      task=task if SCOPE_METHOD == "global" else None)

input_ids = torch.tensor([ids])
with torch.no_grad():
    logits = model(input_ids, attention_mask=torch.ones_like(input_ids)).logits[0].numpy()

n_words = len(words) + (2 if SCOPE_METHOD == "global" else 0)
pred = per_word(logits, word_of, n_words)[:len(words)]   # drop the "[SEP] Negation" suffix

print(" ".join(w.upper() if p == 1 else w for w, p in zip(words, pred)))
# this sentence is from BioScope full papers; its gold scope is
# -> They analyzed 146 prokaryotic genomes , but no LIKELY TRNA OF THE NOVEL
#    AMINO ACID WAS DETECTED .
```

The `cues` labels match `CUE_LABELS`: `1` for a single-word cue, `2` on **every**
word of a multiword cue, `0` for an affix cue. Under `SCOPE_METHOD='global'`
both phenomena use the same `[unused{label+1}]` markers and the task is carried
by the appended text, so pass `task=`. Under `'local'` the task is carried by
the marker id instead (speculation uses `[unused{label+6}]`), so drop the `task`
argument and adjust the marker accordingly.

For the `separate` early-stopping method there is no single `best/`; use
`best/negation/` or `best/speculation/` (scope) or `best_negation.pt` /
`best_speculation.pt` (cue).

## Layout

| File | Contents |
|---|---|
| `main.py` | CLI entry point: loads datasets, runs the train/eval loop |
| `config.py` | Hyperparameters and run configuration (`MAX_LEN`, `EPOCHS`, `CUE_MODEL`, ... and an auto-detected `DEVICE`) |
| `data.py` | `Cues`/`Scopes`/`Data`: BioScope & SFU corpus parsing and `DataLoader` construction |
| `model.py` | `CueModel_Combined`, `CueModel_Separate`, `ScopeModel_Combined`, `ScopeModel_Separate` — the actual task models |
| `metrics.py` | F1/accuracy helpers used during training and evaluation |
| `early_stopping.py` | `EarlyStopping` checkpoint helper |
| `multihead_model.py` | `MultiHeadTokenClassifier` — a dual (negation + speculation) linear head on top of any `transformers.AutoModel` encoder, used for cue detection |

## Notes

- `keras.preprocessing.sequence.pad_sequences` was replaced with a small
  local re-implementation in `data.py`, so this no longer needs a
  tensorflow/keras install.
- Model constructors default `device` to `config.DEVICE`, which is `'cuda'`
  if available and `'cpu'` otherwise, instead of assuming a Colab GPU.
- The original notebook vendored ~2600 lines of an old release of
  `huggingface/transformers` (`BertConfig`, `PreTrainedModel`,
  `BertPreTrainedModel`, XLNet internals, TF-checkpoint loaders, ...) and
  pointed model/config downloads at hardcoded, now-dead S3 URLs
  (`s3.amazonaws.com/models.huggingface.co/...`). All of that is gone.
  `data.py` and `model.py` now use `transformers.AutoTokenizer`,
  `AutoConfig`, `AutoModel`, and `AutoModelForTokenClassification`, which
  resolve model names against the current HuggingFace Hub and handle
  bert/roberta/xlnet (and anything else Auto* supports) uniformly — no more
  `if 'xlnet' in MODEL: ... elif 'roberta' in MODEL: ...` branching.
- The one thing HF doesn't ship out of the box is the dual negation +
  speculation classification head used for cue detection, so that's now the
  small, architecture-agnostic `MultiHeadTokenClassifier` in
  `multihead_model.py` (`AutoModel` backbone + two `nn.Linear` heads).
  Scope resolution uses a single head, so it's just
  `AutoModelForTokenClassification.from_pretrained(...)` directly.
- Tokenization uses the public `tokenizer.tokenize(...)` /
  `tokenizer.convert_tokens_to_ids(...)` API (the notebook called the
  private `_tokenize`/`_convert_token_to_id` methods), with
  `use_fast=False` so the word-by-word sub-tokenization behaviour the
  cue/scope tagging logic expects still applies to whichever tokenizer
  `AutoTokenizer` resolves to.
- `MultiHeadTokenClassifier` re-initializes its two classification heads
  with `N(0, initializer_range)` weights and zero biases, matching the
  `init_weights()` call the notebook's vendored model classes performed
  (rather than PyTorch's default `nn.Linear` init). It also only forwards
  `token_type_ids` to the encoder when one is given, so backbones without
  segment embeddings work too.
- The "Error Analysis" block that was commented out in the notebook is now
  implemented behind the `--error-analysis` flag. Two bugs in the
  commented-out code were fixed in the process: the speculation no-punct
  dataloader variable shadowed the negation one (so speculation would have
  been evaluated on negation data), and the returned dataloader *lists*
  were stored where single dataloaders were expected (missing `[0]`,
  unlike the active test-dataloader code right above it). Empty splits are
  skipped with a message instead of crashing on zero batches.
- `main.py` uses `argparse.BooleanOptionalAction`, so Python 3.9+ is
  required.
- One deliberate deviation from the notebook's parsing: the corpus files
  are split into text/tag tokens *before* HTML entities are unescaped
  (`split_tags` in `data.py`). The notebook unescaped each line first, so a
  literal `<` in sentence text (BioScope's `p &lt; 0.05` etc.) was treated
  as the start of a tag and the remainder of that sentence was dropped.
  Those sentences are now parsed in full. Everything else about the parsing
  is unchanged.
