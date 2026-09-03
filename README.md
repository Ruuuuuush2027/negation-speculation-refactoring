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

```
python main.py
```

or, if your files live elsewhere:

```
python main.py --bioscope-full-papers path/full_papers.xml --sfu path/SFU_Review_Corpus_Negation_Speculation --bioscope-abstracts path/abstracts.xml
```

## Checkpoints

Checkpoints are written to `./check_pts/` by default: one every 10 epochs,
plus the best-validation weights that early stopping tracks. Override with
`--checkpoint-dir` / `--checkpoint-every` (`0` = only keep the best), or edit
`CHECKPOINT_DIR` / `CHECKPOINT_EVERY` in `config.py`. Each run gets its own
sub-directory. The format depends on the model:

- **Scope resolution** models are plain HuggingFace models, so they are
  saved as HuggingFace folders (config + weights + tokenizer, with
  `IN_SCOPE` / `OUT_OF_SCOPE` in `id2label`).
- **Cue detection** models are the custom two-headed
  `MultiHeadTokenClassifier` (one shared encoder, a negation head and a
  speculation head), which HuggingFace cannot load, so they are saved as
  PyTorch `state_dict` files.

```
check_pts/run1/                       scope resolution         cue detection
  every N epochs ................... epoch_010/  epoch_020/   epoch_010.pt  epoch_020.pt
  best validation F1 ............... best.pt + best/          best.pt
  ('separate' early stopping) ...... best_negation.pt + best/negation/     best_negation.pt
                                     best_speculation.pt + best/speculation/  best_speculation.pt
```

`best*.pt` is what gets reloaded for the final test evaluation, exactly as
the notebook did with its `checkpoint.pt`. Every file/folder is a full copy
of the weights (~500 MB for `bert-base` / `roberta-base`).

### Loading a scope-resolution checkpoint

The model is a standard `AutoModelForTokenClassification`, but the input has
to be prepared the way `data.py` does during training: words are
sub-tokenized one at a time, **no** `[CLS]`/`[SEP]` special tokens are
added, text is lower-cased for an `uncased` model, and each cue word is
marked by a `[unused{label+1}]` token in front of it (single-word cue: label
1 → `[unused2]`; every word of a multi-word cue: label 2 → `[unused3]`; with
`SCOPE_METHOD = 'local'` a speculation cue's *first* marker is
`[unused{label+6}]` instead). This is why a plain `pipeline(...)` call on raw
text would not give meaningful results.

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ckpt = "check_pts/run1/best"          # or "check_pts/run1/best/negation" for 'separate'
model = AutoModelForTokenClassification.from_pretrained(ckpt).eval()
tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)

words = "the drug did not reduce fever .".split()   # lower-cased: bert-base-uncased
cue_word_ids = {3}                                   # "not" is a single-word negation cue

tokens, is_first_subword = [], []
for i, word in enumerate(words):
    subwords = tokenizer.tokenize(word)
    if i in cue_word_ids:                            # marker before each cue sub-word
        subwords = [s for sub in subwords for s in ("[unused2]", sub)]
    tokens += subwords
    is_first_subword += [True] + [False] * (len(subwords) - 1)

input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
with torch.no_grad():
    logits = model(input_ids, attention_mask=torch.ones_like(input_ids)).logits[0]

pred = logits.argmax(-1).tolist()
word_labels = [model.config.id2label[p] for p, first in zip(pred, is_first_subword) if first]
print(list(zip(words, word_labels)))
```

(The training/eval code averages the logits of a word's sub-tokens rather
than taking the first one; see `get_scope_dataloader` in `data.py` and the
`evaluate` methods in `model.py` for the exact bookkeeping.)

### Loading a cue-detection checkpoint

```python
import torch
from transformers import AutoTokenizer
from config import CUE_MODEL, CUE_LABELS
from multihead_model import MultiHeadTokenClassifier

model = MultiHeadTokenClassifier(CUE_MODEL, num_labels=5)
model.load_state_dict(torch.load("check_pts/run1/best.pt", map_location="cpu"))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(CUE_MODEL, use_fast=False)

words = "The drug did not reduce fever .".split()   # roberta-base is cased: no lower-casing
tokens, is_first_subword = [], []
for word in words:
    subwords = tokenizer.tokenize(word)
    tokens += subwords
    is_first_subword += [True] + [False] * (len(subwords) - 1)

input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
with torch.no_grad():
    logits_neg, logits_spec = model(input_ids, attention_mask=torch.ones_like(input_ids))[0]

def to_word_labels(logits):
    pred = logits[0].argmax(-1).tolist()
    return [CUE_LABELS[p] for p, first in zip(pred, is_first_subword) if first]

print(list(zip(words, to_word_labels(logits_neg), to_word_labels(logits_spec))))
```

With the `separate` early-stopping method, load `best_negation.pt` for the
negation head and `best_speculation.pt` for the speculation head (each file
contains the whole two-headed model; only the head matching its name was
selected on validation F1).

For scope resolution, pass `--error-analysis` to additionally evaluate the
trained model on two extra test splits per test dataset: sentences whose
gold scope is delimited by punctuation (`*_punct`) and the rest
(`*_no_punct`). This is the "Error Analysis" that was left commented out in
the original notebook. `--no-error-analysis` disables it; with neither flag
the `ERROR_ANALYSIS_FOR_SCOPE` constant in `config.py` decides.

Training behaviour (subtask, models used, epochs, batch size, etc.) is
controlled by the constants at the top of `config.py` — this mirrors the
config cell from the original notebook.

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
