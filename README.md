# Negation & Speculation Cue/Scope Detection

Command-line port of `original_file.ipynb` (a copy of
[adityak6798/Transformers-For-Negation-and-Speculation](https://github.com/adityak6798/Transformers-For-Negation-and-Speculation)).
The Google Drive/Colab mounting cell has been removed — everything now reads
local files.

## Setup

```
pip install -r requirements.txt
```

Place your BioScope / SFU Review corpus files under `data/` (or anywhere,
and point `--bioscope-full-papers` / `--sfu` / `--bioscope-abstracts` at
them).

## Run

```
python main.py --bioscope-full-papers data/bioscope_full_papers.xml --sfu data/sfu_review_corpus.txt --bioscope-abstracts data/bioscope_abstracts.xml
```

All defaults point at `data/<name>`, so if your files are already named and
placed there you can just run `python main.py`.

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
