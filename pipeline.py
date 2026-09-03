"""
A pipeline for loading cue and scope models and do inference

from pipeline import get_cue_and_scope
result = get_cue_and_scope("The degree distribution is not maintained.")
"""
import os
import re
import warnings

# Checkpoints: the best run of each subtask by validation F1 (cue run 2, 0.9128;
# scope run 3, 0.9455). Both are XLNet + BF+BA+SFU + combined early stopping,
# scope with global preprocessing.
CUE_CHECKPOINT = os.environ.get(
    "CUE_CHECKPOINT",
    "check_pts/cue_detection/xlnet-base-cased_BF+BA+SFU_combined/run2/best.pt",
)
SCOPE_CHECKPOINT = os.environ.get(
    "SCOPE_CHECKPOINT",
    "check_pts/scope_resolution/xlnet-base-cased_BF+BA+SFU_global_combined/run3/best",
)

from evaluate_cue import infer_from_path as _infer_cue_path
from evaluate_scope import infer_from_path as _infer_scope_path

_cue_cfg = _infer_cue_path(CUE_CHECKPOINT)
_scope_cfg = _infer_scope_path(SCOPE_CHECKPOINT)
if _cue_cfg.get("model"):
    os.environ.setdefault("CUE_MODEL", _cue_cfg["model"])
if _scope_cfg.get("model"):
    os.environ.setdefault("SCOPE_MODEL", _scope_cfg["model"])
if _scope_cfg.get("scope_method"):
    os.environ.setdefault("SCOPE_METHOD", _scope_cfg["scope_method"])

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import CUE_LABELS, CUE_MODEL, DEVICE, MAX_LEN, SCOPE_LABELS, SCOPE_METHOD, SCOPE_MODEL
from multihead_model import MultiHeadTokenClassifier

NOT_CUE = 3          # CUE_LABELS: 0 AFFIX_CUE, 1 CUE, 2 MULTIWORD_CUE, 3 NOT_CUE, 4 PAD
MULTIWORD_CUE = 2
PAD_CUE = 4
IN_SCOPE = 1
NUM_CUE_LABELS = 5
TASKS = ("negation", "speculation")
# The scope model is asked about one phenomenon at a time, by the literal word
# appended after "[SEP]" -- capitalised exactly as data.py writes it.
TASK_SUFFIX = {"negation": "Negation", "speculation": "Speculation"}


def _load_cue_model():
    """The two-headed cue model: architecture from the backbone, weights from
    the checkpoint. HuggingFace cannot load `MultiHeadTokenClassifier`, so
    `main.py` saves it as a plain state_dict."""
    model = MultiHeadTokenClassifier(CUE_MODEL, num_labels=NUM_CUE_LABELS)
    model.load_state_dict(torch.load(CUE_CHECKPOINT, map_location="cpu"))
    return model.to(DEVICE).eval()


def _load_scope_model():
    """The scope model, saved as a HuggingFace folder (or a raw state_dict).

    Stock `AutoModelForTokenClassification` is enough: XLNet trains through a
    subclass that only adds a dropout layer, which is inactive at eval.
    """
    if os.path.isdir(SCOPE_CHECKPOINT):
        return AutoModelForTokenClassification.from_pretrained(SCOPE_CHECKPOINT).to(DEVICE).eval()
    model = AutoModelForTokenClassification.from_pretrained(
        SCOPE_MODEL, num_labels=len(SCOPE_LABELS), id2label=SCOPE_LABELS,
        label2id={v: k for k, v in SCOPE_LABELS.items()},
    )
    model.load_state_dict(torch.load(SCOPE_CHECKPOINT, map_location="cpu"))
    return model.to(DEVICE).eval()


def _load_tokenizer(model_name, lower):
    """data.py's tokenizer: the slow one, with do_lower_case tied to the backbone."""
    return AutoTokenizer.from_pretrained(model_name, do_lower_case=lower, use_fast=False)


# Loaded once, at import. An uncased backbone lower-cases its input, exactly as
# data.py decides it: `do_lower_case = 'uncased' in MODEL`.
CUE_LOWER = "uncased" in CUE_MODEL
SCOPE_LOWER = "uncased" in SCOPE_MODEL
CUE_TOKENIZER = _load_tokenizer(CUE_MODEL, CUE_LOWER)
SCOPE_TOKENIZER = _load_tokenizer(
    SCOPE_CHECKPOINT if os.path.isdir(SCOPE_CHECKPOINT) else SCOPE_MODEL, SCOPE_LOWER
)
CUE_MODEL_OBJ = _load_cue_model()
SCOPE_MODEL_OBJ = _load_scope_model()


def _simple_tokenize(text):
    """Split raw text into words roughly the way the corpora are tokenized.

    BioScope and SFU ship pre-tokenized, with punctuation as separate tokens,
    and the models were trained on that. This separates punctuation and keeps
    word characters together, which is close enough for ordinary prose -- but
    if you already have gold tokenization, pass it as `words=` instead of
    relying on this.
    """
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _encode(words, tokenizer, lower, cues=None, task=None, scope_method="global"):
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


def _truncate(ids, word_of):
    """Cut to MAX_LEN sub-tokens, as data.py's pad_sequences does when training."""
    if len(ids) <= MAX_LEN:
        return ids, word_of
    warnings.warn(
        f"Input is {len(ids)} sub-tokens; truncating to MAX_LEN={MAX_LEN}, which is what "
        "training did too. Words past the cut get the default label.",
        stacklevel=3,
    )
    return ids[:MAX_LEN], word_of[:MAX_LEN]


def _per_word(logits, word_of, n_words, default):
    """Average each word's sub-token logits, as model.py does, then argmax.

    A word with no surviving positions (truncated away) takes `default`.
    """
    by_word = {}
    for i, w in enumerate(word_of):
        by_word.setdefault(w, []).append(logits[i])
    return [
        int(np.argmax(np.mean(by_word[word], axis=0))) if word in by_word else default
        for word in range(n_words)
    ]


@torch.no_grad()
def _run_cue_model(words):
    """Per-word negation and speculation cue labels for one sentence."""
    ids, word_of = _truncate(*_encode(words, CUE_TOKENIZER, CUE_LOWER))
    input_ids = torch.tensor([ids], device=DEVICE)
    logits_neg, logits_spec = CUE_MODEL_OBJ(
        input_ids, attention_mask=(input_ids > 0).long()
    )[0]
    return {
        "negation": _per_word(logits_neg[0].cpu().numpy(), word_of, len(words), NOT_CUE),
        "speculation": _per_word(logits_spec[0].cpu().numpy(), word_of, len(words), NOT_CUE),
    }


@torch.no_grad()
def _run_scope_model(words, cues, task):
    """Per-word IN_SCOPE/OUT_OF_SCOPE labels for one cue.

    `cues` is {word index: cue label} for that one cue only -- the scope model
    resolves one cue at a time, which is how the corpora are laid out too (one
    row per cue, not per sentence).
    """
    ids, word_of = _encode(
        words, SCOPE_TOKENIZER, SCOPE_LOWER,
        cues=cues, task=TASK_SUFFIX[task], scope_method=SCOPE_METHOD,
    )
    ids, word_of = _truncate(ids, word_of)
    input_ids = torch.tensor([ids], device=DEVICE)
    logits = SCOPE_MODEL_OBJ(
        input_ids, attention_mask=(input_ids > 0).long()
    ).logits[0].cpu().numpy()
    # Under 'global' the encoding carries two extra words ("[SEP]", the task
    # name); score them and then drop them.
    n_words = len(words) + (2 if SCOPE_METHOD == "global" else 0)
    return _per_word(logits, word_of, n_words, 0)[:len(words)]


def _cue_instances(labels):
    """Split one sentence's merged cue labels into individual cues.

    Yields {word index: cue label} dicts, one per cue. Single-word cues (CUE,
    AFFIX_CUE) stand alone; a contiguous run of MULTIWORD_CUE words is one cue.
    PAD (4) is never trained as a real prediction -- it carries class weight 0 --
    so it is treated as "no cue" if it turns up.
    """
    instances = []
    run = {}
    for index, label in enumerate(labels):
        if label == MULTIWORD_CUE:
            run[index] = label
            continue
        if run:
            instances.append(run)
            run = {}
        if label not in (NOT_CUE, PAD_CUE):
            instances.append({index: label})
    if run:
        instances.append(run)
    return instances


def get_cue_and_scope(text, words=None, debug = False):
    """Detect negation and speculation cues in `text`, then resolve each scope.

    text:  the sentence. One sentence at a time -- both models were trained on
           single sentences, and the scope of a cue never crosses one.
    words: optional pre-tokenized words. Pass these when you have the corpus'
           own tokenization; otherwise `text` is split with `_simple_tokenize`.

    Returns a dict:

        text             the input string
        words            the tokens everything below is indexed by
        cue_label_ids    {task: [label id per word]}   -- raw cue-model output
        cue_labels       {task: [label name per word]} -- CUE_LABELS applied
        cues             one entry per detected cue, each with:
            task             "negation" or "speculation"
            cue_type         AFFIX_CUE / CUE / MULTIWORD_CUE
            cue_indices      word indices making up the cue
            cue_words        those words
            scope_label_ids  [0/1 per word] -- raw scope-model output
            scope_labels     [OUT_OF_SCOPE/IN_SCOPE per word]
            scope_indices    word indices predicted IN_SCOPE
            scope_words      those words

    `cues` is empty when no cue is found, and the scope model is not run at all
    in that case -- scope resolution is defined relative to a cue.
    """
    if debug:
        print(f"get_cue_and_scope: text={text}")
        print(f"cue model:   {CUE_MODEL} <- {CUE_CHECKPOINT}")
        print(f"scope model: {SCOPE_MODEL} ({SCOPE_METHOD}) <- {SCOPE_CHECKPOINT}")
        print(f"device:      {DEVICE}\n")

    words = list(words) if words is not None else _simple_tokenize(text)
    if not words:
        return {"text": text, "words": [], "cue_label_ids": {t: [] for t in TASKS},
                "cue_labels": {t: [] for t in TASKS}, "cues": []}

    cue_label_ids = _run_cue_model(words)

    cues = []
    for task in TASKS:
        for instance in _cue_instances(cue_label_ids[task]):
            scope_label_ids = _run_scope_model(words, instance, task)
            indices = sorted(instance)
            scope_indices = [i for i, label in enumerate(scope_label_ids) if label == IN_SCOPE]
            cues.append({
                "task": task,
                "cue_type": CUE_LABELS[cue_label_ids[task][indices[0]]],
                "cue_indices": indices,
                "cue_words": [words[i] for i in indices],
                "scope_label_ids": scope_label_ids,
                "scope_labels": [SCOPE_LABELS[label] for label in scope_label_ids],
                "scope_indices": scope_indices,
                "scope_words": [words[i] for i in scope_indices],
            })

    return {
        "text": text,
        "words": words,
        "cue_label_ids": cue_label_ids,
        "cue_labels": {task: [CUE_LABELS[label] for label in ids]
                       for task, ids in cue_label_ids.items()},
        "cues": cues,
    }


def format_result(result):
    """Render one `get_cue_and_scope` result as readable lines."""
    lines = [f"text:  {result['text']}", f"words: {' '.join(result['words'])}"]
    for task in TASKS:
        marked = [
            f"{word}[{CUE_LABELS[label]}]" if label not in (NOT_CUE, PAD_CUE) else word
            for word, label in zip(result["words"], result["cue_label_ids"][task])
        ]
        lines.append(f"\n{task} cues: {' '.join(marked)}")
    if not result["cues"]:
        lines.append("\nNo cues found, so no scopes were resolved.")
    for cue in result["cues"]:
        lines.append(
            f"\n{cue['task']} cue {cue['cue_words']} ({cue['cue_type']}) "
            f"at {cue['cue_indices']}"
        )
        lines.append("  scope: " + " ".join(
            word.upper() if label == IN_SCOPE else word
            for word, label in zip(result["words"], cue["scope_label_ids"])
        ))
        lines.append(f"  scope words: {cue['scope_words']}")
    return "\n".join(lines)