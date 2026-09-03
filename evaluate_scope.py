"""Evaluate a saved scope-resolution checkpoint on each test corpus separately.

Loads one checkpoint (a HuggingFace folder, or a `best.pt` state_dict, written
by `main.py --subtask scope_resolution`), builds a negation and a speculation
test dataloader per corpus exactly the way `main.py` does, runs
`ScopeModel_*.evaluate()` on each, and collects three different F1 definitions
side by side so the run can be lined up cell-by-cell against Tables 6-11:

  * Token macro F1  -- the "Macro F1 Average (Token-level)" the paper's
                       Section 4 describes. This is the metric the broken
                       `f1_scope(level='token')` was meant to compute; the fix
                       is in metrics.py (its flattening comprehension had its
                       two `for` clauses in the wrong order, so the branch
                       raised NameError and was never usable).
  * Token F1        -- the IN_SCOPE class alone, which is what the authors'
    (IN_SCOPE)         code actually reports and so most likely what generated
                       the published tables.
  * Scope-level F1  -- exact whole-scope match, precision 1 by construction
                       because scopes are predicted at gold cues only.

    python evaluate_scope.py --checkpoint check_pts/scope_resolution/\
xlnet-base-cased_BF+BA+SFU_global_combined/run3/best --csv scope_run3.csv

The backbone, the cue-marker preprocessing scheme, the training corpora and the
early-stopping scheme are read out of the checkpoint path
(`<model>_<BF+BA+SFU>_<global|local>_<combined|separate>/runN/...`), which is how
main.py names it; pass --model / --scope-method / --train-datasets /
--early-stopping to override. Those have to be set before `config` is imported
(it binds its values at import time), so all project imports happen in main().

  !! The 15% test split is drawn with `np.random.randint(1, 2020)` inside
  data.py and is never recorded, so the exact split a training run held out
  CANNOT be reproduced after the fact. For a corpus the checkpoint trained on,
  a fresh split therefore overlaps that run's training data and the numbers
  come out optimistic. Use --repeats N to see how much the estimate moves
  across splits, and --seed to make a future run reproducible.
"""
import argparse
import csv
import os
import re
import statistics
import sys

# Order in which main.py assigns first_dataset/other_datasets, and so the order
# of the test dataloaders it gets back. Must match, or the per-corpus results
# get swapped.
DATALOADER_ORDER = ("sfu", "bioscope_full_papers", "bioscope_abstracts")
# Order main.py writes the corpus combo of an experiment tag in, so a combo
# printed here reads the same as the checkpoint directory it came from.
TAG_ORDER = ("bioscope_full_papers", "bioscope_abstracts", "sfu")
CORPUS_CODES = {"bioscope_full_papers": "BF", "bioscope_abstracts": "BA", "sfu": "SFU"}
CODE_TO_CORPUS = {v: k for k, v in CORPUS_CODES.items()}
DATASET_CHOICES = ("bioscope_full_papers", "bioscope_abstracts", "sfu")

# The fractions main.py passes for a corpus that was not trained on: data.py
# still calls train_test_split, which rejects an exact 0.0 or 1.0. A test_size
# at or above 0.99 is also what tells get_scope_dataloader to skip building
# train/val dataloaders it would not use.
WHOLE_CORPUS_TEST_SIZE = 0.9999999
WHOLE_CORPUS_VAL_SIZE = 0.00000001


def parse_tag(tag):
    """Pull the configuration back out of an experiment-tag directory name.

    main.py builds these as `<model>_<corpus combo>_<scope method>_<early
    stopping>`, e.g. `xlnet-base-cased_BF+BA+SFU_global_combined`. The corpus
    combo is the only field with a fixed shape, so it anchors the parse.
    """
    parts = tag.split("_")
    combo_re = re.compile(r"^(BF|BA|SFU)(\+(BF|BA|SFU))*$")
    for i, part in enumerate(parts):
        if combo_re.match(part):
            rest = parts[i + 1:]
            return {
                "model": "_".join(parts[:i]),
                "train_datasets": [CODE_TO_CORPUS[c] for c in part.split("+")],
                "scope_method": next((r for r in rest if r in ("global", "local")), None),
                "early_stopping": next((r for r in rest if r in ("combined", "separate")), None),
            }
    return None


def infer_from_path(path):
    """Search a checkpoint path for the experiment-tag component."""
    for part in reversed(re.split(r"[\\/]+", os.path.abspath(path))):
        parsed = parse_tag(part)
        if parsed:
            return parsed
    return {}


def resolve_checkpoint(path, early_stopping):
    """Return the checkpoint location(s) to load, given a path from main.py.

    Scope models are saved both as a HuggingFace folder (`best/`, or
    `best/<task>/` under 'separate' early stopping) and, for the combined
    scheme, as the raw `best.pt` state_dict early stopping wrote. A run
    directory is resolved to whichever of those it actually holds.
    """
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise SystemExit(f"No such checkpoint: {path}")
    if os.path.isfile(os.path.join(path, "config.json")):
        return [path]
    if early_stopping == "separate":
        paths = [os.path.join(path, "best", t) for t in ("negation", "speculation")]
        if all(os.path.isfile(os.path.join(p, "config.json")) for p in paths):
            return paths
        raise SystemExit(
            f"{path} holds no best/negation/ and best/speculation/ HuggingFace folders. "
            "Point --checkpoint at them directly."
        )
    for candidate in (os.path.join(path, "best"), os.path.join(path, "best.pt")):
        if os.path.isfile(os.path.join(candidate, "config.json")) or os.path.isfile(candidate):
            return [candidate]
    raise SystemExit(
        f"{path} contains neither best/ nor best.pt. Point --checkpoint at a checkpoint "
        "directly, or pass --early-stopping separate if this run used that scheme."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="A scope-resolution HuggingFace folder or .pt state_dict, or a runN/ directory "
             "containing best/ (best/negation/ + best/speculation/ for --early-stopping separate).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Backbone the checkpoint was trained from (e.g. xlnet-base-cased). Inferred from "
             "the checkpoint path when omitted. It selects the tokenizer data.py encodes with "
             "and, for XLNet, the dropout-patched classification head.",
    )
    parser.add_argument(
        "--scope-method", choices=("global", "local"), default=None,
        help="Cue-marker preprocessing scheme the checkpoint was trained with (paper Section "
             "3.2). Inferred from the checkpoint path when omitted, defaulting to global. "
             "Getting this wrong silently encodes the test data differently from training.",
    )
    parser.add_argument(
        "--train-datasets", nargs="+", choices=DATASET_CHOICES, default=None, metavar="NAME",
        help="Corpora the checkpoint was trained on. Inferred from the checkpoint path when "
             "omitted. Only these get a held-out split; the others are evaluated in full.",
    )
    parser.add_argument(
        "--test-datasets", nargs="+", choices=DATASET_CHOICES, default=list(DATASET_CHOICES),
        metavar="NAME", help="Corpora to evaluate on, each on its own. Default: all three.",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=("negation", "speculation"),
        default=["negation", "speculation"], metavar="TASK",
        help="Which scope tasks to evaluate. Default: both.",
    )
    parser.add_argument(
        "--early-stopping", choices=("combined", "separate"), default=None,
        help="Early-stopping scheme the checkpoint came from. Inferred from the path when "
             "omitted, defaulting to 'combined'.",
    )
    parser.add_argument(
        "--full-corpus", action="store_true",
        help="Evaluate on every corpus in full, including the ones trained on. Removes the "
             "random-split noise but scores the model partly on its own training sentences.",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Re-draw the held-out splits and evaluate this many times, reporting mean and "
             "standard deviation. Corpora evaluated in full are deterministic and are scored "
             "once regardless. Default: 1.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed numpy/torch so the splits are reproducible. Note this cannot recover the "
             "split of an already-finished training run that was launched without a seed -- "
             "those were drawn from an unrecorded np.random state.",
    )
    parser.add_argument(
        "--match-run", type=int, default=None, metavar="N",
        help="Evaluate on the split that `main.py --seed <same seed>` held out on its run N "
             "(1-based). Splits are drawn in the same order here as there, so this discards "
             "N-1 splits before scoring. Requires --seed, and the same --train-datasets and "
             "--test-datasets that training used.",
    )
    parser.add_argument("--val-size", type=float, default=None, help="Override VAL_SIZE (default: config.py's 0.15).")
    parser.add_argument("--test-size", type=float, default=None, help="Override TEST_SIZE (default: config.py's 0.15).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the eval batch size (default: config.py's bs).")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available).")
    parser.add_argument("--bioscope-full-papers", default="data/full_papers.xml")
    parser.add_argument("--bioscope-abstracts", default="data/abstracts.xml")
    parser.add_argument("--sfu", default="data/SFU_Review_Corpus_Negation_Speculation")
    parser.add_argument("--csv", default=None, help="Also write the summary table to this CSV file.")
    return parser.parse_args()


class DataCache:
    """Parse a corpus the first time it is asked for, then hand back the same
    `Data` object. Evaluating on BioScope alone should not pay for reading the
    several hundred SFU review files.
    """
    def __init__(self, paths):
        self.paths = paths
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._cache:
            from data import Data
            path, dataset_name = self.paths[name]
            print(f"Reading {name} from {path} ...")
            self._cache[name] = Data(path, dataset_name=dataset_name)
        return self._cache[name]


def order_datasets(data_objs, train_datasets):
    """Split the training corpora into main.py's (first_dataset, other_datasets)."""
    ordered = [data_objs[name] for name in DATALOADER_ORDER if name in train_datasets]
    return ordered[0], ordered[1:]


def build_test_dataloaders(data_objs, train_datasets, test_datasets, val_size, test_size, whole_corpus_cache):
    """Per-corpus {task: dataloader}, built the way main.py's scope branch does.

    A corpus that was trained on is represented by its held-out split (taken
    from the combined call, so the split matches the one that call's training
    data implies); one that was not is taken in full.
    """
    dataloaders = {}
    if any(name in train_datasets for name in test_datasets):
        first, others = order_datasets(data_objs, train_datasets)
        _, _, [neg_test_dls, spec_test_dls] = first.get_scope_dataloader(
            val_size=val_size, test_size=test_size, other_datasets=others
        )
        idx = 0
        for name in DATALOADER_ORDER:
            if name in train_datasets:
                if name in test_datasets:
                    dataloaders[name] = {"negation": neg_test_dls[idx], "speculation": spec_test_dls[idx]}
                idx += 1
    for name in test_datasets:
        if name not in dataloaders:
            if name not in whole_corpus_cache:
                _, _, [neg_dls, spec_dls] = data_objs[name].get_scope_dataloader(
                    test_size=WHOLE_CORPUS_TEST_SIZE, val_size=WHOLE_CORPUS_VAL_SIZE
                )
                whole_corpus_cache[name] = {"negation": neg_dls[0], "speculation": spec_dls[0]}
            dataloaders[name] = whole_corpus_cache[name]
    return dataloaders


def load_model(checkpoint_paths, early_stopping, device, train_dl_name):
    """Build the scope model and load the checkpoint weights into it.

    The model classes' own `train=False` path calls `torch.load` on a pickled
    *model object*, which is not what main.py writes; and their `train=True`
    path also builds an Adam optimizer over every parameter, which evaluation
    does not need. So the instance is created without running __init__ and only
    the attributes `evaluate()` reads are set.
    """
    import torch

    from config import SCOPE_LABELS, SCOPE_METHOD, SCOPE_MODEL
    from model import ScopeModel_Combined, ScopeModel_Separate, _scope_model_class

    num_labels = 2
    cls = ScopeModel_Separate if early_stopping == "separate" else ScopeModel_Combined
    model_wrapper = object.__new__(cls)
    model_wrapper.model_name = SCOPE_MODEL
    model_wrapper.task = "scope_resolution"
    model_wrapper.num_labels = num_labels
    model_wrapper.scope_method = SCOPE_METHOD
    model_wrapper.device = torch.device(device)
    model_wrapper.train_dl_name = train_dl_name

    def _load(path):
        label2id = {v: k for k, v in SCOPE_LABELS.items()}
        if os.path.isdir(path):
            net = _scope_model_class().from_pretrained(
                path, num_labels=num_labels, id2label=SCOPE_LABELS, label2id=label2id
            )
        else:
            # A raw state_dict: build the architecture from the backbone, then
            # overwrite its weights.
            net = _scope_model_class().from_pretrained(
                SCOPE_MODEL, num_labels=num_labels, id2label=SCOPE_LABELS, label2id=label2id
            )
            net.load_state_dict(torch.load(path, map_location="cpu"))
        net.to(model_wrapper.device)
        net.eval()
        return net

    model_wrapper.model = _load(checkpoint_paths[0])
    if early_stopping == "separate":
        model_wrapper.model_2 = _load(checkpoint_paths[1])
    return model_wrapper


def format_table(rows, repeats):
    """Render the collected per-corpus, per-task metrics."""
    header = ["Test corpus", "Task", "Split", "Token macro F1", "Token F1 (IN_SCOPE)",
              "Token acc", "Scope-level F1"]
    lines = [header]
    for row in rows:
        def cell(key):
            values = row[key]
            if repeats > 1 and len(values) > 1:
                return f"{statistics.mean(values):.4f}Â±{statistics.stdev(values):.4f}"
            return f"{statistics.mean(values):.4f}"
        lines.append([
            row["corpus"], row["task"], row["split"],
            cell("token_macro_f1"), cell("in_scope_f1"), cell("accuracy"), cell("scope_f1"),
        ])
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    out = []
    for i, line in enumerate(lines):
        out.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(line)).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def main():
    args = parse_args()

    inferred = infer_from_path(args.checkpoint)
    model_name = args.model or inferred.get("model")
    if not model_name:
        raise SystemExit(
            "Could not infer the backbone from the checkpoint path; pass --model "
            "(e.g. --model xlnet-base-cased)."
        )
    train_datasets = args.train_datasets or inferred.get("train_datasets")
    if train_datasets is None:
        raise SystemExit(
            "Could not infer the training corpora from the checkpoint path; pass "
            "--train-datasets to name what it trained on."
        )
    scope_method = args.scope_method or inferred.get("scope_method") or "global"
    early_stopping = args.early_stopping or inferred.get("early_stopping") or "combined"

    # config.py reads these at import time and data.py/model.py bind the values
    # when they import it, so they have to be in place before any of that runs.
    os.environ["SCOPE_MODEL"] = model_name
    os.environ["SCOPE_METHOD"] = scope_method
    if args.batch_size is not None:
        os.environ["BS"] = str(args.batch_size)

    import numpy as np
    import torch

    from config import DEVICE, TEST_SIZE, VAL_SIZE

    device = args.device or DEVICE
    val_size = args.val_size if args.val_size is not None else VAL_SIZE
    test_size = args.test_size if args.test_size is not None else TEST_SIZE
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.match_run is not None:
        if args.seed is None:
            raise SystemExit("--match-run needs --seed: the seed the training run was launched with.")
        if args.full_corpus:
            raise SystemExit("--match-run and --full-corpus contradict each other.")
        if args.match_run < 1:
            raise SystemExit("--match-run is 1-based.")

    split_corpora = [] if args.full_corpus else [d for d in args.test_datasets if d in train_datasets]
    effective_train_datasets = [] if args.full_corpus else train_datasets
    repeats = max(1, args.repeats)
    if not split_corpora or args.match_run is not None:
        repeats = 1

    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Backbone:        {model_name}")
    print(f"Scope method:    {scope_method}")
    print(f"Trained on:      {'+'.join(CORPUS_CODES[d] for d in TAG_ORDER if d in train_datasets)}")
    print(f"Early stopping:  {early_stopping}")
    print(f"Test corpora:    {', '.join(args.test_datasets)}")
    print(f"Tasks:           {', '.join(args.tasks)}")
    print(f"Device:          {device}")
    print(f"Repeats:         {repeats}")
    if split_corpora and args.match_run is not None:
        print(f"\nScoring the split `main.py --seed {args.seed}` held out on run {args.match_run}.\n")
    elif split_corpora:
        print(
            "\nNOTE: " + ", ".join(split_corpora) + " were trained on, so they are scored on a "
            f"freshly drawn {test_size:.0%} split. data.py seeds that split with an unrecorded "
            "np.random.randint, so unless the training run was launched with --seed and you pass "
            "the same one here with --match-run, this is NOT the split it held out -- some of "
            "these sentences were in that run's training data and the scores are optimistic.\n"
        )

    checkpoint_paths = resolve_checkpoint(args.checkpoint, early_stopping)
    data_objs = DataCache({
        "bioscope_full_papers": (args.bioscope_full_papers, "bioscope"),
        "bioscope_abstracts": (args.bioscope_abstracts, "bioscope"),
        "sfu": (args.sfu, "sfu"),
    })
    train_dl_name = ",".join(d for d in DATALOADER_ORDER if d in train_datasets)
    model = load_model(checkpoint_paths, early_stopping, device, train_dl_name)

    # main.py draws one set of splits per run, in this same order, so discarding
    # N-1 of them lands on run N's. Each warm-up gets a fresh whole-corpus cache
    # because main.py rebuilds those dataloaders every run too, and they consume
    # draws of their own.
    for skipped in range(1, args.match_run or 1):
        print(f"Discarding the splits of run {skipped} ...")
        build_test_dataloaders(
            data_objs, effective_train_datasets, args.test_datasets, val_size, test_size, {}
        )

    results = {}
    whole_corpus_cache = {}
    for repeat in range(repeats):
        if repeats > 1:
            print(f"\n=========== Repeat {repeat + 1}/{repeats} ===========")
        dataloaders = build_test_dataloaders(
            data_objs, effective_train_datasets, args.test_datasets,
            val_size, test_size, whole_corpus_cache,
        )
        for name in args.test_datasets:
            # A corpus scored in full is deterministic -- evaluating it again on
            # a later repeat would only stack copies of the same number.
            if repeat > 0 and name not in split_corpora:
                continue
            for task in args.tasks:
                print(f"\nEvaluate on {name} ({task}):")
                metrics = model.evaluate(dataloaders[name][task], test_dl_name=name, task=task)
                entry = results.setdefault((name, task), {
                    "corpus": name,
                    "task": task,
                    "split": "held-out split" if name in split_corpora else "full corpus",
                    "token_macro_f1": [], "in_scope_f1": [], "accuracy": [],
                    "scope_f1": [], "scope_recall": [], "in_scope_precision": [], "in_scope_recall": [],
                })
                entry["token_macro_f1"].append(metrics["Token Macro F1"])
                entry["in_scope_f1"].append(metrics["F1"])
                entry["in_scope_precision"].append(metrics["Precision"])
                entry["in_scope_recall"].append(metrics["Recall"])
                entry["accuracy"].append(metrics["Token Accuracy"])
                entry["scope_f1"].append(metrics["Scope Level - F1"])
                entry["scope_recall"].append(metrics["Scope Level - Recall"])

    rows = [results[(name, task)] for name in args.test_datasets for task in args.tasks]
    print("\n\n============ Scope resolution summary ============")
    print(f"{model_name} | {scope_method} | trained on {train_dl_name} | {args.checkpoint}")
    print("Token macro F1 is the paper's stated metric (Section 4, 'Macro F1 Average")
    print("(Token-level)'); token F1 (IN_SCOPE) is what the authors' code reports and is the")
    print("likelier source of Tables 6-11. Scope-level F1 is exact whole-scope match, whose")
    print("precision is 1 by construction because scopes are predicted at gold cues only.\n")
    print(format_table(rows, repeats))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "checkpoint", "model", "scope_method", "train_datasets", "early_stopping",
                "test_corpus", "task", "split", "n", "token_macro_f1", "token_f1_in_scope",
                "token_precision_in_scope", "token_recall_in_scope", "token_accuracy",
                "scope_level_f1", "scope_level_recall",
                "token_macro_f1_std", "token_f1_in_scope_std", "scope_level_f1_std",
            ])
            for row in rows:
                n = len(row["token_macro_f1"])
                std = (lambda values: statistics.stdev(values) if len(values) > 1 else "")
                writer.writerow([
                    args.checkpoint, model_name, scope_method,
                    "+".join(CORPUS_CODES[d] for d in TAG_ORDER if d in train_datasets),
                    early_stopping, row["corpus"], row["task"], row["split"], n,
                    statistics.mean(row["token_macro_f1"]), statistics.mean(row["in_scope_f1"]),
                    statistics.mean(row["in_scope_precision"]), statistics.mean(row["in_scope_recall"]),
                    statistics.mean(row["accuracy"]), statistics.mean(row["scope_f1"]),
                    statistics.mean(row["scope_recall"]),
                    std(row["token_macro_f1"]), std(row["in_scope_f1"]), std(row["scope_f1"]),
                ])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
