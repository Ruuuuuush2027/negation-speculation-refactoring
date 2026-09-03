"""Command-line entry point: train and evaluate the cue detection /
scope resolution models on local BioScope / SFU Review data files."""
import argparse
import os

from config import (
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    CUE_MODEL,
    EARLY_STOPPING_METHOD,
    EPOCHS,
    ERROR_ANALYSIS_FOR_SCOPE,
    PAPER_TEST_DATASETS,
    PATIENCE,
    INITIAL_LEARNING_RATE,
    SCOPE_METHOD,
    SCOPE_MODEL,
    SUBTASK,
    TEST_SIZE,
    TEST_DATASETS,
    TRAIN_DATASETS,
    VAL_SIZE,
    runs_for,
)
from data import Data
from model import CueModel_Combined, CueModel_Separate, ScopeModel_Combined, ScopeModel_Separate


DATASET_CHOICES = ("bioscope_full_papers", "bioscope_abstracts", "sfu")

# The paper's short corpus names, used to label checkpoint directories so they
# read like the columns of its result tables (BF+BA, BF+BA+SFU, ...).
CORPUS_CODES = {
    "bioscope_full_papers": "BF",
    "bioscope_abstracts": "BA",
    "sfu": "SFU",
}


def experiment_tag(subtask, train_datasets, early_stopping_method):
    """Directory name identifying this cell of the paper's experiment grid.

    Sweeping the grid means running the same subtask many times over different
    backbones, corpus combinations and early-stopping schemes; without this in
    the path they would all write into the same run<N>/ and overwrite one
    another.
    """
    combo = "+".join(CORPUS_CODES[d] for d in PAPER_TEST_DATASETS if d in train_datasets)
    model = CUE_MODEL if subtask == "cue_detection" else SCOPE_MODEL
    parts = [model, combo]
    if subtask == "scope_resolution":
        parts.append(SCOPE_METHOD)
    parts.append(early_stopping_method)
    return "_".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate negation & speculation cue detection and scope resolution models."
    )
    parser.add_argument(
        "--bioscope-full-papers", default="data/full_papers.xml",
        help="Path to the BioScope full-papers XML file.",
    )
    parser.add_argument(
        "--sfu", default="data/SFU_Review_Corpus_Negation_Speculation",
        help="Path to the SFU Review corpus *directory* (the one containing the "
             "BOOKS/, CARS/, ... category sub-directories of annotated XML files).",
    )
    parser.add_argument(
        "--bioscope-abstracts", default="data/abstracts.xml",
        help="Path to the BioScope abstracts XML file.",
    )
    parser.add_argument(
        "--subtask", choices=("cue_detection", "scope_resolution"), default=SUBTASK,
        help="Which subtask to train/evaluate. Defaults to SUBTASK from config.py. "
             "The two subtasks are independent (scope resolution trains on the corpus' gold "
             "cues, not on the cue model's predictions), so they can be run concurrently; "
             "each writes to its own checkpoint sub-directory.",
    )
    parser.add_argument(
        "--train-datasets", nargs="+", choices=DATASET_CHOICES, default=list(TRAIN_DATASETS),
        metavar="NAME",
        help="Corpora to train on. The paper reports seven combinations (the columns of its "
             "result tables): each corpus alone, each pair, and all three. Defaults to "
             "TRAIN_DATASETS from config.py.",
    )
    parser.add_argument(
        "--test-datasets", nargs="+", choices=DATASET_CHOICES, default=list(TEST_DATASETS),
        metavar="NAME",
        help="Corpora to evaluate on. The paper always tests on all three regardless of what "
             "was trained on. A corpus that was trained on is evaluated on its held-out 15%% "
             "test split; one that was not is evaluated in full.",
    )
    parser.add_argument(
        "--num-runs", type=int, default=None,
        help="How many independent runs (each re-splits the data and starts from fresh "
             "pretrained weights). Defaults to the paper's rule: 5 runs when training on a "
             "single corpus, 3 when training on a combination. Average the runs to compare "
             "against the paper's tables.",
    )
    parser.add_argument(
        "--early-stopping", choices=("combined", "separate"), default=EARLY_STOPPING_METHOD,
        help="Early-stopping scheme (paper Section 4). 'combined' uses one counter on the "
             "pooled negation+speculation validation F1; 'separate' keeps a per-task best "
             "checkpoint from the same training run. Defaults to EARLY_STOPPING_METHOD from "
             "config.py.",
    )
    parser.add_argument(
        "--checkpoint-dir", default=CHECKPOINT_DIR,
        help="Directory to save checkpoints into. Each run writes to "
             "<dir>/<subtask>/<model>_<corpora>[_<scope method>]_<early stopping>/run<N>/: a "
             "checkpoint every --checkpoint-every epochs plus the best-validation weights. The "
             "middle levels keep the cells of the paper's experiment grid from overwriting one "
             "another. Scope models are saved as HuggingFace folders (epoch_NNN/, best/); cue "
             "models as state_dict files (epoch_NNN.pt, best.pt). See README 'Checkpoints' for "
             "how to load them.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
        help="Save a checkpoint every N epochs (0 = only save best/ at the end). "
             "Defaults to CHECKPOINT_EVERY from config.py.",
    )
    parser.add_argument(
        "--error-analysis", action=argparse.BooleanOptionalAction, default=ERROR_ANALYSIS_FOR_SCOPE,
        help="After scope-resolution training, additionally evaluate on test splits whose gold scope "
             "is delimited by punctuation ('punct') vs. the rest ('no_punct'). "
             "Defaults to ERROR_ANALYSIS_FOR_SCOPE from config.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    error_analysis = args.error_analysis
    subtask = args.subtask
    train_datasets = args.train_datasets
    test_datasets = args.test_datasets
    early_stopping_method = args.early_stopping
    num_runs = args.num_runs if args.num_runs is not None else runs_for(train_datasets)

    tag = experiment_tag(subtask, train_datasets, early_stopping_method)
    print(f"Subtask: {subtask} | train: {','.join(train_datasets)} | "
          f"test: {','.join(test_datasets)} | early stopping: {early_stopping_method} | "
          f"runs: {num_runs}")
    print(f"Experiment: {tag}")

    bioscope_full_papers_data = Data(args.bioscope_full_papers, dataset_name='bioscope', error_analysis=error_analysis)
    sfu_data = Data(args.sfu, dataset_name='sfu', error_analysis=error_analysis)
    bioscope_abstracts_data = Data(args.bioscope_abstracts, dataset_name='bioscope', error_analysis=error_analysis)

    for run_num in range(num_runs):
        # Namespaced by subtask and by experiment tag so different cells of the
        # paper's grid never overwrite each other's run<N> checkpoints.
        run_checkpoint_dir = os.path.join(args.checkpoint_dir, subtask, tag, f"run{run_num+1}")
        first_dataset = None
        other_datasets = []
        if 'sfu' in train_datasets:
            first_dataset = sfu_data
        if 'bioscope_full_papers' in train_datasets:
            if first_dataset == None:
                first_dataset = bioscope_full_papers_data
            else:
                other_datasets.append(bioscope_full_papers_data)
        if 'bioscope_abstracts' in train_datasets:
            if first_dataset == None:
                first_dataset = bioscope_abstracts_data
            else:
                other_datasets.append(bioscope_abstracts_data)

        if subtask == 'cue_detection':
            train_dl, val_dls, test_dls = first_dataset.get_cue_dataloader(val_size = VAL_SIZE, test_size = TEST_SIZE, other_datasets = other_datasets)

            test_dataloaders = {}
            idx = 0
            if 'sfu' in train_datasets:
                if 'sfu' in test_datasets:
                    test_dataloaders['sfu'] = test_dls[idx]
                idx+=1
            elif 'sfu' in test_datasets:
                sfu_dl, _, _ = sfu_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['sfu'] = sfu_dl
            if 'bioscope_full_papers' in train_datasets:
                if 'bioscope_full_papers' in test_datasets:
                    test_dataloaders['bioscope_full_papers'] = test_dls[idx]
                idx+=1
            elif 'bioscope_full_papers' in test_datasets:
                bioscope_full_papers_dl, _, _ = bioscope_full_papers_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
            if 'bioscope_abstracts' in train_datasets:
                if 'bioscope_abstracts' in test_datasets:
                    test_dataloaders['bioscope_abstracts'] = test_dls[idx]
                idx+=1
            elif 'bioscope_abstracts' in test_datasets:
                bioscope_abstracts_dl, _, _ = bioscope_abstracts_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl
            if early_stopping_method == 'separate':
                model = CueModel_Separate(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            elif early_stopping_method == 'combined':
                model = CueModel_Combined(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            else:
                raise ValueError("--early-stopping must be one of 'separate' and 'combined'")
            model.train(train_dl, val_dls, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(train_datasets), val_dl_name = ','.join(train_datasets), checkpoint_dir = run_checkpoint_dir, checkpoint_every = args.checkpoint_every)
            for k in test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                model.evaluate(test_dataloaders[k], test_dl_name = k)
        
            
        elif subtask == 'scope_resolution':
            train_dl, [neg_val_dl, spec_val_dl], [neg_test_dls, spec_test_dls] = first_dataset.get_scope_dataloader(val_size = VAL_SIZE, test_size = TEST_SIZE, other_datasets = other_datasets)
        
            neg_test_dataloaders = {}
            spec_test_dataloaders = {}
            neg_punct_test_dataloaders = {}
            spec_punct_test_dataloaders = {}
            neg_no_punct_test_dataloaders = {}
            spec_no_punct_test_dataloaders = {}
            idx = 0
            if 'sfu' in train_datasets:
                if 'sfu' in test_datasets:
                    neg_test_dataloaders['sfu'] = neg_test_dls[idx]
                    spec_test_dataloaders['sfu'] = spec_test_dls[idx]
                idx+=1
            elif 'sfu' in test_datasets:
                _, _, [neg_sfu_dl, spec_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001)
                neg_test_dataloaders['sfu'] = neg_sfu_dl[0]
                spec_test_dataloaders['sfu'] = spec_sfu_dl[0]
            if 'bioscope_full_papers' in train_datasets:
                if 'bioscope_full_papers' in test_datasets:
                    neg_test_dataloaders['bioscope_full_papers'] = neg_test_dls[idx]
                    spec_test_dataloaders['bioscope_full_papers'] = spec_test_dls[idx]
                idx+=1
            elif 'bioscope_full_papers' in test_datasets:
                _, _, [neg_bioscope_full_papers_dl, spec_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001)
                neg_test_dataloaders['bioscope_full_papers'] = neg_bioscope_full_papers_dl[0]
                spec_test_dataloaders['bioscope_full_papers'] = spec_bioscope_full_papers_dl[0]
            if 'bioscope_abstracts' in train_datasets:
                if 'bioscope_abstracts' in test_datasets:
                    neg_test_dataloaders['bioscope_abstracts'] = neg_test_dls[idx]
                    spec_test_dataloaders['bioscope_abstracts'] = spec_test_dls[idx]
                idx+=1
            elif 'bioscope_abstracts' in test_datasets:
                _, _, [neg_bioscope_abstracts_dl, spec_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.99999999, val_size = 0.00000001)
                neg_test_dataloaders['bioscope_abstracts'] = neg_bioscope_abstracts_dl[0]
                spec_test_dataloaders['bioscope_abstracts'] = spec_bioscope_abstracts_dl[0]

            # Error Analysis: build test dataloaders over the punctuation-delimited
            # ("punct") and remaining ("no_punct") gold-scope sentence splits.
            if error_analysis:
                if 'sfu' in test_datasets:
                    _, _, [neg_punct_sfu_dl, spec_punct_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
                    _, _, [neg_no_punct_sfu_dl, spec_no_punct_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
                    neg_punct_test_dataloaders['sfu_punct'] = neg_punct_sfu_dl[0]
                    spec_punct_test_dataloaders['sfu_punct'] = spec_punct_sfu_dl[0]
                    neg_no_punct_test_dataloaders['sfu_no_punct'] = neg_no_punct_sfu_dl[0]
                    spec_no_punct_test_dataloaders['sfu_no_punct'] = spec_no_punct_sfu_dl[0]
                if 'bioscope_full_papers' in test_datasets:
                    _, _, [neg_punct_bioscope_full_papers_dl, spec_punct_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
                    _, _, [neg_no_punct_bioscope_full_papers_dl, spec_no_punct_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
                    neg_punct_test_dataloaders['bioscope_full_papers_punct'] = neg_punct_bioscope_full_papers_dl[0]
                    spec_punct_test_dataloaders['bioscope_full_papers_punct'] = spec_punct_bioscope_full_papers_dl[0]
                    neg_no_punct_test_dataloaders['bioscope_full_papers_no_punct'] = neg_no_punct_bioscope_full_papers_dl[0]
                    spec_no_punct_test_dataloaders['bioscope_full_papers_no_punct'] = spec_no_punct_bioscope_full_papers_dl[0]
                if 'bioscope_abstracts' in test_datasets:
                    _, _, [neg_punct_bioscope_abstracts_dl, spec_punct_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
                    _, _, [neg_no_punct_bioscope_abstracts_dl, spec_no_punct_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
                    neg_punct_test_dataloaders['bioscope_abstracts_punct'] = neg_punct_bioscope_abstracts_dl[0]
                    spec_punct_test_dataloaders['bioscope_abstracts_punct'] = spec_punct_bioscope_abstracts_dl[0]
                    neg_no_punct_test_dataloaders['bioscope_abstracts_no_punct'] = neg_no_punct_bioscope_abstracts_dl[0]
                    spec_no_punct_test_dataloaders['bioscope_abstracts_no_punct'] = spec_no_punct_bioscope_abstracts_dl[0]

            if early_stopping_method == 'separate':
                model = ScopeModel_Separate(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            elif early_stopping_method == 'combined':
                model = ScopeModel_Combined(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            else:
                raise ValueError("--early-stopping must be one of 'separate' and 'combined'")
            model.train(train_dl, neg_val_dl, spec_val_dl, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(train_datasets), val_dl_name = ','.join(train_datasets), checkpoint_dir = run_checkpoint_dir, checkpoint_every = args.checkpoint_every)
            for k in neg_test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                model.evaluate(neg_test_dataloaders[k], test_dl_name = k, task = 'negation')
            for k in spec_test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                model.evaluate(spec_test_dataloaders[k], test_dl_name = k, task = 'speculation')

            # Error Analysis
            if error_analysis:
                analysis_runs = [
                    (neg_punct_test_dataloaders, 'negation'),
                    (spec_punct_test_dataloaders, 'speculation'),
                    (neg_no_punct_test_dataloaders, 'negation'),
                    (spec_no_punct_test_dataloaders, 'speculation'),
                ]
                for dataloaders, task in analysis_runs:
                    for k in dataloaders.keys():
                        if len(dataloaders[k].dataset) == 0:
                            print(f"Skipping {k} ({task}): split contains no sentences.")
                            continue
                        print(f"Evaluate on {k}:")
                        model.evaluate(dataloaders[k], test_dl_name = k, task = task)
            
        else:
            raise ValueError("Unsupported subtask. Supported values are: cue_detection, scope_resolution")

        print(f"\n\n************ RUN {run_num+1} DONE! **************\n\n")


if __name__ == "__main__":
    main()
