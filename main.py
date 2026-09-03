"""Command-line entry point: train and evaluate the cue detection /
scope resolution models on local BioScope / SFU Review data files."""
import argparse
import os

from config import (
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    EARLY_STOPPING_METHOD,
    EPOCHS,
    ERROR_ANALYSIS_FOR_SCOPE,
    NUM_RUNS,
    PATIENCE,
    INITIAL_LEARNING_RATE,
    SUBTASK,
    TEST_DATASETS,
    TRAIN_DATASETS,
)
from data import Data
from model import CueModel_Combined, CueModel_Separate, ScopeModel_Combined, ScopeModel_Separate


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
        "--checkpoint-dir", default=CHECKPOINT_DIR,
        help="Directory to save checkpoints into. Each run writes to <dir>/run<N>/: a checkpoint "
             "every --checkpoint-every epochs plus the best-validation weights. Scope models are "
             "saved as HuggingFace folders (epoch_NNN/, best/); cue models as state_dict files "
             "(epoch_NNN.pt, best.pt). See README 'Checkpoints' for how to load them.",
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

    bioscope_full_papers_data = Data(args.bioscope_full_papers, dataset_name='bioscope', error_analysis=error_analysis)
    sfu_data = Data(args.sfu, dataset_name='sfu', error_analysis=error_analysis)
    bioscope_abstracts_data = Data(args.bioscope_abstracts, dataset_name='bioscope', error_analysis=error_analysis)

    for run_num in range(NUM_RUNS):
        run_checkpoint_dir = os.path.join(args.checkpoint_dir, f"run{run_num+1}")
        first_dataset = None
        other_datasets = []
        if 'sfu' in TRAIN_DATASETS:
            first_dataset = sfu_data
        if 'bioscope_full_papers' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = bioscope_full_papers_data
            else:
                other_datasets.append(bioscope_full_papers_data)
        if 'bioscope_abstracts' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = bioscope_abstracts_data
            else:
                other_datasets.append(bioscope_abstracts_data)

        if SUBTASK == 'cue_detection':
            train_dl, val_dls, test_dls = first_dataset.get_cue_dataloader(other_datasets = other_datasets)

            test_dataloaders = {}
            idx = 0
            if 'sfu' in TRAIN_DATASETS:
                if 'sfu' in TEST_DATASETS:
                    test_dataloaders['sfu'] = test_dls[idx]
                idx+=1
            elif 'sfu' in TEST_DATASETS:
                sfu_dl, _, _ = sfu_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['sfu'] = sfu_dl
            if 'bioscope_full_papers' in TRAIN_DATASETS:
                if 'bioscope_full_papers' in TEST_DATASETS:
                    test_dataloaders['bioscope_full_papers'] = test_dls[idx]
                idx+=1
            elif 'bioscope_full_papers' in TEST_DATASETS:
                bioscope_full_papers_dl, _, _ = bioscope_full_papers_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
            if 'bioscope_abstracts' in TRAIN_DATASETS:
                if 'bioscope_abstracts' in TEST_DATASETS:
                    test_dataloaders['bioscope_abstracts'] = test_dls[idx]
                idx+=1
            elif 'bioscope_abstracts' in TEST_DATASETS:
                bioscope_abstracts_dl, _, _ = bioscope_abstracts_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl
            if EARLY_STOPPING_METHOD == 'separate':
                model = CueModel_Separate(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            elif EARLY_STOPPING_METHOD == 'combined':
                model = CueModel_Combined(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            else:
                raise ValueError("EARLY_STOPPING_METHOD must be one of 'separate' and 'combined'")
            model.train(train_dl, val_dls, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(TRAIN_DATASETS), val_dl_name = ','.join(TRAIN_DATASETS), checkpoint_dir = run_checkpoint_dir, checkpoint_every = args.checkpoint_every)
            for k in test_dataloaders.keys():
                print(f"Evaluate on {k}:")
                model.evaluate(test_dataloaders[k], test_dl_name = k)
        
            
        elif SUBTASK == 'scope_resolution':
            train_dl, [neg_val_dl, spec_val_dl], [neg_test_dls, spec_test_dls] = first_dataset.get_scope_dataloader(other_datasets = other_datasets)
        
            neg_test_dataloaders = {}
            spec_test_dataloaders = {}
            neg_punct_test_dataloaders = {}
            spec_punct_test_dataloaders = {}
            neg_no_punct_test_dataloaders = {}
            spec_no_punct_test_dataloaders = {}
            idx = 0
            if 'sfu' in TRAIN_DATASETS:
                if 'sfu' in TEST_DATASETS:
                    neg_test_dataloaders['sfu'] = neg_test_dls[idx]
                    spec_test_dataloaders['sfu'] = spec_test_dls[idx]
                idx+=1
            elif 'sfu' in TEST_DATASETS:
                _, _, [neg_sfu_dl, spec_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001)
                neg_test_dataloaders['sfu'] = neg_sfu_dl[0]
                spec_test_dataloaders['sfu'] = spec_sfu_dl[0]
            if 'bioscope_full_papers' in TRAIN_DATASETS:
                if 'bioscope_full_papers' in TEST_DATASETS:
                    neg_test_dataloaders['bioscope_full_papers'] = neg_test_dls[idx]
                    spec_test_dataloaders['bioscope_full_papers'] = spec_test_dls[idx]
                idx+=1
            elif 'bioscope_full_papers' in TEST_DATASETS:
                _, _, [neg_bioscope_full_papers_dl, spec_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001)
                neg_test_dataloaders['bioscope_full_papers'] = neg_bioscope_full_papers_dl[0]
                spec_test_dataloaders['bioscope_full_papers'] = spec_bioscope_full_papers_dl[0]
            if 'bioscope_abstracts' in TRAIN_DATASETS:
                if 'bioscope_abstracts' in TEST_DATASETS:
                    neg_test_dataloaders['bioscope_abstracts'] = neg_test_dls[idx]
                    spec_test_dataloaders['bioscope_abstracts'] = spec_test_dls[idx]
                idx+=1
            elif 'bioscope_abstracts' in TEST_DATASETS:
                _, _, [neg_bioscope_abstracts_dl, spec_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.99999999, val_size = 0.00000001)
                neg_test_dataloaders['bioscope_abstracts'] = neg_bioscope_abstracts_dl[0]
                spec_test_dataloaders['bioscope_abstracts'] = spec_bioscope_abstracts_dl[0]

            # Error Analysis: build test dataloaders over the punctuation-delimited
            # ("punct") and remaining ("no_punct") gold-scope sentence splits.
            if error_analysis:
                if 'sfu' in TEST_DATASETS:
                    _, _, [neg_punct_sfu_dl, spec_punct_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
                    _, _, [neg_no_punct_sfu_dl, spec_no_punct_sfu_dl] = sfu_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
                    neg_punct_test_dataloaders['sfu_punct'] = neg_punct_sfu_dl[0]
                    spec_punct_test_dataloaders['sfu_punct'] = spec_punct_sfu_dl[0]
                    neg_no_punct_test_dataloaders['sfu_no_punct'] = neg_no_punct_sfu_dl[0]
                    spec_no_punct_test_dataloaders['sfu_no_punct'] = spec_no_punct_sfu_dl[0]
                if 'bioscope_full_papers' in TEST_DATASETS:
                    _, _, [neg_punct_bioscope_full_papers_dl, spec_punct_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
                    _, _, [neg_no_punct_bioscope_full_papers_dl, spec_no_punct_bioscope_full_papers_dl] = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
                    neg_punct_test_dataloaders['bioscope_full_papers_punct'] = neg_punct_bioscope_full_papers_dl[0]
                    spec_punct_test_dataloaders['bioscope_full_papers_punct'] = spec_punct_bioscope_full_papers_dl[0]
                    neg_no_punct_test_dataloaders['bioscope_full_papers_no_punct'] = neg_no_punct_bioscope_full_papers_dl[0]
                    spec_no_punct_test_dataloaders['bioscope_full_papers_no_punct'] = spec_no_punct_bioscope_full_papers_dl[0]
                if 'bioscope_abstracts' in TEST_DATASETS:
                    _, _, [neg_punct_bioscope_abstracts_dl, spec_punct_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = True)
                    _, _, [neg_no_punct_bioscope_abstracts_dl, spec_no_punct_bioscope_abstracts_dl] = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.9999999, val_size = 0.00000001, error_analysis = True, punct_dl = False)
                    neg_punct_test_dataloaders['bioscope_abstracts_punct'] = neg_punct_bioscope_abstracts_dl[0]
                    spec_punct_test_dataloaders['bioscope_abstracts_punct'] = spec_punct_bioscope_abstracts_dl[0]
                    neg_no_punct_test_dataloaders['bioscope_abstracts_no_punct'] = neg_no_punct_bioscope_abstracts_dl[0]
                    spec_no_punct_test_dataloaders['bioscope_abstracts_no_punct'] = spec_no_punct_bioscope_abstracts_dl[0]

            if EARLY_STOPPING_METHOD == 'separate':
                model = ScopeModel_Separate(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            elif EARLY_STOPPING_METHOD == 'combined':
                model = ScopeModel_Combined(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
            else:
                raise ValueError("EARLY_STOPPING_METHOD must be one of 'separate' and 'combined'")
            model.train(train_dl, neg_val_dl, spec_val_dl, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(TRAIN_DATASETS), val_dl_name = ','.join(TRAIN_DATASETS), checkpoint_dir = run_checkpoint_dir, checkpoint_every = args.checkpoint_every)
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
