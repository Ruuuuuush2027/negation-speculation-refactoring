"""Cue detection and scope resolution models (combined vs. separate
early-stopping variants), built on top of the Bert/Roberta/XLNet backbones."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, XLNetForTokenClassification

from config import (
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    CUE_MODEL,
    DEVICE,
    SCOPE_LABELS,
    SCOPE_MODEL,
    SCOPE_METHOD,
    SUBTASK,
)
from early_stopping import EarlyStopping
from metrics import f1_cues, f1_scope, flat_accuracy, flat_accuracy_positive_cues, report_per_class_accuracy, scope_accuracy
from multihead_model import MultiHeadTokenClassifier


_tokenizer_cache = {}


def _load_tokenizer(model_name):
    """The (slow, as used in data.py) tokenizer saved alongside scope checkpoints."""
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return _tokenizer_cache[model_name]


class XLNetForTokenClassificationWithDropout(XLNetForTokenClassification):
    """HuggingFace's XLNetForTokenClassification has no dropout between the
    encoder and the classifier. The original notebook did not use HuggingFace's
    head; it defined its own (cell "Our implementation of
    XLNetForTokenClassification"), which applies nn.Dropout(config.dropout)
    before the projection -- the same shape as BERT's head. This subclass
    restores that layer so XLNet scope resolution trains as in the notebook.

    It adds no parameters, so a checkpoint saved from it loads with plain
    AutoModelForTokenClassification (dropout is inactive at eval time).
    """
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.classifier(self.dropout(outputs[0]))
        return (logits,)


def _scope_model_class():
    """The token-classification class for SCOPE_MODEL: HuggingFace's for every
    backbone except XLNet, whose head is patched to match the notebook."""
    return XLNetForTokenClassificationWithDropout if 'xlnet' in SCOPE_MODEL else AutoModelForTokenClassification


def save_checkpoint(model, path, model_name):
    """Save a checkpoint of `model` at `path` (given without extension).

    * Scope models are plain HuggingFace models, so they are saved as a
      HuggingFace folder `<path>/` (config + weights + tokenizer), loadable
      with `AutoModelForTokenClassification.from_pretrained(path)`.
    * Cue models are the custom two-headed `MultiHeadTokenClassifier`, which
      HuggingFace cannot load, so their `state_dict` goes to `<path>.pt`, to be
      loaded with `MultiHeadTokenClassifier(...).load_state_dict(torch.load(...))`.
    """
    if isinstance(model, MultiHeadTokenClassifier):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(model.state_dict(), path + ".pt")
        print(f"Saved checkpoint to {path}.pt")
    else:
        model.save_pretrained(path)
        _load_tokenizer(model_name).save_pretrained(path)
        print(f"Saved checkpoint to {path}/")


def save_best_checkpoint(model, checkpoint_dir, model_name, task=None):
    """Called at the end of training with the best-validation weights loaded.

    Early stopping already saved those weights as a state_dict (`best.pt`, or
    `best_negation.pt` / `best_speculation.pt`), which is the final checkpoint
    for cue models. Scope models are additionally saved as a HuggingFace folder
    `best/` (`best/<task>/` for the 'separate' early-stopping method).
    """
    if isinstance(model, MultiHeadTokenClassifier):
        return
    path = os.path.join(checkpoint_dir, 'best') if task is None else os.path.join(checkpoint_dir, 'best', task)
    save_checkpoint(model, path, model_name)

def word_level_predictions(logits, word_mask, attention_mask=None):
    """Collapse per-sub-word logits into one predicted label per word.

    ``word_mask`` is 1 on the first sub-word of every word and 0 on its
    continuation sub-words (built in data.py); ``attention_mask`` is 0 on
    padding, which is skipped. A word's label is the argmax of the mean of
    the logits of all of its sub-words.
    """
    preds = []
    current = []
    for idx, (logit, first) in enumerate(zip(logits, word_mask)):
        if attention_mask is not None and attention_mask[idx] == 0:
            continue
        if first == 1:
            if current:
                preds.append(int(np.argmax(np.mean(current, axis=0))))
            current = [logit]
        elif current:
            current.append(logit)
    if current:
        preds.append(int(np.argmax(np.mean(current, axis=0))))
    return preds

class CueModel_Combined:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Cue_Detection.pickle', device = DEVICE, learning_rate = 3e-5, class_weight = [100, 100, 100, 1, 0], num_labels = 5):
        self.model_name = CUE_MODEL
        if train == True:
            self.model = MultiHeadTokenClassifier(CUE_MODEL, num_labels=num_labels)
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        if device == 'cuda':
            self.model.cuda()
            #self.model_2.cuda()
        else:
            self.model.cpu()
            #self.model_2.cpu()
            
        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3, checkpoint_dir = CHECKPOINT_DIR, checkpoint_every = CHECKPOINT_EVERY):
        
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0}
        train_loss = []
        valid_loss = []
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, 'best.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path = best_path)
        #early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint2.pt')
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} batches", leave=False)):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch
                logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                loss = loss_neg + loss_spec
                loss.backward()
                tr_loss += loss.item()
                if step % 100 == 0:
                    tqdm.write(f"Batch {step}, loss {loss.item()}")
                train_loss.append(loss.item())
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            if checkpoint_every and epoch % checkpoint_every == 0:
                save_checkpoint(self.model, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}"), self.model_name)
            
            self.model.eval()
            eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
            predictions_neg , true_labels_neg, predictions_spec , true_labels_spec, ip_mask = [], [], [], [], []
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch

                    with torch.no_grad():
                        logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_labels_neg = b_labels_neg.view(-1)[active_loss]
                        active_labels_spec = b_labels_spec.view(-1)[active_loss]
                        tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                        tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                        tmp_eval_loss = (tmp_eval_loss_neg.mean().item()+tmp_eval_loss_spec.mean().item())/2
                        
                    logits_neg = logits_neg.detach().cpu().numpy()
                    logits_spec = logits_spec.detach().cpu().numpy()
                    label_ids_neg = b_labels_neg.to('cpu').numpy()
                    label_ids_spec = b_labels_spec.to('cpu').numpy()
                    
                    mymasks = b_mymasks.to('cpu').numpy()
                    
                    logits_neg = [list(p) for p in logits_neg]
                    logits_spec = [list(p) for p in logits_spec]
                    
                    actual_logits_neg = []
                    actual_label_ids_neg = []
                    actual_logits_spec = []
                    actual_label_ids_spec = []
                    
                    attention = b_input_mask.to('cpu').numpy()
                    for l_n,lid_n,l_s,lid_s,m,a in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks, attention):
                        actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                        actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])
                        actual_logits_neg.append(word_level_predictions(l_n, m, a))
                        actual_logits_spec.append(word_level_predictions(l_s, m, a))
                        
                    logits_neg = actual_logits_neg
                    label_ids_neg = actual_label_ids_neg
                    logits_spec = actual_logits_spec
                    label_ids_spec = actual_label_ids_spec
                    
                    predictions_neg.append(logits_neg)
                    true_labels_neg.append(label_ids_neg)
                    predictions_spec.append(logits_spec)
                    true_labels_spec.append(label_ids_spec)
                    
                    tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
                    #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
                    eval_loss += tmp_eval_loss
                    valid_loss.append(tmp_eval_loss)
                    eval_accuracy += tmp_eval_accuracy
                    
                    nb_eval_examples += b_input_ids.size(0)
                    nb_eval_steps += 1
            
            eval_loss = eval_loss/nb_eval_steps
            
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
            labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
            pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
            pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
            labels_flat_neg = [l for l in labels_flat_neg if l!=4]
            labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
            pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
            pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
            labels_flat_spec = [l for l in labels_flat_spec if l!=4]
            report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
            report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
            print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
            labels_flat = labels_flat_neg + labels_flat_spec
            pred_flat = pred_flat_neg + pred_flat_spec
            p,r,f1 = f1_cues(labels_flat, pred_flat)
            #p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)

            if f1>return_dict['Best F1'] and early_stopping.early_stop == False:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            if early_stopping.early_stop == False:
                early_stopping(f1, self.model)
            else:
                print("Early stopping")
                break

            '''labels_flat = [int(i!=3) for i in labels_flat]
            pred_flat = [int(i!=3) for i in pred_flat]
            print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))'''
            
        self.model.load_state_dict(torch.load(best_path))
        save_best_checkpoint(self.model, checkpoint_dir, self.model_name)
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

    def evaluate(self, test_dataloader, test_dl_name):
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Negation - Precision": 0,
                       "Negation - Recall": 0,
                       "Negation - F1": 0,
                       "Speculation - Precision": 0,
                       "Speculation - Recall": 0,
                       "Speculation - F1": 0}
        self.model.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
        predictions_neg, true_labels_neg, predictions_spec, true_labels_spec, ip_mask = [], [], [], [], []
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch
            
            with torch.no_grad():
                logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                #_, logits_spec = self.model_2(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                tmp_eval_loss = tmp_eval_loss_neg+tmp_eval_loss_spec
                logits_neg = logits_neg.detach().cpu().numpy()
                logits_spec = logits_spec.detach().cpu().numpy()

            label_ids_neg = b_labels_neg.to('cpu').numpy()
            label_ids_spec = b_labels_spec.to('cpu').numpy()

            mymasks = b_mymasks.to('cpu').numpy()
            logits_neg = [list(p) for p in logits_neg]
            logits_spec = [list(p) for p in logits_spec]
                
            actual_logits_neg = []
            actual_label_ids_neg = []
            actual_logits_spec = []
            actual_label_ids_spec = []

            attention = b_input_mask.to('cpu').numpy()
            for l_n,lid_n,l_s,lid_s,m,a in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks, attention):
                actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])
                actual_logits_neg.append(word_level_predictions(l_n, m, a))
                actual_logits_spec.append(word_level_predictions(l_s, m, a))
                
            logits_neg = actual_logits_neg
            label_ids_neg = actual_label_ids_neg
            logits_spec = actual_logits_spec
            label_ids_spec = actual_label_ids_spec
            
            predictions_neg.append(logits_neg)
            true_labels_neg.append(label_ids_neg)
            predictions_spec.append(logits_spec)
            true_labels_spec.append(label_ids_spec)
            
            tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
            #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
            eval_loss += tmp_eval_loss
            #valid_loss.append(tmp_eval_loss)
            eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
        labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
        pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
        pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
        labels_flat_neg = [l for l in labels_flat_neg if l!=4]
        report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
        labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
        pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
        pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
        labels_flat_spec = [l for l in labels_flat_spec if l!=4]
        report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
        print(classification_report(labels_flat_neg, pred_flat_neg))
        print(classification_report(labels_flat_spec, pred_flat_spec))
        print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
        print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
        p_n,r_n,f1_n = f1_cues(labels_flat_neg, pred_flat_neg)
        p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)
        return_dict['Negation - F1'] = f1_n
        return_dict['Negation - Precision'] = p_n
        return_dict['Negation - Recall'] = r_n
        return_dict['Speculation - F1'] = f1_s
        return_dict['Speculation - Precision'] = p_s
        return_dict['Speculation - Recall'] = r_s

        return return_dict

class ScopeModel_Combined:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Scope_Resolution_Augment.pickle', device = DEVICE, learning_rate = 3e-5):
        self.model_name = SCOPE_MODEL
        self.task = SUBTASK
        self.num_labels = 2
        self.scope_method = SCOPE_METHOD
        if train == True:
            self.model = _scope_model_class().from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, id2label=SCOPE_LABELS, label2id={v: k for k, v in SCOPE_LABELS.items()})
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        if device=='cuda':
            self.model.cuda()
            #self.model_2.cuda()

        else:
            self.model.cpu()
            #self.model_2.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)
  
    def train(self, train_dataloader, valid_dataloader_negation, valid_dataloader_speculation, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3, checkpoint_dir = CHECKPOINT_DIR, checkpoint_every = CHECKPOINT_EVERY):
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multitask Scope Resolution - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0,
                       }
        train_loss = []
        valid_loss = []
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, 'best.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path = best_path)
        #early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = 'checkpoint2.pt')

        loss_fn = CrossEntropyLoss()
        for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} batches", leave=False)):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #2 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                train_loss.append(loss.item())
                if step%100 == 0:
                    tqdm.write(f"Batch {step}, loss {loss.item()}")
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            if checkpoint_every and epoch % checkpoint_every == 0:
                save_checkpoint(self.model, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}"), self.model_name)
            
            self.model.eval()
            
            eval_loss_neg, eval_accuracy_neg, eval_scope_accuracy_neg = 0, 0, 0
            nb_eval_steps_neg, nb_eval_examples_neg = 0, 0
            predictions_negation , true_labels_negation, ip_mask_neg = [], [], []
            for batch in valid_dataloader_negation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)
                    
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()
                    
                logits = [list(p) for p in logits]
                
                actual_logits = []
                actual_label_ids = []
                
                attention = b_input_mask.to('cpu').numpy()
                for l,lid,m,a in zip(logits, label_ids, mymasks, attention):
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    actual_logits.append(word_level_predictions(l, m, a))
                    
                predictions_negation.append(actual_logits)
                true_labels_negation.append(actual_label_ids)    
                
                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_neg += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_neg += tmp_eval_loss.mean().item()
                eval_accuracy_neg += tmp_eval_accuracy

                nb_eval_examples_neg += len(b_input_ids)
                nb_eval_steps_neg += 1
            
            eval_loss_neg = eval_loss_neg/nb_eval_steps_neg
            print("Negation Validation loss: {}".format(eval_loss_neg))
            print("Negation Validation Accuracy: {}".format(eval_accuracy_neg/nb_eval_steps_neg))
            print("Negation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_neg/nb_eval_steps_neg))
            f1_scope([j for i in true_labels_negation for j in i], [j for i in predictions_negation for j in i], level='scope')
            labels_flat_neg = [l_ii for l in true_labels_negation for l_i in l for l_ii in l_i]
            pred_flat_neg = [p_ii for p in predictions_negation for p_i in p for p_ii in p_i]
            
            #Speculation
            eval_loss_spec, eval_accuracy_spec, eval_scope_accuracy_spec = 0, 0, 0
            nb_eval_steps_spec, nb_eval_examples_spec = 0, 0
            predictions_speculation , true_labels_speculation, ip_mask = [], [], [] 
            for batch in valid_dataloader_speculation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)
                    
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()
                    
                logits = [list(p) for p in logits]
                
                actual_logits = []
                actual_label_ids = []
                
                attention = b_input_mask.to('cpu').numpy()
                for l,lid,m,a in zip(logits, label_ids, mymasks, attention):
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    actual_logits.append(word_level_predictions(l, m, a))
                    
                predictions_speculation.append(actual_logits)
                true_labels_speculation.append(actual_label_ids)    
                
                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_spec += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_spec += tmp_eval_loss.mean().item()
                eval_accuracy_spec += tmp_eval_accuracy

                nb_eval_examples_spec += len(b_input_ids)
                nb_eval_steps_spec += 1

            eval_loss_spec = eval_loss_spec/nb_eval_steps_spec
            print("Speculation Validation loss: {}".format(eval_loss_spec))
            print("Speculation Validation Accuracy: {}".format(eval_accuracy_spec/nb_eval_steps_spec))
            print("Speculation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_spec/nb_eval_steps_spec))
            f1_scope([j for i in true_labels_speculation for j in i], [j for i in predictions_speculation for j in i], level='scope')
            labels_flat_spec = [l_ii for l in true_labels_speculation for l_i in l for l_ii in l_i]
            pred_flat_spec = [p_ii for p in predictions_speculation for p_i in p for p_ii in p_i]
            labels_flat = labels_flat_neg + labels_flat_spec
            pred_flat = pred_flat_neg + pred_flat_spec
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Best F1'] and early_stopping.early_stop == False:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            print("F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            if early_stopping.early_stop == False:
                early_stopping(f1, self.model)
            else:
                print("Early stopping")
                break
        
        self.model.load_state_dict(torch.load(best_path))
        save_best_checkpoint(self.model, checkpoint_dir, self.model_name)
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

    def evaluate(self, test_dataloader, test_dl_name = "SFU", task = "Negation"):
        return_dict = {"Task": f"Multitask Scope Resolution - {task} - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss()
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)
                
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            b_input_ids = b_input_ids.to('cpu').numpy()
            
            mymasks = b_mymasks.to('cpu').numpy()
                
            logits = [list(p) for p in logits]
            
            actual_logits = []
            actual_label_ids = []
            
            attention = b_input_mask.to('cpu').numpy()
            for l,lid,m,a in zip(logits, label_ids, mymasks, attention):
                actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                actual_logits.append(word_level_predictions(l, m, a))
                
            predictions.append(actual_logits)
            true_labels.append(actual_label_ids)

            tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
            tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
            eval_scope_accuracy += tmp_eval_scope_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += len(b_input_ids)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
        f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
        p = classification_dict["1"]["precision"]
        r = classification_dict["1"]["recall"]
        f1 = classification_dict["1"]["f1-score"]
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        print("Classification Report:")
        print(classification_report(labels_flat, pred_flat))
        return return_dict


class CueModel_Separate:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Cue_Detection.pickle', device = DEVICE, learning_rate = 3e-5, class_weight = [100, 100, 100, 1, 0], num_labels = 5):
        self.model_name = CUE_MODEL
        if train == True:
            self.model = MultiHeadTokenClassifier(CUE_MODEL, num_labels=num_labels)
            self.model_2 = MultiHeadTokenClassifier(CUE_MODEL, num_labels=num_labels)
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        if device == 'cuda':
            self.model.cuda()
            self.model_2.cuda()
        else:
            self.model.cpu()
            self.model_2.cpu()
            
        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3, checkpoint_dir = CHECKPOINT_DIR, checkpoint_every = CHECKPOINT_EVERY):
        
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Negation - Best Precision": 0,
                       "Negation - Best Recall": 0,
                       "Negation - Best F1": 0,
                       "Speculation - Best Precision": 0,
                       "Speculation - Best Recall": 0,
                       "Speculation - Best F1": 0}
        train_loss = []
        valid_loss = []
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, 'best_negation.pt')
        spec_best_path = os.path.join(checkpoint_dir, 'best_speculation.pt')
        early_stopping_neg = EarlyStopping(patience=patience, verbose=True, save_path = best_path)
        early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = spec_best_path)
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} batches", leave=False)):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch
                logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                loss = loss_neg + loss_spec
                loss.backward()
                tr_loss += loss.item()
                if step % 100 == 0:
                    tqdm.write(f"Batch {step}, loss {loss.item()}")
                train_loss.append(loss.item())
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            if checkpoint_every and epoch % checkpoint_every == 0:
                save_checkpoint(self.model, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}"), self.model_name)
            
            self.model.eval()
            eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
            predictions_neg , true_labels_neg, predictions_spec , true_labels_spec, ip_mask = [], [], [], [], []
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch

                    with torch.no_grad():
                        logits_neg, logits_spec = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_labels_neg = b_labels_neg.view(-1)[active_loss]
                        active_labels_spec = b_labels_spec.view(-1)[active_loss]
                        tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                        tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                        tmp_eval_loss = (tmp_eval_loss_neg.mean().item()+tmp_eval_loss_spec.mean().item())/2
                        
                    logits_neg = logits_neg.detach().cpu().numpy()
                    logits_spec = logits_spec.detach().cpu().numpy()
                    label_ids_neg = b_labels_neg.to('cpu').numpy()
                    label_ids_spec = b_labels_spec.to('cpu').numpy()
                    
                    mymasks = b_mymasks.to('cpu').numpy()
                    
                    logits_neg = [list(p) for p in logits_neg]
                    logits_spec = [list(p) for p in logits_spec]
                    
                    actual_logits_neg = []
                    actual_label_ids_neg = []
                    actual_logits_spec = []
                    actual_label_ids_spec = []
                    
                    attention = b_input_mask.to('cpu').numpy()
                    for l_n,lid_n,l_s,lid_s,m,a in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks, attention):
                        actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                        actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])
                        actual_logits_neg.append(word_level_predictions(l_n, m, a))
                        actual_logits_spec.append(word_level_predictions(l_s, m, a))
                        
                    logits_neg = actual_logits_neg
                    label_ids_neg = actual_label_ids_neg
                    logits_spec = actual_logits_spec
                    label_ids_spec = actual_label_ids_spec
                    
                    predictions_neg.append(logits_neg)
                    true_labels_neg.append(label_ids_neg)
                    predictions_spec.append(logits_spec)
                    true_labels_spec.append(label_ids_spec)
                    
                    tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
                    #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
                    eval_loss += tmp_eval_loss
                    valid_loss.append(tmp_eval_loss)
                    eval_accuracy += tmp_eval_accuracy
                    
                    nb_eval_examples += b_input_ids.size(0)
                    nb_eval_steps += 1
            
            eval_loss = eval_loss/nb_eval_steps
            
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
            labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
            pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
            pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
            labels_flat_neg = [l for l in labels_flat_neg if l!=4]
            labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
            pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
            pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
            labels_flat_spec = [l for l in labels_flat_spec if l!=4]
            report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
            report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print(classification_report(labels_flat_neg, pred_flat_neg))
            print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
            print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
            p_n,r_n,f1_n = f1_cues(labels_flat_neg, pred_flat_neg)
            p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)

            if f1_n>return_dict['Negation - Best F1'] and early_stopping_neg.early_stop == False:
                return_dict['Negation - Best F1'] = f1_n
                return_dict['Negation - Best Precision'] = p_n
                return_dict['Negation - Best Recall'] = r_n
            if early_stopping_neg.early_stop == False:
                early_stopping_neg(f1_n, self.model)
            if f1_s>return_dict['Speculation - Best F1'] and early_stopping_spec.early_stop == False:
                return_dict['Speculation - Best F1'] = f1_s
                return_dict['Speculation - Best Precision'] = p_s
                return_dict['Speculation - Best Recall'] = r_s
            if early_stopping_spec.early_stop == False:
                early_stopping_spec(f1_s, self.model)
        
            if early_stopping_neg.early_stop and early_stopping_spec.early_stop:
                print("Early stopping")
                break

            '''labels_flat = [int(i!=3) for i in labels_flat]
            pred_flat = [int(i!=3) for i in pred_flat]
            print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))'''
            
        self.model.load_state_dict(torch.load(best_path))
        self.model_2.load_state_dict(torch.load(spec_best_path))
        # model holds the best-negation weights, model_2 the best-speculation ones.
        save_best_checkpoint(self.model, checkpoint_dir, self.model_name, task='negation')
        save_best_checkpoint(self.model_2, checkpoint_dir, self.model_name, task='speculation')
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

    def evaluate(self, test_dataloader, test_dl_name):
        return_dict = {"Task": f"Multidata Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Negation - Precision": 0,
                       "Negation - Recall": 0,
                       "Negation - F1": 0,
                       "Speculation - Precision": 0,
                       "Speculation - Recall": 0,
                       "Speculation - F1": 0}
        self.model.eval()
        self.model_2.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
        predictions_neg, true_labels_neg, predictions_spec, true_labels_spec, ip_mask = [], [], [], [], []
        loss_fn_neg = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        loss_fn_spec = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels_neg, b_labels_spec, b_mymasks = batch
            
            with torch.no_grad():
                logits_neg, _ = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                _, logits_spec = self.model_2(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits_neg = logits_neg.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_neg = b_labels_neg.view(-1)[active_loss]
                active_logits_spec = logits_spec.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels_spec = b_labels_spec.view(-1)[active_loss]
                tmp_eval_loss_neg = loss_fn_neg(active_logits_neg, active_labels_neg)
                tmp_eval_loss_spec = loss_fn_spec(active_logits_spec, active_labels_spec)
                tmp_eval_loss = tmp_eval_loss_neg+tmp_eval_loss_spec
                logits_neg = logits_neg.detach().cpu().numpy()
                logits_spec = logits_spec.detach().cpu().numpy()

            label_ids_neg = b_labels_neg.to('cpu').numpy()
            label_ids_spec = b_labels_spec.to('cpu').numpy()

            mymasks = b_mymasks.to('cpu').numpy()
            logits_neg = [list(p) for p in logits_neg]
            logits_spec = [list(p) for p in logits_spec]
                
            actual_logits_neg = []
            actual_label_ids_neg = []
            actual_logits_spec = []
            actual_label_ids_spec = []

            attention = b_input_mask.to('cpu').numpy()
            for l_n,lid_n,l_s,lid_s,m,a in zip(logits_neg, label_ids_neg, logits_spec, label_ids_spec, mymasks, attention):
                actual_label_ids_neg.append([i for i,j in zip(lid_n, m) if j==1])
                actual_label_ids_spec.append([i for i,j in zip(lid_s, m) if j==1])
                actual_logits_neg.append(word_level_predictions(l_n, m, a))
                actual_logits_spec.append(word_level_predictions(l_s, m, a))
                
            logits_neg = actual_logits_neg
            label_ids_neg = actual_label_ids_neg
            logits_spec = actual_logits_spec
            label_ids_spec = actual_label_ids_spec
            
            predictions_neg.append(logits_neg)
            true_labels_neg.append(label_ids_neg)
            predictions_spec.append(logits_spec)
            true_labels_spec.append(label_ids_spec)
            
            tmp_eval_accuracy = (flat_accuracy(logits_neg, label_ids_neg)+flat_accuracy(logits_spec, label_ids_spec))/2
            #tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
            eval_loss += tmp_eval_loss
            #valid_loss.append(tmp_eval_loss)
            eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        #print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
        labels_flat_neg = [l_ii for l in true_labels_neg for l_i in l for l_ii in l_i]
        pred_flat_neg = [p_ii for p in predictions_neg for p_i in p for p_ii in p_i]
        pred_flat_neg = [p for p,l in zip(pred_flat_neg, labels_flat_neg) if l!=4]
        labels_flat_neg = [l for l in labels_flat_neg if l!=4]
        report_per_class_accuracy(labels_flat_neg, pred_flat_neg)
        labels_flat_spec = [l_ii for l in true_labels_spec for l_i in l for l_ii in l_i]
        pred_flat_spec = [p_ii for p in predictions_spec for p_i in p for p_ii in p_i]
        pred_flat_spec = [p for p,l in zip(pred_flat_spec, labels_flat_spec) if l!=4]
        labels_flat_spec = [l for l in labels_flat_spec if l!=4]
        report_per_class_accuracy(labels_flat_spec, pred_flat_spec)
        print(classification_report(labels_flat_neg, pred_flat_neg))
        print(classification_report(labels_flat_spec, pred_flat_spec))
        print("Negation: F1-Score Overall: {}".format(f1_score(labels_flat_neg,pred_flat_neg, average='weighted')))
        print("Speculation: F1-Score Overall: {}".format(f1_score(labels_flat_spec,pred_flat_spec, average='weighted')))
        p_n,r_n,f1_n = f1_cues(labels_flat_neg, pred_flat_neg)
        p_s,r_s,f1_s = f1_cues(labels_flat_spec, pred_flat_spec)
        return_dict['Negation - F1'] = f1_n
        return_dict['Negation - Precision'] = p_n
        return_dict['Negation - Recall'] = r_n
        return_dict['Speculation - F1'] = f1_s
        return_dict['Speculation - Precision'] = p_s
        return_dict['Speculation - Recall'] = r_s

        return return_dict

class ScopeModel_Separate:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Scope_Resolution_Augment.pickle', device = DEVICE, learning_rate = 3e-5):
        self.model_name = SCOPE_MODEL
        self.task = SUBTASK
        self.num_labels = 2
        self.scope_method = SCOPE_METHOD
        if train == True:
            self.model = _scope_model_class().from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, id2label=SCOPE_LABELS, label2id={v: k for k, v in SCOPE_LABELS.items()})
            self.model_2 = _scope_model_class().from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, id2label=SCOPE_LABELS, label2id={v: k for k, v in SCOPE_LABELS.items()})
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        if device=='cuda':
            self.model.cuda()
            self.model_2.cuda()

        else:
            self.model.cpu()
            self.model_2.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)
  
    def train(self, train_dataloader, valid_dataloader_negation, valid_dataloader_speculation, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3, checkpoint_dir = CHECKPOINT_DIR, checkpoint_every = CHECKPOINT_EVERY):
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"Multitask Scope Resolution - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Negation - Best Precision": 0,
                       "Negation - Best Recall": 0,
                       "Negation - Best F1": 0,
                       "Speculation - Best Precision": 0,
                       "Speculation - Best Recall": 0,
                       "Speculation - Best F1": 0}
        train_loss = []
        valid_loss = []
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path = os.path.join(checkpoint_dir, 'best_negation.pt')
        spec_best_path = os.path.join(checkpoint_dir, 'best_speculation.pt')
        early_stopping_neg = EarlyStopping(patience=patience, verbose=True, save_path = best_path)
        early_stopping_spec = EarlyStopping(patience=patience, verbose=True, save_path = spec_best_path)

        loss_fn = CrossEntropyLoss()
        for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} batches", leave=False)):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #2 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                train_loss.append(loss.item())
                if step%100 == 0:
                    tqdm.write(f"Batch {step}, loss {loss.item()}")
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            if checkpoint_every and epoch % checkpoint_every == 0:
                save_checkpoint(self.model, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}"), self.model_name)
            
            self.model.eval()
            
            eval_loss_neg, eval_accuracy_neg, eval_scope_accuracy_neg = 0, 0, 0
            nb_eval_steps_neg, nb_eval_examples_neg = 0, 0
            predictions_negation , true_labels_negation, ip_mask_neg = [], [], []
            for batch in valid_dataloader_negation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)
                    
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()
                    
                logits = [list(p) for p in logits]
                
                actual_logits = []
                actual_label_ids = []
                
                attention = b_input_mask.to('cpu').numpy()
                for l,lid,m,a in zip(logits, label_ids, mymasks, attention):
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    actual_logits.append(word_level_predictions(l, m, a))
                    
                predictions_negation.append(actual_logits)
                true_labels_negation.append(actual_label_ids)    
                
                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_neg += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_neg += tmp_eval_loss.mean().item()
                eval_accuracy_neg += tmp_eval_accuracy

                nb_eval_examples_neg += len(b_input_ids)
                nb_eval_steps_neg += 1
            
            eval_loss_neg = eval_loss_neg/nb_eval_steps_neg
            print("Negation Validation loss: {}".format(eval_loss_neg))
            print("Negation Validation Accuracy: {}".format(eval_accuracy_neg/nb_eval_steps_neg))
            print("Negation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_neg/nb_eval_steps_neg))
            f1_scope([j for i in true_labels_negation for j in i], [j for i in predictions_negation for j in i], level='scope')
            labels_flat = [l_ii for l in true_labels_negation for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions_negation for p_i in p for p_ii in p_i]
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Negation - Best F1'] and early_stopping_neg.early_stop == False:
                return_dict['Negation - Best F1'] = f1
                return_dict['Negation - Best Precision'] = p
                return_dict['Negation - Best Recall'] = r
            print("Negation: F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            if early_stopping_neg.early_stop == False:
                early_stopping_neg(f1, self.model)
            
            #Speculation
            eval_loss_spec, eval_accuracy_spec, eval_scope_accuracy_spec = 0, 0, 0
            nb_eval_steps_spec, nb_eval_examples_spec = 0, 0
            predictions_speculation , true_labels_speculation, ip_mask = [], [], [] 
            for batch in valid_dataloader_speculation:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                with torch.no_grad():
                    logits = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask)[0]
                    active_loss = b_input_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = b_labels.view(-1)[active_loss]
                    tmp_eval_loss = loss_fn(active_logits, active_labels)
                    
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_input_ids = b_input_ids.to('cpu').numpy()

                mymasks = b_mymasks.to('cpu').numpy()
                    
                logits = [list(p) for p in logits]
                
                actual_logits = []
                actual_label_ids = []
                
                attention = b_input_mask.to('cpu').numpy()
                for l,lid,m,a in zip(logits, label_ids, mymasks, attention):
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    actual_logits.append(word_level_predictions(l, m, a))
                    
                predictions_speculation.append(actual_logits)
                true_labels_speculation.append(actual_label_ids)    
                
                tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
                tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
                eval_scope_accuracy_spec += tmp_eval_scope_accuracy
                valid_loss.append(tmp_eval_loss.mean().item())

                eval_loss_spec += tmp_eval_loss.mean().item()
                eval_accuracy_spec += tmp_eval_accuracy

                nb_eval_examples_spec += len(b_input_ids)
                nb_eval_steps_spec += 1

            eval_loss_spec = eval_loss_spec/nb_eval_steps_spec
            print("Speculation Validation loss: {}".format(eval_loss_spec))
            print("Speculation Validation Accuracy: {}".format(eval_accuracy_spec/nb_eval_steps_spec))
            print("Speculation Validation Accuracy Scope Level: {}".format(eval_scope_accuracy_spec/nb_eval_steps_spec))
            f1_scope([j for i in true_labels_speculation for j in i], [j for i in predictions_speculation for j in i], level='scope')
            labels_flat = [l_ii for l in true_labels_speculation for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions_speculation for p_i in p for p_ii in p_i]
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Speculation - Best F1'] and early_stopping_spec.early_stop == False:
                return_dict['Speculation - Best F1'] = f1
                return_dict['Speculation - Best Precision'] = p
                return_dict['Speculation - Best Recall'] = r
            print("F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            if early_stopping_spec.early_stop == False:
                early_stopping_spec(f1, self.model)
            if early_stopping_neg.early_stop and early_stopping_spec.early_stop:
                print("Early stopping")
                break
        
        self.model.load_state_dict(torch.load(best_path))
        self.model_2.load_state_dict(torch.load(spec_best_path))
        # model holds the best-negation weights, model_2 the best-speculation ones.
        save_best_checkpoint(self.model, checkpoint_dir, self.model_name, task='negation')
        save_best_checkpoint(self.model_2, checkpoint_dir, self.model_name, task='speculation')
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

    def evaluate(self, test_dataloader, test_dl_name = "SFU", task = "negation"):
        return_dict = {"Task": f"Multitask Separate Scope Resolution - {task} - {self.scope_method}",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        self.model_2.eval()
        valid_loss = []
        eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss()
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch

            with torch.no_grad():
                if task == 'negation':
                    logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                else:
                    logits = self.model_2(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)
                
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            b_input_ids = b_input_ids.to('cpu').numpy()
            
            mymasks = b_mymasks.to('cpu').numpy()
                
            logits = [list(p) for p in logits]
            
            actual_logits = []
            actual_label_ids = []
            
            attention = b_input_mask.to('cpu').numpy()
            for l,lid,m,a in zip(logits, label_ids, mymasks, attention):
                actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                actual_logits.append(word_level_predictions(l, m, a))
                
            predictions.append(actual_logits)
            true_labels.append(actual_label_ids)

            tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
            tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
            eval_scope_accuracy += tmp_eval_scope_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += len(b_input_ids)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
        f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
        p = classification_dict["1"]["precision"]
        r = classification_dict["1"]["recall"]
        f1 = classification_dict["1"]["f1-score"]
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        print("Classification Report:")
        print(classification_report(labels_flat, pred_flat))
        return return_dict
