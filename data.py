"""Dataset parsing (BioScope / SFU Review corpora) and DataLoader construction."""
import html
import os
import re
import string

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import CUE_MODEL, MAX_LEN, SCOPE_MODEL, SCOPE_METHOD, bs


def pad_sequences(sequences, maxlen, dtype="long", value=0, padding="post", truncating="post"):
    """Minimal drop-in replacement for keras.preprocessing.sequence.pad_sequences."""
    padded = []
    for seq in sequences:
        seq = list(seq)
        if len(seq) >= maxlen:
            seq = seq[:maxlen] if truncating == "post" else seq[-maxlen:]
        else:
            fill = [value] * (maxlen - len(seq))
            seq = seq + fill if padding == "post" else fill + seq
        padded.append(seq)
    return np.array(padded, dtype=np.int64)


def split_tags(line):
    """Split one line of annotated XML into alternating text / ``<tag>`` tokens.

    HTML entities are unescaped *after* the split (the notebook unescaped the
    whole line first), so text like ``p &lt; 0.05`` becomes ``p < 0.05`` as a
    text token instead of the ``<`` being mistaken for the start of a tag and
    swallowing the rest of the sentence.
    """
    return [html.unescape(token) for token in re.split("(<.*?>)", line)]


class Cues:
    def __init__(self, data):
        self.sentences = data[0]
        self.negation_cues = data[1]
        self.speculation_cues = data[2]
        self.num_sentences = len(data[0])
class Scopes:
    def __init__(self, data):
        self.negation_sentences = data[0][0]
        self.speculation_sentences = data[1][0]
        self.negation_cues = data[0][1]
        self.speculation_cues = data[1][1]
        self.negation_scopes = data[0][2]
        self.speculation_scopes = data[1][2]
        self.num_sentences = len(data[0])


class Data:
    def __init__(self, file, dataset_name = 'sfu', error_analysis = False):
        '''
        file: The path of the data file.
        dataset_name: The name of the dataset to be preprocessed. Values supported: sfu, bioscope, starsem.
        frac_no_cue_sents: The fraction of sentences to be included in the data object which have no negation/speculation cues.
        '''
            
        def bioscope(f_path, cue_sents_only=False):
            file = open(f_path, encoding = 'utf-8')
            sentences = []
            for s in file:
                sentences += split_tags(s)
            cue_sentence = []
            cue_only_data = []
            negation_cue_cues = []
            speculation_cue_cues = []
            negation_scope_cues = []
            speculation_scope_cues = []
            negation_scope_scopes = []
            speculation_scope_scopes = []
            negation_scope_sentence = []
            speculation_scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            s_idx = []
            cue_id_to_task = {}
            in_sentence = 0
            for token in sentences:
                if token == '':
                    continue
                elif '<sentence' in token:
                    in_sentence = 1
                elif '<cue' in token:
                    in_cue.append(str(re.split('(ref=".*?")',token)[1][4:]))
                    c_idx.append(str(re.split('(ref=".*?")',token)[1][4:]))
                    if c_idx[-1] not in cue.keys():
                        cue[c_idx[-1]] = []
                    if 'speculation' in token:
                        cue_id_to_task[in_cue[-1]] = 'speculation'
                    else:
                        cue_id_to_task[in_cue[-1]] = 'negation'
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    #print(re.split('(id=".*?")',token)[1][3:])
                    in_scope.append(str(re.split('(id=".*?")',token)[1][3:]))
                    s_idx.append(str(re.split('(id=".*?")',token)[1][3:]))
                    scope[s_idx[-1]] = []
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '</sentence' in token:
                    if len(cue.keys())==0:
                        cue_only_data.append([sentence, [3]*len(sentence), [3]*len(sentence)]) # Sentence, Negation Cues, Speculation Cues
                    else:
                        cue_sentence.append(sentence)
                        negation_cue_cues.append([3]*len(sentence))
                        speculation_cue_cues.append([3]*len(sentence))
                        for i in cue.keys():
                            if cue_id_to_task[i] == 'negation':
                                negation_scope_sentence.append(sentence)
                                negation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    negation_cue_cues[-1][cue[i][0]] = 1
                                    negation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        negation_cue_cues[-1][c] = 2
                                        negation_scope_cues[-1][c] = 2
                                negation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        negation_scope_scopes[-1][s] = 1
                            else:
                                speculation_scope_sentence.append(sentence)
                                speculation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    speculation_cue_cues[-1][cue[i][0]] = 1
                                    speculation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        speculation_cue_cues[-1][c] = 2
                                        speculation_scope_cues[-1][c] = 2
                                speculation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        speculation_scope_scopes[-1][s] = 1
                    
                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_sentence = 0
                    c_idx = []
                    s_idx = []
                    cue_id_to_task = {}
                elif '<' not in token:
                    if in_sentence==1:
                        words = token.split()
                        sentence+=words
                        if len(in_cue)!=0:
                            for i in in_cue:
                                cue[i]+=[word_num+i for i in range(len(words))]
                        elif len(in_scope)!=0:
                            for i in in_scope:
                                scope[i]+=[word_num+i for i in range(len(words))]
                        word_num+=len(words)
            
            cue_only_sents = [i[0] for i in cue_only_data]
            negation_cue_only_cues = [i[1] for i in cue_only_data]
            speculation_cue_only_cues = [i[2] for i in cue_only_data]
            cue_train_data = (cue_sentence+cue_only_sents, negation_cue_cues+negation_cue_only_cues, speculation_cue_cues+speculation_cue_only_cues)
            scope_train_data = ([negation_scope_sentence, negation_scope_cues, negation_scope_scopes], [speculation_scope_sentence, speculation_scope_cues, speculation_scope_scopes])
            return [cue_train_data, scope_train_data]
        
        def sfu_review(f_path, cue_sents_only=False, frac_no_cue_sents = 1.0):
            file = open(f_path, encoding = 'utf-8')
            sentences = []
            for s in file:
                sentences += split_tags(s)
            cue_sentence = []
            negation_cue_cues = []
            speculation_cue_cues = []
            negation_scope_cues = []
            speculation_scope_cues = []
            negation_scope_scopes = []
            speculation_scope_scopes = []
            negation_scope_sentence = []
            speculation_scope_sentence = []
            sentence = []
            cue = {}
            scope = {}
            in_scope = []
            in_cue = []
            word_num = 0
            c_idx = []
            cue_only_data = []
            s_idx = []
            in_word = 0
            cue_id_to_task = {}
            for token in sentences:
                if token == '':
                    continue
                elif token == '<W>':
                    in_word = 1
                elif token == '</W>':
                    in_word = 0
                    word_num += 1
                elif '<cue' in token:
                    in_cue.append(int(re.split('(ID=".*?")',token)[1][4:-1]))
                    c_idx.append(in_cue[-1])
                    if c_idx[-1] not in cue.keys():
                        cue[c_idx[-1]] = []
                    if 'speculation' in token:
                        cue_id_to_task[in_cue[-1]] = 'speculation'
                    else:
                        cue_id_to_task[in_cue[-1]] = 'negation'
                elif '</cue' in token:
                    in_cue = in_cue[:-1]
                elif '<xcope' in token:
                    continue
                elif '</xcope' in token:
                    in_scope = in_scope[:-1]
                elif '<ref' in token:
                    in_scope.append([int(i) for i in re.split('(SRC=".*?")',token)[1][5:-1].split(' ')])
                    s_idx.append([int(i) for i in re.split('(SRC=".*?")',token)[1][5:-1].split(' ')])
                    for i in s_idx[-1]:
                        scope[i] = []
                elif '</SENTENCE' in token:
                    if len(cue.keys())==0:
                        cue_only_data.append([sentence, [3]*len(sentence), [3]*len(sentence)]) # Sentence, Negation Cues, Speculation Cues
                    else:
                        cue_sentence.append(sentence)
                        negation_cue_cues.append([3]*len(sentence))
                        speculation_cue_cues.append([3]*len(sentence))
                        for i in cue.keys():
                            if cue_id_to_task[i] == 'negation':
                                negation_scope_sentence.append(sentence)
                                negation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    negation_cue_cues[-1][cue[i][0]] = 1
                                    negation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        negation_cue_cues[-1][c] = 2
                                        negation_scope_cues[-1][c] = 2
                                negation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        negation_scope_scopes[-1][s] = 1
                            else:
                                speculation_scope_sentence.append(sentence)
                                speculation_scope_cues.append([3]*len(sentence))
                                if len(cue[i])==1:
                                    speculation_cue_cues[-1][cue[i][0]] = 1
                                    speculation_scope_cues[-1][cue[i][0]] = 1
                                else:
                                    for c in cue[i]:
                                        speculation_cue_cues[-1][c] = 2
                                        speculation_scope_cues[-1][c] = 2
                                speculation_scope_scopes.append([0]*len(sentence))
                                if i in scope.keys():
                                    for s in scope[i]:
                                        speculation_scope_scopes[-1][s] = 1
                    sentence = []
                    cue = {}
                    scope = {}
                    in_scope = []
                    in_cue = []
                    word_num = 0
                    in_word = 0
                    c_idx = []
                    s_idx = []
                    cue_id_to_task = {}
                elif '<' not in token:
                    if in_word == 1:
                        if len(in_cue)!=0:
                            for i in in_cue:
                                cue[i].append(word_num)
                        if len(in_scope)!=0:
                            for i in in_scope:
                                for j in i:
                                    scope[j].append(word_num)
                        sentence.append(token)
            cue_only_sents = [i[0] for i in cue_only_data]
            negation_cue_only_cues = [i[1] for i in cue_only_data]
            speculation_cue_only_cues = [i[2] for i in cue_only_data]
            cue_train_data = (cue_sentence+cue_only_sents, negation_cue_cues+negation_cue_only_cues, speculation_cue_cues+speculation_cue_only_cues)
            scope_train_data = ([negation_scope_sentence, negation_scope_cues, negation_scope_scopes], [speculation_scope_sentence, speculation_scope_cues, speculation_scope_scopes])
            return [cue_train_data, scope_train_data]
        
        if dataset_name == 'bioscope':
            ret_val = bioscope(file)
            cue_data_to_proc = ret_val[0]
            scope_data_to_proc = ret_val[1]
        elif dataset_name == 'sfu':
            sfu_cues = [[], [], []]
            sfu_scopes = [[[], [], []], [[], [], []]]
            for dir_name in os.listdir(file):
                if '.' not in dir_name:
                    for f_name in os.listdir(file+"//"+dir_name):
                        r_val = sfu_review(file+"//"+dir_name+'//'+f_name)
                        sfu_cues = [a+b for a,b in zip(sfu_cues, r_val[0])]
                        sfu_scopes = [[a+b for a,b in zip(i,j)] for i,j in zip(sfu_scopes, r_val[1])]
                        
            cue_data_to_proc = sfu_cues
            scope_data_to_proc = sfu_scopes
        else:
            raise ValueError("Supported Dataset types are:\n\tbioscope\n\tsfu")
        if error_analysis == True:
            neg_punct, neg_no_punct = [[],[],[]], [[],[],[]]
            for sentence, scope_c, scope in zip(scope_data_to_proc[0][0], scope_data_to_proc[0][1], scope_data_to_proc[0][2]):
                c_ids = [idx for idx, x in enumerate(scope_c) if x != 3]
                min_c_id = min(c_ids)
                max_c_id = max(c_ids)
                scope_a = scope.copy()
                for c in c_ids:
                    scope_a[c] = 1
                punct_ids = set([idx for idx, x in enumerate(sentence) for sym in string.punctuation if sym in x])
                if len(punct_ids) == 0:
                    neg_no_punct[0].append(sentence)
                    neg_no_punct[1].append(scope_c)
                    neg_no_punct[2].append(scope)
                    continue
                min_p_id = [idx for idx in punct_ids if idx < min_c_id]
                if len(min_p_id) == 0:
                    min_p_id = -1
                else:
                    min_p_id = max(min_p_id)
                max_p_id = [idx for idx in punct_ids if idx > max_c_id]
                if len(max_p_id) == 0:
                    max_p_id = -1
                else:
                    max_p_id = min(max_p_id)
                s_ids = [idx for idx, s in enumerate(scope_a) if s==1]
                last_scope_id = max(s_ids)
                first_scope_id = min(s_ids)
                if (last_scope_id+1 == max_p_id or last_scope_id == max_p_id) or (first_scope_id-1 == min_p_id or first_scope_id == min_p_id): # or (last_scope_id in punct_ids)
                    neg_punct[0].append(sentence)
                    neg_punct[1].append(scope_c)
                    neg_punct[2].append(scope)
                else:
                    neg_no_punct[0].append(sentence)
                    neg_no_punct[1].append(scope_c)
                    neg_no_punct[2].append(scope)
            spec_punct, spec_no_punct = [[],[],[]], [[],[],[]]
            for sentence, scope_c, scope in zip(scope_data_to_proc[1][0], scope_data_to_proc[1][1], scope_data_to_proc[1][2]):
                c_ids = [idx for idx, x in enumerate(scope_c) if x != 3]
                min_c_id = min(c_ids)
                max_c_id = max(c_ids)
                scope_a = scope.copy()
                for c in c_ids:
                    scope_a[c] = 1
                punct_ids = set([idx for idx, x in enumerate(sentence) for sym in string.punctuation if sym in x])
                if len(punct_ids) == 0:
                    spec_no_punct[0].append(sentence)
                    spec_no_punct[1].append(scope_c)
                    spec_no_punct[2].append(scope)
                    continue
                min_p_id = [idx for idx in punct_ids if idx < min_c_id]
                if len(min_p_id) == 0:
                    min_p_id = -1
                else:
                    min_p_id = max(min_p_id)
                max_p_id = [idx for idx in punct_ids if idx > max_c_id]
                if len(max_p_id) == 0:
                    max_p_id = -1
                else:
                    max_p_id = min(max_p_id)
                s_ids = [idx for idx, s in enumerate(scope_a) if s==1]
                last_scope_id = max(s_ids)
                first_scope_id = min(s_ids)
                if (last_scope_id+1 == max_p_id or last_scope_id == max_p_id) or (first_scope_id-1 == min_p_id or first_scope_id == min_p_id): # or (last_scope_id in punct_ids)
                    spec_punct[0].append(sentence)
                    spec_punct[1].append(scope_c)
                    spec_punct[2].append(scope)
                else:
                    spec_no_punct[0].append(sentence)
                    spec_no_punct[1].append(scope_c)
                    spec_no_punct[2].append(scope)

            self.scope_data_punct = Scopes([neg_punct, spec_punct])
            self.scope_data_no_punct = Scopes([neg_no_punct, spec_no_punct])
        else:
            self.scope_data_punct =None
            self.scope_data_no_punct = None
        self.cue_data = Cues(cue_data_to_proc)
        self.scope_data = Scopes(scope_data_to_proc)
    
    def get_cue_dataloader(self, val_size = 0.15, test_size = 0.15, other_datasets = []):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''
        do_lower_case = True
        if 'uncased' not in CUE_MODEL:
            do_lower_case = False
        tokenizer = AutoTokenizer.from_pretrained(CUE_MODEL, do_lower_case=do_lower_case, use_fast=False)
        def preprocess_data(obj, tokenizer):
            dl_sents = obj.cue_data.sentences
            dl_negation_cues = obj.cue_data.negation_cues
            dl_speculation_cues = obj.cue_data.speculation_cues
                
            sentences = [" ".join(sent) for sent in dl_sents]

            mytexts = []
            myneglabels = []
            myspeclabels = []
            mymasks = []
            if do_lower_case == True:
                sentences_clean = [sent.lower() for sent in sentences]
            else:
                sentences_clean = sentences
            for sent, neg_tags, spec_tags in zip(sentences_clean, dl_negation_cues, dl_speculation_cues):
                new_neg_tags = []
                new_spec_tags = []
                new_text = []
                new_masks = []
                for word, neg_tag, spec_tag in zip(sent.split(),neg_tags,spec_tags):
                    #print('splitting: ', word)
                    sub_words = tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_neg_tags.append(neg_tag)
                        new_spec_tags.append(spec_tag)
                        new_text.append(sub_word)
                mymasks.append(new_masks)
                mytexts.append(new_text)
                myneglabels.append(new_neg_tags)
                myspeclabels.append(new_spec_tags)
            
            input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in mytexts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            neg_tags = pad_sequences(myneglabels,
                                maxlen=MAX_LEN, value=4, padding="post",
                                dtype="long", truncating="post").tolist()
            
            spec_tags = pad_sequences(myspeclabels,
                                maxlen=MAX_LEN, value=4, padding="post",
                                dtype="long", truncating="post").tolist()
            
            mymasks = pad_sequences(mymasks, maxlen=MAX_LEN, value=0, padding='post', dtype='long', truncating='post').tolist()
            
            attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
            
            random_state = np.random.randint(1,2020)

            tra_inputs, test_inputs, tra_neg_tags, test_neg_tags = train_test_split(input_ids, neg_tags, test_size=test_size, random_state = random_state)
            _, _, tra_spec_tags, test_spec_tags = train_test_split(input_ids, spec_tags, test_size=test_size, random_state = random_state)
            tra_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=test_size, random_state = random_state)
            tra_mymasks, test_mymasks, _, _ = train_test_split(mymasks, input_ids, test_size=test_size, random_state = random_state)

            random_state_2 = np.random.randint(1,2020)

            tr_inputs, val_inputs, tr_neg_tags, val_neg_tags = train_test_split(tra_inputs, tra_neg_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            _, _, tr_spec_tags, val_spec_tags = train_test_split(tra_inputs, tra_spec_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_masks, val_masks, _, _ = train_test_split(tra_masks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            tr_mymasks, val_mymasks, _, _ = train_test_split(tra_mymasks, tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
            
            return [tr_inputs, tr_neg_tags, tr_spec_tags, tr_masks, tr_mymasks], [val_inputs, val_neg_tags, val_spec_tags, val_masks, val_mymasks], [test_inputs, test_neg_tags, test_spec_tags, test_masks, test_mymasks]

        tr_inputs = []
        tr_neg_tags = []
        tr_spec_tags = []
        tr_masks = []
        tr_mymasks = []
        val_inputs = [[] for i in range(len(other_datasets)+1)]
        test_inputs = [[] for i in range(len(other_datasets)+1)]

        train_ret_val, val_ret_val, test_ret_val = preprocess_data(self, tokenizer)
        tr_inputs+=train_ret_val[0]
        tr_neg_tags+=train_ret_val[1]
        tr_spec_tags+=train_ret_val[2]
        tr_masks+=train_ret_val[3]
        tr_mymasks+=train_ret_val[4]
        val_inputs[0].append(val_ret_val[0])
        val_inputs[0].append(val_ret_val[1])
        val_inputs[0].append(val_ret_val[2])
        val_inputs[0].append(val_ret_val[3])
        val_inputs[0].append(val_ret_val[4])
        test_inputs[0].append(test_ret_val[0])
        test_inputs[0].append(test_ret_val[1])
        test_inputs[0].append(test_ret_val[2])
        test_inputs[0].append(test_ret_val[3])
        test_inputs[0].append(test_ret_val[4])
        
        for idx, arg in enumerate(other_datasets, 1):
            train_ret_val, val_ret_val, test_ret_val = preprocess_data(arg, tokenizer)
            tr_inputs+=train_ret_val[0]
            tr_neg_tags+=train_ret_val[1]
            tr_spec_tags+=train_ret_val[2]
            tr_masks+=train_ret_val[3]
            tr_mymasks+=train_ret_val[4]
            val_inputs[idx].append(val_ret_val[0])
            val_inputs[idx].append(val_ret_val[1])
            val_inputs[idx].append(val_ret_val[2])
            val_inputs[idx].append(val_ret_val[3])
            val_inputs[idx].append(val_ret_val[4])
            test_inputs[idx].append(test_ret_val[0])
            test_inputs[idx].append(test_ret_val[1])
            test_inputs[idx].append(test_ret_val[2])
            test_inputs[idx].append(test_ret_val[3])
            test_inputs[idx].append(test_ret_val[4])
        
        tr_inputs = torch.LongTensor(tr_inputs)
        tr_neg_tags = torch.LongTensor(tr_neg_tags)
        tr_spec_tags = torch.LongTensor(tr_spec_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        val_inputs = [[torch.LongTensor(i) for i in j] for j in val_inputs]
        test_inputs = [[torch.LongTensor(i) for i in j] for j in test_inputs]

        train_data = TensorDataset(tr_inputs, tr_masks, tr_neg_tags, tr_spec_tags, tr_mymasks)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        val_dataloaders = []
        for i,j,k,l,m in val_inputs:
            val_data = TensorDataset(i, l, j, k, m)
            val_sampler = RandomSampler(val_data)
            val_dataloaders.append(DataLoader(val_data, sampler=val_sampler, batch_size=bs))

        test_dataloaders = []
        for i,j,k,l,m in test_inputs:
            test_data = TensorDataset(i, l, j, k, m)
            test_sampler = RandomSampler(test_data)
            test_dataloaders.append(DataLoader(test_data, sampler=test_sampler, batch_size=bs))

        return train_dataloader, val_dataloaders, test_dataloaders

    def get_scope_dataloader(self, val_size = 0.15, test_size=0.15, other_datasets = [], error_analysis = False, punct_dl = False):
        '''
        This function returns the dataloader for the cue detection.
        val_size: The size of the validation dataset (Fraction between 0 to 1)
        test_size: The size of the test dataset (Fraction between 0 to 1)
        other_datasets: Other datasets to use to get one combined train dataloader
        Returns: train_dataloader, list of validation dataloaders, list of test dataloaders
        '''

        do_lower_case = True
        if 'uncased' not in SCOPE_MODEL:
            do_lower_case = False
        tokenizer = AutoTokenizer.from_pretrained(SCOPE_MODEL, do_lower_case=do_lower_case, use_fast=False)
        def preprocess_data(obj, tokenizer_obj):
            if error_analysis == False:
                dl_neg_sents = obj.scope_data.negation_sentences
                dl_neg_cues = obj.scope_data.negation_cues
                dl_neg_scopes = obj.scope_data.negation_scopes
                dl_spec_sents = obj.scope_data.speculation_sentences
                dl_spec_cues = obj.scope_data.speculation_cues
                dl_spec_scopes = obj.scope_data.speculation_scopes
            else:
                if punct_dl == False:
                    dl_neg_sents = obj.scope_data_no_punct.negation_sentences
                    dl_neg_cues = obj.scope_data_no_punct.negation_cues
                    dl_neg_scopes = obj.scope_data_no_punct.negation_scopes
                    dl_spec_sents = obj.scope_data_no_punct.speculation_sentences
                    dl_spec_cues = obj.scope_data_no_punct.speculation_cues
                    dl_spec_scopes = obj.scope_data_no_punct.speculation_scopes
                else:
                    dl_neg_sents = obj.scope_data_punct.negation_sentences
                    dl_neg_cues = obj.scope_data_punct.negation_cues
                    dl_neg_scopes = obj.scope_data_punct.negation_scopes
                    dl_spec_sents = obj.scope_data_punct.speculation_sentences
                    dl_spec_cues = obj.scope_data_punct.speculation_cues
                    dl_spec_scopes = obj.scope_data_punct.speculation_scopes
            if SCOPE_METHOD == 'global':
                neg_sentences = [" ".join([s for s in sent+[' [SEP] Negation']]) for sent in dl_neg_sents]
                dl_neg_scopes = [scope_sent+[0,0] for scope_sent in dl_neg_scopes]
                dl_neg_cues = [cue_sent+[3,3] for cue_sent in dl_neg_cues]
                spec_sentences = [" ".join([s for s in sent+[' [SEP] Speculation']]) for sent in dl_spec_sents]
                dl_spec_scopes = [scope_sent+[0,0] for scope_sent in dl_spec_scopes]
                dl_spec_cues = [cue_sent+[3,3] for cue_sent in dl_spec_cues]
            else:
                neg_sentences = [" ".join([s for s in sent]) for sent in dl_neg_sents]
                spec_sentences = [" ".join([s for s in sent]) for sent in dl_spec_sents]
            
            neg_mytexts = []
            neg_mylabels = []
            neg_mycues = []
            neg_mymasks = []
            spec_mytexts = []
            spec_mylabels = []
            spec_mycues = []
            spec_mymasks = []

            if do_lower_case == True:
                neg_sentences_clean = [sent.lower() for sent in neg_sentences]
                spec_sentences_clean = [sent.lower() for sent in spec_sentences]
            else:
                neg_sentences_clean = neg_sentences
                spec_sentences_clean = spec_sentences
            
            for sent, tags, cues in zip(neg_sentences_clean, dl_neg_scopes, dl_neg_cues):
                new_tags = []
                new_text = []
                new_cues = []
                new_masks = []
                for word, tag, cue in zip(sent.split(),tags,cues):
                    sub_words = tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_cues.append(cue)
                        new_text.append(sub_word)
                neg_mymasks.append(new_masks)
                neg_mytexts.append(new_text)
                neg_mylabels.append(new_tags)
                neg_mycues.append(new_cues)

            for sent, tags, cues in zip(spec_sentences_clean, dl_spec_scopes, dl_spec_cues):
                new_tags = []
                new_text = []
                new_cues = []
                new_masks = []
                for word, tag, cue in zip(sent.split(),tags,cues):
                    sub_words = tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        mask = 1
                        if count > 0:
                            mask = 0
                        new_masks.append(mask)
                        new_tags.append(tag)
                        new_cues.append(cue)
                        new_text.append(sub_word)
                spec_mymasks.append(new_masks)
                spec_mytexts.append(new_text)
                spec_mylabels.append(new_tags)
                spec_mycues.append(new_cues)

            final_negation_sentences = []
            final_negation_labels = []
            final_negation_masks = []
            final_speculation_sentences = []
            final_speculation_labels = []
            final_speculation_masks = []
            
            if SCOPE_METHOD == 'global':
                for sent,cues,labels,masks in zip(neg_mytexts, neg_mycues, neg_mylabels, neg_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(mask)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_negation_sentences.append(temp_sent)
                    final_negation_labels.append(temp_label)
                    final_negation_masks.append(temp_masks)

                for sent,cues,labels,masks in zip(spec_mytexts, spec_mycues, spec_mylabels, spec_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(mask)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(mask)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_speculation_sentences.append(temp_sent)
                    final_speculation_labels.append(temp_label)
                    final_speculation_masks.append(temp_masks)

            elif SCOPE_METHOD == 'local':

                for sent,cues,labels,masks in zip(neg_mytexts, neg_mycues, neg_mylabels, neg_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+1}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(0)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_negation_sentences.append(temp_sent)
                    final_negation_labels.append(temp_label)
                    final_negation_masks.append(temp_masks)

                for sent,cues,labels,masks in zip(spec_mytexts, spec_mycues, spec_mylabels, spec_mymasks):
                    temp_sent = []
                    temp_label = []
                    temp_masks = []
                    first_part = 0
                    for token,cue,label,mask in zip(sent,cues,labels,masks):
                        if cue!=3:
                            if first_part == 0:
                                first_part = 1
                                temp_sent.append(f'[unused{cue+6}]')
                                temp_masks.append(1)
                                temp_label.append(label)
                                temp_sent.append(token)
                                temp_masks.append(0)
                                temp_label.append(label)
                                continue
                            temp_sent.append(f'[unused{cue+1}]')
                            temp_masks.append(0)
                            temp_label.append(label)
                        else:
                            first_part = 0
                        temp_masks.append(mask)
                        temp_sent.append(token)
                        temp_label.append(label)
                    final_speculation_sentences.append(temp_sent)
                    final_speculation_labels.append(temp_label)
                    final_speculation_masks.append(temp_masks)
    
            else:
                raise ValueError("Supported methods for scope detection are:\nrglobal\nlocal")

            neg_input_ids = pad_sequences([[tokenizer_obj.convert_tokens_to_ids(word) for word in txt] for txt in final_negation_sentences],
                                      maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            spec_input_ids = pad_sequences([[tokenizer_obj.convert_tokens_to_ids(word) for word in txt] for txt in final_speculation_sentences],
                                      maxlen=MAX_LEN, dtype="long", truncating="post", padding="post").tolist()

            neg_tags = pad_sequences(final_negation_labels,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            spec_tags = pad_sequences(final_speculation_labels,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()
            
            neg_final_masks = pad_sequences(final_negation_masks,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            spec_final_masks = pad_sequences(final_speculation_masks,
                                maxlen=MAX_LEN, value=0, padding="post",
                                dtype="long", truncating="post").tolist()

            neg_attention_masks = [[float(i>0) for i in ii] for ii in neg_input_ids]

            spec_attention_masks = [[float(i>0) for i in ii] for ii in spec_input_ids]

            if test_size > 0.99:
                neg_tr_inputs, neg_tr_tags, neg_tr_masks, neg_tr_mymasks = [], [], [], []
                neg_val_inputs, neg_val_tags, neg_val_masks, neg_val_mymasks = [], [], [], []
                neg_test_inputs, neg_test_tags, neg_test_masks, neg_test_mymasks = neg_input_ids, neg_tags, neg_attention_masks, neg_final_masks
                spec_tr_inputs, spec_tr_tags, spec_tr_masks, spec_tr_mymasks = [], [], [], []
                spec_val_inputs, spec_val_tags, spec_val_masks, spec_val_mymasks = [], [], [], []
                spec_test_inputs, spec_test_tags, spec_test_masks, spec_test_mymasks = spec_input_ids, spec_tags, spec_attention_masks, spec_final_masks
            
            else:
                random_state = np.random.randint(1,2020)

                neg_tra_inputs, neg_test_inputs, neg_tra_tags, neg_test_tags = train_test_split(neg_input_ids, neg_tags, test_size=test_size, random_state = random_state)
                neg_tra_masks, neg_test_masks, _, _ = train_test_split(neg_attention_masks, neg_input_ids, test_size=test_size, random_state = random_state)
                neg_tra_mymasks, neg_test_mymasks, _, _ = train_test_split(neg_final_masks, neg_input_ids, test_size=test_size, random_state = random_state)
                
                spec_tra_inputs, spec_test_inputs, spec_tra_tags, spec_test_tags = train_test_split(spec_input_ids, spec_tags, test_size=test_size, random_state = random_state)
                spec_tra_masks, spec_test_masks, _, _ = train_test_split(spec_attention_masks, spec_input_ids, test_size=test_size, random_state = random_state)
                spec_tra_mymasks, spec_test_mymasks, _, _ = train_test_split(spec_final_masks, spec_input_ids, test_size=test_size, random_state = random_state)
                
                random_state_2 = np.random.randint(1,2020)

                neg_tr_inputs, neg_val_inputs, neg_tr_tags, neg_val_tags = train_test_split(neg_tra_inputs, neg_tra_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                neg_tr_masks, neg_val_masks, _, _ = train_test_split(neg_tra_masks, neg_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                neg_tr_mymasks, neg_val_mymasks, _, _ = train_test_split(neg_tra_mymasks, neg_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)

                spec_tr_inputs, spec_val_inputs, spec_tr_tags, spec_val_tags = train_test_split(spec_tra_inputs, spec_tra_tags, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                spec_tr_masks, spec_val_masks, _, _ = train_test_split(spec_tra_masks, spec_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)
                spec_tr_mymasks, spec_val_mymasks, _, _ = train_test_split(spec_tra_mymasks, spec_tra_inputs, test_size=(val_size/(1-test_size)), random_state = random_state_2)

            return (([neg_tr_inputs, neg_tr_tags, neg_tr_masks, neg_tr_mymasks], [neg_val_inputs, neg_val_tags, neg_val_masks, neg_val_mymasks], [neg_test_inputs, neg_test_tags, neg_test_masks, neg_test_mymasks]), ([spec_tr_inputs, spec_tr_tags, spec_tr_masks, spec_tr_mymasks], [spec_val_inputs, spec_val_tags, spec_val_masks, spec_val_mymasks], [spec_test_inputs, spec_test_tags, spec_test_masks, spec_test_mymasks]))

        tr_inputs = []
        tr_tags = []
        tr_masks = []
        tr_mymasks = []
        neg_val_inputs = []
        neg_val_tags = []
        neg_val_masks = []
        neg_val_mymasks = []
        spec_val_inputs = []
        spec_val_tags = []
        spec_val_masks = []
        spec_val_mymasks = []
        neg_test_inputs = [[] for i in range(len(other_datasets)+1)]
        spec_test_inputs = [[] for i in range(len(other_datasets)+1)]

        r_val = preprocess_data(self, tokenizer)
        [neg_train_ret_val, neg_val_ret_val, neg_test_ret_val] = r_val[0]
        [spec_train_ret_val, spec_val_ret_val, spec_test_ret_val] = r_val[1]
        tr_inputs += neg_train_ret_val[0]
        tr_tags += neg_train_ret_val[1]
        tr_masks += neg_train_ret_val[2]
        tr_mymasks += neg_train_ret_val[3]
        tr_inputs += spec_train_ret_val[0]
        tr_tags += spec_train_ret_val[1]
        tr_masks += spec_train_ret_val[2]
        tr_mymasks += spec_train_ret_val[3]

        neg_val_inputs += neg_val_ret_val[0]
        neg_val_tags += neg_val_ret_val[1]
        neg_val_masks += neg_val_ret_val[2]
        neg_val_mymasks += neg_val_ret_val[3]
        spec_val_inputs += spec_val_ret_val[0]
        spec_val_tags += spec_val_ret_val[1]
        spec_val_masks += spec_val_ret_val[2]
        spec_val_mymasks += spec_val_ret_val[3]
        
        neg_test_inputs[0].append(neg_test_ret_val[0])
        neg_test_inputs[0].append(neg_test_ret_val[1])
        neg_test_inputs[0].append(neg_test_ret_val[2])
        neg_test_inputs[0].append(neg_test_ret_val[3])

        spec_test_inputs[0].append(spec_test_ret_val[0])
        spec_test_inputs[0].append(spec_test_ret_val[1])
        spec_test_inputs[0].append(spec_test_ret_val[2])
        spec_test_inputs[0].append(spec_test_ret_val[3])
        
        for idx, arg in enumerate(other_datasets, 1):
            [neg_train_ret_val, neg_val_ret_val, neg_test_ret_val], [spec_train_ret_val, spec_val_ret_val, spec_test_ret_val] = preprocess_data(arg, tokenizer)
            tr_inputs += neg_train_ret_val[0]
            tr_tags += neg_train_ret_val[1]
            tr_masks += neg_train_ret_val[2]
            tr_mymasks += neg_train_ret_val[3]
            tr_inputs += spec_train_ret_val[0]
            tr_tags += spec_train_ret_val[1]
            tr_masks += spec_train_ret_val[2]
            tr_mymasks += spec_train_ret_val[3]

            neg_val_inputs += neg_val_ret_val[0]
            neg_val_tags += neg_val_ret_val[1]
            neg_val_masks += neg_val_ret_val[2]
            neg_val_mymasks += neg_val_ret_val[3]
            spec_val_inputs += spec_val_ret_val[0]
            spec_val_tags += spec_val_ret_val[1]
            spec_val_masks += spec_val_ret_val[2]
            spec_val_mymasks += spec_val_ret_val[3]
            
            neg_test_inputs[idx].append(neg_test_ret_val[0])
            neg_test_inputs[idx].append(neg_test_ret_val[1])
            neg_test_inputs[idx].append(neg_test_ret_val[2])
            neg_test_inputs[idx].append(neg_test_ret_val[3])

            spec_test_inputs[idx].append(spec_test_ret_val[0])
            spec_test_inputs[idx].append(spec_test_ret_val[1])
            spec_test_inputs[idx].append(spec_test_ret_val[2])
            spec_test_inputs[idx].append(spec_test_ret_val[3])

        tr_inputs = torch.LongTensor(tr_inputs)
        tr_tags = torch.LongTensor(tr_tags)
        tr_masks = torch.LongTensor(tr_masks)
        tr_mymasks = torch.LongTensor(tr_mymasks)
        neg_val_inputs = torch.LongTensor(neg_val_inputs)
        neg_val_tags = torch.LongTensor(neg_val_tags)
        neg_val_masks = torch.LongTensor(neg_val_masks)
        neg_val_mymasks = torch.LongTensor(neg_val_mymasks)
        spec_val_inputs = torch.LongTensor(spec_val_inputs)
        spec_val_tags = torch.LongTensor(spec_val_tags)
        spec_val_masks = torch.LongTensor(spec_val_masks)
        spec_val_mymasks = torch.LongTensor(spec_val_mymasks)
        neg_test_inputs = [[torch.LongTensor(i) for i in j] for j in neg_test_inputs]
        spec_test_inputs = [[torch.LongTensor(i) for i in j] for j in spec_test_inputs]

        if test_size < 0.99:
            train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_mymasks)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

            neg_val_data = TensorDataset(neg_val_inputs, neg_val_masks, neg_val_tags, neg_val_mymasks)
            neg_val_sampler = RandomSampler(neg_val_data)
            neg_val_dataloader = DataLoader(neg_val_data, sampler=neg_val_sampler, batch_size=bs)

            spec_val_data = TensorDataset(spec_val_inputs, spec_val_masks, spec_val_tags, spec_val_mymasks)
            spec_val_sampler = RandomSampler(spec_val_data)
            spec_val_dataloader = DataLoader(spec_val_data, sampler=spec_val_sampler, batch_size=bs)
        
        else:
            train_data = []
            train_sampler = []
            train_dataloader = []

            neg_val_data = []
            neg_val_sampler = []
            neg_val_dataloader = []

            spec_val_data = []
            spec_val_sampler = []
            spec_val_dataloader = []

        neg_test_dataloaders = []
        for i,j,k,l in neg_test_inputs:
            neg_test_data = TensorDataset(i, k, j, l)
            neg_test_sampler = RandomSampler(neg_test_data)
            neg_test_dataloaders.append(DataLoader(neg_test_data, sampler=neg_test_sampler, batch_size=bs))

        spec_test_dataloaders = []
        for i,j,k,l in spec_test_inputs:
            spec_test_data = TensorDataset(i, k, j, l)
            spec_test_sampler = RandomSampler(spec_test_data)
            spec_test_dataloaders.append(DataLoader(spec_test_data, sampler=spec_test_sampler, batch_size=bs))

        return train_dataloader, [neg_val_dataloader, spec_val_dataloader], [neg_test_dataloaders, spec_test_dataloaders]
