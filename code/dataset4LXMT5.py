#!user/bin/env python
# -*- coding:utf-8 -*-
import collections
import json


import string

import numpy as np
from model_LXM2T5 import T5tokenizer, LXMT52T5, LXMtokenizer
import pickle
import torch
from torch.utils.data import Dataset

from config4LXMT5_DDP import args
print('dataset4T5',args)
from random import sample


def normalize_wiki(s):
    stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_stop_w(text):
        to_be_removed = set(stopwords)
        text_list = text.split(' ')
        text_list = [item for item in text_list if item not in to_be_removed]
        return ' '.join(text_list)

    return white_space_fix(remove_stop_w(remove_punc(lower(s))))



if args.dataset == 'okvqa':
    with open('../data/image_features/vqa_img_feature_train.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    if args.pretrain:
        with open('../data/pretrain/vqa_train_filter.json','r') as f:
            vqa2 = json.load(f)
        train_row = vqa2
    else:
        with open('../data/finetune/okvqa_train.json','r') as f:
            train_row = json.load(f)

    if args.pretrain:
        with open('../data/pretrain/caption_predict_vqav2train.json', 'r') as f:
            captions_train = json.load(f)
        with open('../data/pretrain/labeling_predict_vqav2train.json', 'r') as f:
            labelings_train = json.load(f)
        with open('../data/pretrain/ocr_predict_vqav2train.json', 'r') as f:
            ocrs_train = json.load(f)

        with open('../data/pretrain/wiki_100sim_train.json', 'r') as f:
            wikis_train = json.load(f)

    else:
        with open('../data/finetune/caption_predict_train.json', 'r') as f:
            captions_train = json.load(f)
        with open('../data/finetune/labeling_predict_train.json', 'r') as f:
            labelings_train = json.load(f)
        with open('../data/finetune/ocr_predict_train.json', 'r') as f:
            ocrs_train = json.load(f)
        if args.ofa=="normal":
            with open('../data/finetune/ofa_predictions/OFA_zerorate_predict_train.json', 'r') as f:
                ofas_train = json.load(f)#key为数字
            with open('../data/finetune/ofa_predictions/OFA_zerorate_evidence_train.json', 'r') as f:
                evid_train = json.load(f)#key为字符串
        elif args.ofa=="finetune":
            with open('../data/finetune/ofa_predictions/OFAvqa_zerorate_answer_train.json', 'r') as f:
                ofas_train = json.load(f)#key为字符串
            with open('../data/finetune/ofa_predictions/OFAvqa_zerorate_evidence_train.json', 'r') as f:
                evid_train = json.load(f)#key为字符串
        else:
            assert 0==1
        with open("../data/finetune/gpt3_okvqa_train2014_answers.pkl", 'rb') as f:
            gpt3_train = pickle.load(f)
        with open('../data/finetune/wiki_100sim_train.json', 'r') as f:
            wikis_train = json.load(f)

else:
    assert 0==1



def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sxo' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

image_ids = []
qids = []
questions = []
answers = []
labels = []
objects = []
answer_ids = []
answers_lists = []
question_lengths = []
answers_most = []
neg_answer = []


train_captions = {}
for item in captions_train:
    if item['image_id'] in train_captions.keys():
        print("IMG caption REPEATED!")
        assert 0==1
    train_captions[item['image_id']] = item['caption']

train_labelings = {}
for item in labelings_train:
    if item['image_id'] in train_labelings.keys():
        print("IMG labelings REPEATED!")
        assert 0==1
    train_labelings[str(item['image_id'])] = item['labeling']
print("labeling number:", len(train_labelings.keys()))

train_ocrs = {}
for item in ocrs_train:
    if item['image_id'] in train_ocrs.keys():
        print("IMG ocrs REPEATED!")
        assert 0==1
    train_ocrs[str(item['image_id'])] = item['ocr']


if not args.pretrain:
    train_ofas = {}
    if args.ofa=="normal":
        for item in ofas_train:
            if item['question_id'] in train_ofas.keys():
                print("IMG ofas REPEATED!")
                assert 0==1
            train_ofas[str(item['question_id'])] = item['OFA_answer']+", "+evid_train[str(item['question_id'])]
    elif  args.ofa=="finetune":
        for k in evid_train.keys():
            train_ofas[k] = ofas_train[k]+", "+evid_train[k]
    else:
        assert 0==1

    train_gpt3 = {}
    for k in gpt3_train.keys():
        qid = k.split("#")[1]
    
        train_gpt3[str(qid)] = ", ".join(gpt3_train[k][0])#[(ans, evid)]


train_wikis = wikis_train


if args.pretrain: 
    if args.num_wiki > 51:
        for key in train_wikis.keys():
            for i in range(args.num_wiki):
                train_wikis[key][i]=normalize_wiki(train_wikis[key][i])




n = 0


for qid, item in train_row.items():
    img_id = str(item['image_id'])
    image_ids.append(img_id)
    qids.append(qid)
    question_clean = item['question']# + answer_sentence
    questions.append(question_clean)
 


    # multi-answer
    if args.dataset == 'okvqa':
        answers.append(item['multi_answers'])
        # m_ans_id = [a_dic.get(i, 0) for i in item['multi_answers']]
        # most_answer_ids.append(m_ans_id)


    #single answer
    else:
        answers.append(item['answer'])




def _create_gpt3_entry(imgage_ids, q_ids, questions, answer, captions,labelings, ocrs,ofas, gpt3, wikis,final_txt):

    if not args.pretrain:
        entry = {
            'img_id': imgage_ids,
            'qid': q_ids,
            'question': questions,
            'answer': answer,
            'caption': captions,
            'labeling':labelings,
            'ocr': ocrs,
            'ofa':ofas,
            'gpt3':gpt3,
            'wiki':wikis,
            'final_txt':final_txt}

    return entry



def _create_entry(imgage_ids, q_ids, questions, answer, captions,labelings, ocrs,ofas, wikis,final_txt):
    if not args.pretrain:
        entry = {
            'img_id': imgage_ids,
            'qid': q_ids,
            'question': questions,
            'answer': answer,
            'caption': captions,
            'labeling':labelings,
            'ocr': ocrs,
            'ofa':ofas,
            'wiki':wikis,
            'final_txt':final_txt}
    return entry


def _create_vqav2_entry(imgage_ids, q_ids, questions, answer, captions,labelings, ocrs,wikis,final_txt):
    if args.pretrain:
        entry = {
            'img_id': imgage_ids,
            'qid': q_ids,
            'question': questions,
            'answer': answer,
            'caption': captions,
            'labeling':labelings,
            'ocr': ocrs,
            'wiki':wikis,
            'final_txt':final_txt}
    # else:
    return entry


def _load_dataset(train_row):
    entries=[]
    for qid, item in train_row.items():
        qid = str(qid)
        img_id = str(item['image_id'])
        question = item['question']
 


        # multi-answer
        if args.dataset == 'okvqa':
            answers=item['multi_answers']
            

        #single answer
        else:
            answers=item['answer']
            
        caption=train_captions[img_id]
        labeling=train_labelings[img_id]
        ocr_list=train_ocrs[img_id]
        ocr = ", ".join(str(i) for i in ocr_list)
        if not args.pretrain:
            ofa=train_ofas[qid]
            gpt3=train_gpt3[qid]
        wiki=train_wikis[qid]
        
        if args.pretrain:
            if args.num_wiki > 51:
                final_txt = [question + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + x for x in wiki[:args.num_wiki]]
            else:
                final_txt = [question + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + x for x in wiki[:args.num_wiki]]
        else:
            if args.seed > 1000:
                print("seed > 1000 denotes that ablation study on 2 encoders")
                assert args.input_type==0
            if args.gpt3:
                if args.input_type==0:
                
                    if args.num_wiki > 51: 
			# When there are a large number of Wiki passages, to save on GPU memory usage, Wiki passages are processed.
                        final_txt = [question + " [SEP] " + ofa + " " + gpt3 + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + normalize_wiki(x) for x in wiki[:args.num_wiki]]
                    else:
                        final_txt = [question + " [SEP] " + ofa + " " + gpt3 + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + x for x in wiki[:args.num_wiki]]
                elif args.input_type==1: 
                    final_txt = question + " [SEP] " + ofa + " " + gpt3 + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr
                elif args.input_type==2:
                    if args.num_wiki > 51:
                        final_txt = [question + " [SEP] "  + gpt3 + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + normalize_wiki(x) for x in wiki[:args.num_wiki]]
                    else:
                        final_txt = [question + " [SEP] "  + gpt3 + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + x for x in wiki[:args.num_wiki]]
                elif args.input_type==3: 
                    final_txt = question + " [SEP] " + gpt3 + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr
                else:
                    print('choose input-type in [0,1,2,3]')
                    assert 0==1
                
                
            else:
                if args.input_type==0:
                
                    if args.num_wiki > 51:
                        final_txt = [question + " [SEP] " + ofa + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + normalize_wiki(x) for x in wiki[:args.num_wiki]]
                    else:
                        final_txt = [question + " [SEP] " + ofa + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + x for x in wiki[:args.num_wiki]]
                elif args.input_type==1: 
                    final_txt = question + " [SEP] " + ofa + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr
                elif args.input_type==2: 
                    if args.num_wiki > 51:
                        final_txt = [question + " [SEP] "  + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + normalize_wiki(x) for x in wiki[:args.num_wiki]]
                    else:
                        final_txt = [question + " [SEP] "  + caption + " [SEP] " + labeling + " [SEP] " + ocr + " [SEP] " + x for x in wiki[:args.num_wiki]]
                elif args.input_type==3:
                    final_txt = question + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr
                else:
                    print('choose input-type in [0,1,2,3,4,5]')
                    assert 0==1
                
                
                
        
        if args.pretrain:
            entries.append(_create_vqav2_entry(img_id, qid, question, answers, caption,labeling, ocr, wiki, final_txt))
        else:
            if args.gpt3:
                entries.append(_create_gpt3_entry(img_id, qid, question, answers, caption,labeling, ocr,ofa,gpt3, wiki, final_txt))
            else:
                entries.append(_create_entry(img_id, qid, question, answers, caption,labeling, ocr,ofa, wiki, final_txt))

    return entries





def _create_pretrain_entry(imgage_ids, q_ids, questions, answer):#, captions,labelings, ocrs,ofas,final_txt):
    entry = {
        'img_id': imgage_ids,
        'qid': q_ids,
        'question': questions,
        'answer': answer}#,
    return entry

def _load_pretrain_dataset(train_row):
    entries=[]
    for qid, item in train_row.items():
        qid = str(qid)

        img_id = str(item['image_id'])
        question = item['question']


        # multi-answer
        if args.dataset == 'okvqa':
            answers=item['multi_answers']
            # answers.append(item['multi_answers'])
            # m_ans_id = [a_dic.get(i, 0) for i in item['multi_answers']]
            # most_answer_ids.append(m_ans_id)


        #single answer
        else:
            answers=item['answer']
        
        entries.append(_create_pretrain_entry(img_id, qid, question, answers))
    return entries



class KgDataset(Dataset):
    def __init__(self, val=False, val_test=False):
        self.entries = _load_dataset(train_row)
        self.tokenize()

    def __len__(self):
        return len(self.entries)
    def tokenize(self):
        if args.input_type==0:
            if args.num_wiki > 51:
                max_source_length=200
            else:
                max_source_length=250 #300
        else:
            max_source_length=128
        max_target_length=5
        max_que_length=16
        for entry in self.entries:
            T5_input_seq, T5_input_ids, T5_input_masks = self.tokenizer_func( T5tokenizer, entry['final_txt'], max_length=max_source_length)
            LXM_input_seq, LXM_input_ids, LXM_input_masks = self.tokenizer_func( LXMtokenizer, entry['question'], max_length=max_que_length)
            
            
            all_Ans_T5_target_seq = []
            all_Ans_T5_target_ids = []
            all_Ans_T5_target_masks = []
            if args.allAns:
                for i in range(10):  
                    if i%2==0:
                        T5_target_seq, T5_target_ids, T5_target_masks = self.tokenizer_func( T5tokenizer, entry['answer'][i], max_length=max_target_length)
                        all_Ans_T5_target_seq.append(T5_target_seq)
                        all_Ans_T5_target_ids.append(torch.from_numpy(np.array(T5_target_ids)))
                        all_Ans_T5_target_masks.append(torch.from_numpy(np.array(T5_target_masks)))
                        # print()
                all_Ans_T5_target_ids=torch.stack(all_Ans_T5_target_ids)
                all_Ans_T5_target_masks=torch.stack(all_Ans_T5_target_masks)
                
                entry['T5_target_seq']=all_Ans_T5_target_seq
                entry['T5_target_ids']=all_Ans_T5_target_ids
                entry['T5_target_masks']=all_Ans_T5_target_masks

            else:
                T5_target_seq, T5_target_ids, T5_target_masks = self.tokenizer_func( T5tokenizer, entry['answer'][0], max_length=max_target_length)
                entry['T5_target_seq']=T5_target_seq#torch.from_numpy(np.array(T5_target_seq))
                entry['T5_target_ids']=torch.from_numpy(np.array(T5_target_ids))
                entry['T5_target_masks']=torch.from_numpy(np.array(T5_target_masks))
            entry['T5_input_seq']=T5_input_seq#torch.from_numpy(np.array(T5_input_seq))
            entry['T5_input_ids']=torch.from_numpy(np.array(T5_input_ids))
            entry['T5_input_masks']=torch.from_numpy(np.array(T5_input_masks))
            entry['LXM_input_seq']=LXM_input_seq#torch.from_numpy(np.array(LXM_input_seq))
            entry['LXM_input_ids']=torch.from_numpy(np.array(LXM_input_ids))
            entry['LXM_input_masks']=torch.from_numpy(np.array(LXM_input_masks))
            


    def tokenizer_func(self, tokenizer, text, max_length=0):
        if max_length==0:
            print('plz set the max length of input sequence!')
            assert 1==2
       
        out_seq = tokenizer(
            text,
            # batch_data['final_txt'],
            padding='max_length',
            max_length=max_length,
            truncation=True,
            # return_tensors="pt",
            )
        
        tokens=out_seq.input_ids #['input_ids']
        masks=out_seq.attention_mask
        length = len(tokens)

        return out_seq, tokens, masks

    def __getitem__(self, index):

        entry = self.entries[index]
        qid=entry['qid']
        question=entry['question']
        answer=entry['answer']
        img_id=entry['img_id']
        image_feature = pretrain_feature[img_id]['feats']
        
        image_caption = entry['caption']
        image_labeling = entry['labeling']
        image_ocr_list = entry['ocr']
        image_ocr = ", ".join(str(i) for i in image_ocr_list)
        if not args.pretrain:
            ofa = entry['ofa'] 
            if args.gpt3:
                gpt3 = entry['gpt3'] 
        wiki = entry['wiki']
        final_txt = entry['final_txt']
        
        
        spatial_feature = pretrain_feature[img_id]['sp_feats']
        
        T5_input_seq, T5_input_ids, T5_input_masks = entry['T5_input_seq'], entry['T5_input_ids'], entry['T5_input_masks']#self.tokenizer_func( T5tokenizer, final_txt, max_length=max_source_length)
        
        LXM_input_seq, LXM_input_ids, LXM_input_masks = entry['LXM_input_seq'], entry['LXM_input_ids'], entry['LXM_input_masks']
        
        LXM_token_type_ids = torch.from_numpy(np.array(LXM_input_seq['token_type_ids']))#.to(device)
        
        T5_target_seq, T5_target_ids, T5_target_masks=entry['T5_target_seq'],entry['T5_target_ids'],entry['T5_target_masks']
        
        if not args.pretrain:
            if not args.gpt3:
                return qid, question, answer, image_feature, spatial_feature, image_caption, image_labeling, image_ocr, ofa, wiki, final_txt, T5_input_seq,T5_input_ids,T5_input_masks,LXM_input_ids,LXM_input_masks,LXM_token_type_ids,T5_target_seq,T5_target_ids,T5_target_masks  
            elif args.gpt3:
                return qid, question, answer, image_feature, spatial_feature, image_caption, image_labeling, image_ocr, ofa, gpt3, wiki, final_txt, T5_input_seq,T5_input_ids,T5_input_masks,LXM_input_ids,LXM_input_masks,LXM_token_type_ids,T5_target_seq,T5_target_ids,T5_target_masks  
        else:
            return qid, question, answer, image_feature, spatial_feature, image_caption, image_labeling, image_ocr, wiki, final_txt, T5_input_seq,T5_input_ids,T5_input_masks,LXM_input_ids,LXM_input_masks,LXM_token_type_ids,T5_target_seq,T5_target_ids,T5_target_masks  

def my_collate(batch):
    batch = list(zip(*batch))
    if not args.pretrain:
        if not args.gpt3:
            res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                    'img': batch[3], 'spatial': batch[4],
                    'caption': batch[5], 'labeling': batch[6], 'ocr': batch[7], 'ofa': batch[8], 'wiki': batch[9], 'final_txt': batch[10],
                    'T5_input_seq': batch[11], 'T5_input_ids': batch[12],'T5_input_masks': batch[13],'LXM_input_ids':batch[14], 'LXM_input_masks':batch[15], 'LXM_token_type_ids':batch[16], 'T5_target_seq':batch[17],'T5_target_ids':batch[18],'T5_target_masks':batch[19]}
        elif args.gpt3:
            res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                    'img': batch[3], 'spatial': batch[4],
                    'caption': batch[5], 'labeling': batch[6], 'ocr': batch[7], 'ofa': batch[8], 'gpt3': batch[9], 'wiki': batch[10], 'final_txt': batch[11],
                    'T5_input_seq': batch[12], 'T5_input_ids': batch[13],'T5_input_masks': batch[14],'LXM_input_ids':batch[15], 'LXM_input_masks':batch[16], 'LXM_token_type_ids':batch[17], 'T5_target_seq':batch[18],'T5_target_ids':batch[19],'T5_target_masks':batch[20]}


    else:
        res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                'img': batch[3], 'spatial': batch[4],
                'caption': batch[5], 'labeling': batch[6], 'ocr': batch[7], 'wiki': batch[8], 'final_txt': batch[9],
                'T5_input_seq': batch[10], 'T5_input_ids': batch[11],'T5_input_masks': batch[12],'LXM_input_ids':batch[13], 'LXM_input_masks':batch[14], 'LXM_token_type_ids':batch[15], 'T5_target_seq':batch[16],'T5_target_ids':batch[17],'T5_target_masks':batch[18]}

            
    del batch
    return res

def my_val_collate(batch):
    batch = list(zip(*batch))
    if 1:
        res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                'img': batch[3], 'spatial': batch[4],
                'caption': batch[5], 'labeling': batch[6], 'ocr': batch[7], 'ofa': batch[8], 'wiki': batch[9], 'final_txt': batch[10],
                'T5_input_seq': batch[11], 'T5_input_ids': batch[12],'T5_input_masks': batch[13],'LXM_input_ids':batch[14], 'LXM_input_masks':batch[15], 'LXM_token_type_ids':batch[16], 'T5_target_seq':batch[17],'T5_target_ids':batch[18],'T5_target_masks':batch[19]}
    del batch
    return res





def my_gpt3_collate(batch):
    batch = list(zip(*batch))
    if 1:
        res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                'img': batch[3], 'spatial': batch[4],
                'caption': batch[5], 'labeling': batch[6], 'ocr': batch[7], 'ofa': batch[8],'gpt3': batch[9], 'wiki': batch[10], 'final_txt': batch[11],
                'T5_input_seq': batch[12], 'T5_input_ids': batch[13],'T5_input_masks': batch[14],'LXM_input_ids':batch[15], 'LXM_input_masks':batch[16], 'LXM_token_type_ids':batch[17], 'T5_target_seq':batch[18],'T5_target_ids':batch[19],'T5_target_masks':batch[20]}
    del batch
    return res

def my_val_gpt3_collate(batch):
    batch = list(zip(*batch))
    if 1:
        res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
                'img': batch[3], 'spatial': batch[4],
                'caption': batch[5], 'labeling': batch[6], 'ocr': batch[7], 'ofa': batch[8],'gpt3': batch[9], 'wiki': batch[10], 'final_txt': batch[11],
                'T5_input_seq': batch[12], 'T5_input_ids': batch[13],'T5_input_masks': batch[14],'LXM_input_ids':batch[15], 'LXM_input_masks':batch[16], 'LXM_token_type_ids':batch[17], 'T5_target_seq':batch[18],'T5_target_ids':batch[19],'T5_target_masks':batch[20]}
    del batch
    return res
