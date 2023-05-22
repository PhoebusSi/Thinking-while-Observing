#!user/bin/env python
# -*- coding:utf-8 -*-
import collections
import pickle
from model_LXM2T5 import T5tokenizer, LXMT52T5, LXMtokenizer

from torch.utils.data import Dataset
import json
import pickle
import numpy as np
import torch
import string


from config4LXMT5_DDP import args
print('dataset_val4T5',args)
from random import sample


def normalize_wiki(s):
    stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    # def remove_articles(text):
    #     return regex.sub(r'\b(a|an|the)\b', ' ', text)

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
    with open('../data/validate/okvqa_val.json','r') as f:
        val_row = json.load(f)
    with open('../data/image_features/vqa_img_feature_val.pickle', 'rb') as f:
        pretrain_feature = pickle.load(f)
    with open('../data/validate/caption_predict_val.json', 'r') as f:
        captions_val = json.load(f)
    with open('../data/validate/labeling_predict_val.json', 'r') as f:
        labelings_val = json.load(f)
    with open('../data/validate/ocr_predict_val.json', 'r') as f:
        ocrs_val = json.load(f)
    
    if args.ofa=="normal":
        with open('../data/validate/ofa_predictions/OFA_zerorate_predict_val.json', 'r') as f:
            ofas_val = json.load(f)
        with open('../data/validate/ofa_predictions/OFA_zerorate_evidence_val.json', 'r') as f:
            evid_val = json.load(f)
    elif args.ofa=="finetune":
        with open('../data/validate/ofa_predictions/OFAvqa_zerorate_answer_val.json', 'r') as f:
            ofas_val = json.load(f)
        with open('../data/validate/ofa_predictions/OFAvqa_zerorate_evidence_val.json', 'r') as f:
            evid_val = json.load(f)
    else:
        assert 0==1
    with open("../data/validate/gpt3_okvqa_val2014_answers.pkl", 'rb') as f:
        gpt3_val = pickle.load(f)
    with open('../data/validate/wiki_100sim_val.json', 'r') as f:
        wikis_val = json.load(f)
    

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
most_answer = []
neg_answer = []

val_captions = {}
for item in captions_val:
    if item['image_id'] in val_captions.keys():
        print("IMG caption REPEATED!")
        assert 0==1
    val_captions[item['image_id']] = item['caption']

val_labelings = {}
for item in labelings_val:
    if item['image_id'] in val_labelings.keys():
        print("IMG labelings REPEATED!")
        assert 0==1
    val_labelings[str(item['image_id'])] = item['labeling']

val_ocrs = {}
for item in ocrs_val:
    if item['image_id'] in val_ocrs.keys():
        print("IMG ocrs REPEATED!")
        assert 0==1
    val_ocrs[str(item['image_id'])] = item['ocr']
    

val_ofas = {}

if args.ofa=="normal":
    for item in ofas_val:
        if item['question_id'] in val_ofas.keys():
            print("IMG ofas REPEATED!")
            assert 0==1
        val_ofas[str(item['question_id'])] = item['OFA_answer']+", "+evid_val[str(item['question_id'])]
elif  args.ofa=="finetune":
    for k in evid_val.keys():
        val_ofas[k] = ofas_val[k]+", "+evid_val[k]
else:
    assert 0==1



val_gpt3 = {}
for k in gpt3_val.keys():
    qid = k.split("#")[1]
    
    val_gpt3[str(qid)] = ", ".join(gpt3_val[k][0]) #[(ans, evid)]


val_wikis = wikis_val


for qid, item in val_row.items():
    img_id = str(item['image_id'])
    image_ids.append(img_id)
    qids.append(qid)

    question_clean = item['question']  # + answer_sentence
    questions.append(question_clean)
    if args.dataset == 'okvqa' or args.dataset == 'vqav2':
        answers.append(item['multi_answers'])
        if args.dataset == 'okvqa':
            objects.append(item['label'])
    else:
        answers.append(item['answer'])



def _create_gpt3_entry(imgage_ids, q_ids, questions, answer, captions,labelings, ocrs,ofas,gpt3, wikis, final_txt):
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
        'wiki': wikis,
        'final_txt':final_txt}
    return entry


def _create_entry(imgage_ids, q_ids, questions, answer, captions,labelings, ocrs,ofas, wikis, final_txt):
    entry = {
        'img_id': imgage_ids,
        'qid': q_ids,
        'question': questions,
        'answer': answer,
        'caption': captions,
        'labeling':labelings,
        'ocr': ocrs,
        'ofa':ofas,
        'wiki': wikis,
        'final_txt':final_txt}
    return entry

def _load_dataset(val_row):
    entries=[]
    for qid, item in val_row.items():
        qid = str(qid)
        img_id = str(item['image_id'])
        question = item['question']# + answer_sentence

        if args.dataset == 'okvqa':
            answers=item['multi_answers']


        else:
            answers=item['answer']
        caption=val_captions[img_id]
        labeling=val_labelings[img_id]
        ocr_list=val_ocrs[img_id]
        ocr = ", ".join(str(i) for i in ocr_list)
        ofa=val_ofas[qid]
        gpt3=val_gpt3[qid]
        wiki=val_wikis[qid]

        if args.seed > 1000:
            print("seed > 1000 denotes that ablation study on 2 encoders")
            assert args.input_type==0

        if args.gpt3:
            if args.input_type==0:   
                if args.num_wiki > 51:
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

            entries.append(_create_gpt3_entry(img_id, qid, question, answers, caption,labeling, ocr,ofa,gpt3, wiki, final_txt))
            
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
            elif args.input_type==3: #什么知识都不加。知识单独的性能4（不要预训练）:什么知识都不加，只有视觉属性。
                final_txt = question + " [SEP] " + ofa + " [SEP] " + caption + " [SEP] " + labeling + " [SEP] " + ocr
            else:
                print('choose input-type in [1,2,3,4,5]')
                assert 0==1              

            entries.append(_create_entry(img_id, qid, question, answers, caption,labeling, ocr,ofa, wiki, final_txt))
    return entries






class KgDatasetVal(Dataset):
    def __init__(self, val=False, val_test=False):
        self.entries = _load_dataset(val_row)
        self.tokenize()


    def __len__(self):
        return len(self.entries)
    def tokenize(self):
        if args.input_type%2==0 : #当input_type=0或者2的时候，有wiki在，所以句子长度要长
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
            T5_target_seq, T5_target_ids, T5_target_masks = self.tokenizer_func( T5tokenizer, entry['answer'][0], max_length=max_target_length)
            entry['T5_input_seq']=T5_input_seq#torch.from_numpy(np.array(T5_input_seq))
            entry['T5_input_ids']=torch.from_numpy(np.array(T5_input_ids))
            entry['T5_input_masks']=torch.from_numpy(np.array(T5_input_masks))
            entry['LXM_input_seq']=LXM_input_seq#torch.from_numpy(np.array(LXM_input_seq))
            entry['LXM_input_ids']=torch.from_numpy(np.array(LXM_input_ids))
            entry['LXM_input_masks']=torch.from_numpy(np.array(LXM_input_masks))
            entry['T5_target_seq']=T5_target_seq#torch.from_numpy(np.array(T5_target_seq))
            entry['T5_target_ids']=torch.from_numpy(np.array(T5_target_ids))
            entry['T5_target_masks']=torch.from_numpy(np.array(T5_target_masks))

    def tokenizer_func(self, tokenizer, text, max_length=0):
        if max_length==0:
            print('plz set the max length of input sequence!')
            assert 1==2

        out_seq = tokenizer(
            text,
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


        
        if args.gpt3:
            return qid, question, answer, image_feature, spatial_feature, image_caption, image_labeling, image_ocr, ofa, gpt3, wiki, final_txt, T5_input_seq,T5_input_ids,T5_input_masks,LXM_input_ids,LXM_input_masks,LXM_token_type_ids,T5_target_seq,T5_target_ids,T5_target_masks  
        elif not args.gpt3:
            return qid, question, answer, image_feature, spatial_feature, image_caption, image_labeling, image_ocr, ofa, wiki, final_txt, T5_input_seq,T5_input_ids,T5_input_masks,LXM_input_ids,LXM_input_masks,LXM_token_type_ids,T5_target_seq,T5_target_ids,T5_target_masks  


