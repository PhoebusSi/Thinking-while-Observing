#!user/bin/env python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import datetime
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bisect import bisect
from math import fabs
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LxmertTokenizer
from dist_train import get_world_size, get_rank, get_local_rank, barrier, reduce_sum

import numpy as np
from transformers.tokenization_utils_base import ENCODE_KWARGS_DOCSTRING
from config4LXMT5_DDP import args

from dataset4LXMT5 import KgDataset, my_collate, my_val_gpt3_collate, my_val_collate 
from dataset_val4LXMT5 import KgDatasetVal

if args.visualBERT:
    from model_ViB2T5 import T5tokenizer, ViBT52T5, LXMtokenizer
else:
    from model_LXM2T5 import T5tokenizer, LXMT52T5, LXMtokenizer

from transformers import get_linear_schedule_with_warmup
from transformers import LxmertConfig, LxmertTokenizer, LxmertModel,BertTokenizer

dist.init_process_group(backend='nccl',timeout=datetime.timedelta(seconds=5400))
torch.cuda.set_device(args.local_rank)


# LR = 1e-5
LR = args.learning_rate
LR_LXM = args.learning_rate_LXM
# LR = 1e-4

torch.multiprocessing.set_sharing_strategy('file_system')

torch.cuda.set_device(get_local_rank())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone().float()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()#.float() 
    return rt

def set_seed(rank):
    random.seed(args.seed+rank) 
    np.random.seed(args.seed+rank)
    torch.manual_seed(args.seed+rank)
    torch.cuda.manual_seed(args.seed+rank)
    torch.cuda.manual_seed_all(args.seed+rank)
    torch.backends.cudnn.deterministic = True

set_seed(get_rank())





def cal_acc_multi(ground_truth, preds, return_id = False):
    all_num = len(ground_truth)
    acc_num = 0
    ids = []
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        # ids.append([i, int(pred)])
        cnt = 0
        for aid in answer_id:
            if pred == aid:
                cnt += 1
        if cnt ==1:
            acc_num += 1/3 
        elif cnt == 2:
            acc_num += 2/3 
        elif cnt > 2:
            acc_num += 1

    if return_id:
        return acc_num / all_num, ids
    else:
        return acc_num,  all_num
   
def cal_acc(ground_truth, preds, return_id = False):
    all_num = len(ground_truth)
    acc_num = 0
    ids = []
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        # ids.append([i, int(pred)])
        cnt = 0
        for aid in answer_id:
            if pred == aid:
                acc_num += 1
    if return_id:
        return acc_num / all_num, ids
    else:
        return acc_num, all_num


def train():
    if not args.describe:
        print('please set the description for the saved-model name! use --describe !')
        assert 1==0
    else:
        model_name=args.describe
    if not args.pretrain:
        train_dataset = KgDataset(val=False)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler,#shuffle=True,
                                      num_workers=0, collate_fn=my_collate)#, pin_memory=True)

        if args.validate:
            test_dataset = KgDatasetVal(val=False)
            if args.gpt3:
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=0, collate_fn=my_val_gpt3_collate)
            elif not args.gpt3:
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=0, collate_fn=my_val_collate)
    else:
        train_dataset = KgDataset(val=False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,#pin_memory=True,
                                      num_workers=0, collate_fn=my_collate,  sampler=train_sampler)
        if args.validate:
            test_dataset = KgDatasetVal(val=False)
            if args.gpt3:
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                            num_workers=0, collate_fn=my_val_gpt3_collate, shuffle=False)#sampler=test_sampler)
            elif not args.gpt3:
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=0, collate_fn=my_val_collate)

    if args.pretrain:
        if get_rank() == 0: 
            print('pre-training!')
        if args.visualBERT:
            model=  ViBT52T5()
        else:
            model = LXMT52T5()
    else:
        if get_rank() == 0: 
            print('fine-tuning!')
        if args.visualBERT:
            model = ViBT52T5()
        else:
            model = LXMT52T5()

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if get_world_size() > 1:
        if get_rank() == 0: 
            print("Let's use", get_world_size(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[get_local_rank()], output_device=get_local_rank(),find_unused_parameters=True)

    print(model.named_modules)
    if get_world_size() > 1:
        if args.visualBERT:
            optimizer = optim.AdamW([
                {'params': model.module.T5model.parameters(), 'lr': LR},
                {'params': model.module.ViBmodel.parameters(), 'lr': LR_LXM},
                {'params': model.module.mapping.parameters(), 'lr': LR_LXM},
            ])
        else:
            optimizer = optim.AdamW([
                {'params': model.module.T5model.parameters(), 'lr': LR},
                {'params': model.module.LXMmodel.parameters(), 'lr': LR_LXM},
                {'params': model.module.mapping.parameters(), 'lr': LR_LXM},
    
            ])
    else:
        if args.visualBERT:
            optimizer = optim.AdamW([
                {'params': model.T5model.parameters(), 'lr': LR},
                {'params': model.ViBmodel.parameters(), 'lr': LR_LXM},
                {'params': model.mapping.parameters(), 'lr': LR_LXM},
            ])
        else:
            optimizer = optim.AdamW([
                {'params': model.T5model.parameters(), 'lr': LR},
                {'params': model.LXMmodel.parameters(), 'lr': LR_LXM},
                {'params': model.mapping.parameters(), 'lr': LR_LXM},
            ])
    
    if args.pretrain:
        steps_num = 100000 # batch_size should be set small 
    else:
        steps_num = 4000 



    args.num_epochs = steps_num // (len(train_dataset) / (args.batch_size * get_world_size())) \
        if len(train_dataset) % args.batch_size == 0 \
        else  (steps_num // (len(train_dataset) / (args.batch_size * get_world_size())) )+1 
    args.num_epochs = int(args.num_epochs)

    if get_rank() == 0: 
        print('total_epoch', args.num_epochs)
        print('total_steps', "we set steps=",steps_num)
        print('warmup_steps', int(steps_num/10)) #0.05*total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps_num/10), #0.01 * total_steps,
                                                num_training_steps=steps_num)
    

    if args.load_pthpath == "":
        start_epoch = 0
    else:
        if get_rank() == 0: 
            print('load model')
        start_epoch = 0

        if get_world_size() > 1:
            model.module.load_state_dict(torch.load(args.load_pthpath))
        else:
            model.load_state_dict(torch.load(args.load_pthpath))


    best_acc_t = 0
    best_epoch_t = 0
    best_acc_t3 = 0
    step_ind = 0

    for epoch in range(start_epoch, args.num_epochs):
        train_preds_trip = []
        train_sampler.set_epoch(epoch)
        train_answers_trip = []
        s=0
        for batch_data in tqdm(train_dataloader):
            step_ind+=1
            if get_rank()==0:
                print("step_ind",step_ind)
            s=s+1
            visual_faetures = torch.from_numpy(np.array(batch_data['img'], dtype=float)).float().to(device)
            spatial_features = torch.tensor(np.array(batch_data['spatial'])).float().to(device)
            if 1:
                T5_input_id = torch.stack(batch_data['T5_input_ids']).to(device)
                T5_input_mask = torch.stack(batch_data['T5_input_masks']).to(device)


            LXM_input_id = torch.stack(batch_data['LXM_input_ids']).to(device)
            LXM_input_mask = torch.stack(batch_data['LXM_input_masks']).to(device)
            LXM_token_type_ids = torch.stack(batch_data['LXM_token_type_ids']).to(device)

            T5_target_id = torch.stack(batch_data['T5_target_ids']).to(device)

            neg100 = torch.ones_like(T5_target_id)*(-100)
            T5_target_id = torch.where(T5_target_id==T5tokenizer.pad_token_id,neg100, T5_target_id)


            

            model.zero_grad()


            optimizer.zero_grad()
            if args.pretrain:
                outputs = model(train=True, LXM_source_ids=LXM_input_id, LXM_source_masks=LXM_input_mask,T5_source_ids=T5_input_id, T5_source_masks=T5_input_mask,token_type_ids=LXM_token_type_ids, visual_features=visual_faetures, spatial_features=spatial_features,T5_target_ids=T5_target_id)#,T5_target_masks=None
            else:
                outputs = model(train=True, LXM_source_ids=LXM_input_id, LXM_source_masks=LXM_input_mask,T5_source_ids=T5_input_id, T5_source_masks=T5_input_mask,token_type_ids=LXM_token_type_ids, visual_features=visual_faetures, spatial_features=spatial_features,T5_target_ids=T5_target_id)#,T5_target_masks=None
            loss = outputs.loss

            loss_stat = torch.mean(loss.detach()).item()

            if get_rank() == 0: 
                print("loss on GPU0", loss_stat)
            loss.sum().backward()
            optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                if args.pretrain:
                    eval_outputs = model(train=False, LXM_source_ids=LXM_input_id, LXM_source_masks=LXM_input_mask,T5_source_ids=T5_input_id, T5_source_masks=T5_input_mask,token_type_ids=LXM_token_type_ids, visual_features=visual_faetures, spatial_features=spatial_features,T5_target_ids=T5_target_id)
                else:
                    eval_outputs = model(train=False, LXM_source_ids=LXM_input_id, LXM_source_masks=LXM_input_mask,T5_source_ids=T5_input_id, T5_source_masks=T5_input_mask,token_type_ids=LXM_token_type_ids, visual_features=visual_faetures, spatial_features=spatial_features,T5_target_ids=T5_target_id)
                trip_predict = T5tokenizer.batch_decode(eval_outputs, skip_special_tokens=True)
                if get_rank() == 0: 
                    print('epoch', epoch, 'step', s, '>>>', '\tans:', batch_data['ans'][0], 'pred:', trip_predict[0])
                for i, pre in enumerate(batch_data['ans']):
                    train_answers_trip.append(batch_data['ans'][i])
                    train_preds_trip.append(trip_predict[i])
                            
            model.train()            
            barrier()
            








        barrier()

        if 1:
            train_acc_1_num, train_total_1_num = cal_acc_multi(train_answers_trip, train_preds_trip)

            train_reduce_acc_num=reduce_tensor(torch.tensor(train_acc_1_num).cuda(args.local_rank)).item()
            train_reduce_total_num=reduce_tensor(torch.tensor(train_total_1_num).cuda(args.local_rank)).item()
            train_acc_1_trip = train_reduce_acc_num/train_reduce_total_num
            if get_rank() == 0:
                print('epoch %d train_loss of GPU0= %.1f, acc_trip on all GPUs= %.4f' % (epoch, loss_stat,
                                                                          train_acc_1_trip))
        if args.validate:
            model.eval()  
            answers = []  # [batch_answers,...]
            preds = []  # [batch_preds,...]
            preds_trip = []
            preds_trip_3 = []
            answers_trip = []
            id2pred_trip = {}
            print(f"\nValidation after epoch {epoch}:")
            for i, batch_data in enumerate(tqdm(test_dataloader)):
                with torch.no_grad():
                    val_T5_input_id = torch.stack(batch_data['T5_input_ids']).to(device)
                    val_T5_input_mask = torch.stack(batch_data['T5_input_masks']).to(device)

                    val_visual_faetures = torch.tensor(np.array(batch_data['img'])).float().to(device)
                    val_spatial_features = torch.tensor(np.array(batch_data['spatial'])).float().to(device)

                    val_LXM_input_id = torch.stack(batch_data['LXM_input_ids']).to(device)
                    val_LXM_input_mask = torch.stack(batch_data['LXM_input_masks']).to(device)
                    val_LXM_token_type_ids = torch.stack(batch_data['LXM_token_type_ids']).to(device)


                    if args.pretrain:
                        val_outputs = model(train=False, LXM_source_ids=val_LXM_input_id, LXM_source_masks=val_LXM_input_mask,T5_source_ids=val_T5_input_id, T5_source_masks=val_T5_input_mask,token_type_ids=val_LXM_token_type_ids, visual_features=val_visual_faetures, spatial_features=val_spatial_features,T5_target_ids=None)
                    else:
                        val_outputs = model(train=False, LXM_source_ids=val_LXM_input_id, LXM_source_masks=val_LXM_input_mask,T5_source_ids=val_T5_input_id, T5_source_masks=val_T5_input_mask,token_type_ids=val_LXM_token_type_ids, visual_features=val_visual_faetures, spatial_features=val_spatial_features,T5_target_ids=None)


                    val_trip_predict = T5tokenizer.batch_decode(val_outputs, skip_special_tokens=True)


                    
                    for i, pre in enumerate(batch_data['ans']):
                        preds_trip.append(val_trip_predict[i])
                        answers_trip.append(batch_data['ans'][i])

                        id2pred_trip[str(batch_data['id'][i])]=val_trip_predict[i]


            if args.dataset == 'krvqa':
                acc_1_num, total_1_num = cal_acc(answers_trip, preds_trip)
                reduce_acc_num=reduce_tensor(torch.tensor(acc_1_num).cuda(args.local_rank)).item()
                reduce_total_num=reduce_tensor(torch.tensor(total_1_num).cuda(args.local_rank)).item()
                acc_1_trip = reduce_acc_num/reduce_total_num
                if get_rank() == 0: 
                    print('epoch %d ,  acc_trip on all GPUs= %.4f' % (epoch, acc_1_trip))

            else:
                acc_1_num, total_1_num = cal_acc_multi(answers_trip, preds_trip)
                reduce_acc_num=reduce_tensor(torch.tensor(acc_1_num).cuda(args.local_rank)).item()
                reduce_total_num=reduce_tensor(torch.tensor(total_1_num).cuda(args.local_rank)).item()
                acc_1_trip = reduce_acc_num/reduce_total_num
                if get_rank() == 0: 
                    print('epoch %d ,  acc_trip on all GPUs= %.4f' % (epoch, acc_1_trip))

            if acc_1_trip > best_acc_t:
                
                best_acc_t = acc_1_trip
                best_epoch_t = epoch
                if not args.pretrain:
                    if get_rank() == 0:
                        f=open(args.model_dir+"/predictions.json", 'w')
                        json.dump(id2pred_trip, f)
                        f.close()

                        """
			# ablations on two encoders
			#  LXMERTenc-T5dec
                            if args.load_pthpath == "":
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/C1_LXMERTencOnly_noPre_predictions.json", 'w') #GPT-noPre
                            else:
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/C3_LXMERTencOnly_predictions.json", 'w') #GPT

                            json.dump(id2pred_trip, fx)
                            fx.close()
			# T5enc-T5dec
                            if args.load_pthpath == "":
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/C2_T5encOnly_noPre_predictions.json", 'w') #GPT-noPre
                            else:
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/C4_T5encOnly_predictions.json", 'w') #GPT
                            
                            json.dump(id2pred_trip, fx)
                            fx.close()
			"""
                        """
			# ablations on Knowledge types
			if args.gpt3:
                            if args.input_type==0 and args.load_pthpath == "":
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/A2_noPre_predictions.json", 'w') #GPT-noPre
                                json.dump(id2pred_trip, fx)
                                fx.close()
                            elif args.input_type==1 and (args.load_pthpath != ""):
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/A3_noWiki_predictions.json", 'w') #GPT
                                json.dump(id2pred_trip, fx)
                                fx.close()
                            elif args.input_type==2 and (args.load_pthpath != ""):
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/A4_noOFA_predictions.json", 'w') #GPT
                                json.dump(id2pred_trip, fx)
                                fx.close()
                            elif args.input_type==3 and (args.load_pthpath == ""):
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/B1_onlyGPT3_predictions.json", 'w')  #GPT-noPre
                                json.dump(id2pred_trip, fx)
                                fx.close()
                        else:
                            if args.input_type==0 and args.load_pthpath != "":
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/A5_noGPT3_predictions.json", 'w')    #noGPT
                                json.dump(id2pred_trip, fx)
                                fx.close()
                            elif args.input_type==0 and args.load_pthpath == "":
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/A6_noGPT3noPre_predictions.json", 'w')    #noGPT
                                json.dump(id2pred_trip, fx)
                                fx.close()
                            
                            elif args.input_type==1 and (args.load_pthpath == ""):
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/B2_onlyOFA_predictions.json", 'w')   #noGPT-noPre
                                json.dump(id2pred_trip, fx)
                                fx.close()
                            elif args.input_type==2 and (args.load_pthpath == ""):
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/B3_onlyWiki_predictions.json", 'w')  #noGPT-noPre
                                json.dump(id2pred_trip, fx)
                                fx.close()
                            elif args.input_type==3 and (args.load_pthpath == ""):
                                fx=open("/mnt/bn/qingyi-bn-lq/okvqa-output/B4_onlyVisualNoKnowledge_predictions.json", 'w') #noGPT-noPre
                                json.dump(id2pred_trip, fx)
                                fx.close()

                        """     
                        

                        print('saving model at epoch', epoch, '!!')
                        if get_world_size() > 1:
                            torch.save(model.module.state_dict(), args.model_dir+'/best_finetuned_model_'+model_name+'.pth')
                        else:
                            torch.save(model.state_dict(), args.model_dir+'/best_finetuned_model_'+model_name+'.pth')

            if get_rank() == 0: 
                print("best_acc@1t={:.2%}, epoch{}\n\n".format(best_acc_t, best_epoch_t))

            model.train()        
        if args.pretrain:
            if get_rank() == 0:
                if get_world_size() > 1:
                    torch.save(model.module.state_dict(), args.model_dir+ '/model_for_epoch_%d.pth' % epoch)
                else:
                    torch.save(model.state_dict(), args.model_dir+ '/model_for_epoch_%d.pth' % epoch)

        barrier()
        

    dist.destroy_process_group()
if __name__ == "__main__":
    train()
