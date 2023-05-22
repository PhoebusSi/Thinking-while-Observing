#!user/bin/env python
# -*- coding:utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--inference", action="store_true", help='complete dataset or not')
parser.add_argument("--pretrain", default=False, action="store_true", help='use vqa2.0 or not')
parser.add_argument("--gpt3", default=False, action="store_true", help='use gpt3 to train on okvqa')
parser.add_argument("--visualBERT", default=False, action="store_true", help='use visualBERT, if false use LXMERT')

parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--seed', type=int, default=4, 
                    help='random seed!')
parser.add_argument('--num_wiki', type=int, default=25, 
                    help='the number of wiki passages')
parser.add_argument('--num_epochs', type=int, default=40, 
                    help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='LR')
parser.add_argument('--learning_rate_LXM', type=float, default=0.00001, 
                    help='LR_LXM')                 
parser.add_argument('--model_dir', type=str, default='xxx/',
                    help='model file path')
parser.add_argument('--input_type', type=int, default=1,#200,
                    help='input types: 1==Q-OFA-C-L-O; 2==Q-C-L-O; 3==Q-OFA-L-O; 4==Q-OFA-C-O; 5==Q-OFA-C-L')
parser.add_argument('--describe', type=str, default='',
                    help='the model description used as the saved-model name')
parser.add_argument("--load_pthpath", default="",
                    help="To continue training, path to .pth file of saved checkpoint.")
parser.add_argument("--validate", default='True', action="store_true", help="Whether to validate on val split after every epoch.")
parser.add_argument("--dataset", default="okvqa", help="dataset that model training on")
parser.add_argument("--ofa", default="normal", help=" normal or finetune --- load the knowledge from Normal OFA or vqav2-Finetuned OFA")
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
print(args)
