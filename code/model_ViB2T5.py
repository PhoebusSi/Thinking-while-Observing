import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import copy
from config4LXMT5_DDP import args
import collections
from transformers import LxmertConfig, LxmertTokenizer, LxmertModel,BertTokenizer
from transformers import T5Tokenizer, T5Model, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import  BaseModelOutputWithPastAndCrossAttentions
T5tokenizer = T5Tokenizer.from_pretrained("../model/t5-large")#"t5-large")
LXMtokenizer = BertTokenizer.from_pretrained('../model/bert-base-uncased/vocab.txt') 
T5config = T5Config.from_pretrained('../model/t5-large')
from transformers import VisualBertConfig, VisualBertModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ViBT52T5(nn.Module):
    def __init__(self):
        super(ViBT52T5, self).__init__()
        self.T5model = T5ForConditionalGeneration.from_pretrained("../model/t5-large").to(device)
        self.ViBmodel = VisualBertModel.from_pretrained('../model/visualBERT').to(device)
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.ReLU(inplace=True),   
            torch.nn.Linear(1024, 1024)
            )


    def LXMT5end2T5dec(self,  train=None, LXM_source_ids=None, LXM_source_masks=None,T5_source_ids=None, T5_source_masks=None,token_type_ids=None, visual_features=None, spatial_features=None,T5_target_ids=None,T5_target_masks=None):
        if 1:

            ViB_encoder_output_seq = self.ViBmodel(input_ids=LXM_source_ids, attention_mask=LXM_source_masks,token_type_ids=token_type_ids, visual_embeds=visual_features) 
            ViB_VL_encoder_output_seq = ViB_encoder_output_seq[0]
            final_ViB_encoder_output_seq = self.mapping(ViB_VL_encoder_output_seq)




	# w/o wiki passages
        #T5_encoder_output_seq = self.T5model.encoder(input_ids=T5_source_ids, attention_mask=T5_source_masks)
        #final_encoder_output_seq = torch.cat((final_ViB_encoder_output_seq, T5_encoder_output_seq["last_hidden_state"]),1)
        

        # w/ wiki passages
	if 1:
            final_encoder_output_seq_list = []
            for ind in range(args.num_wiki):
                T5_encoder_output_seq = self.T5model.encoder(input_ids=T5_source_ids[:,ind,:], attention_mask=T5_source_masks[:,ind,:])
                tmp_encoder_output_seq = torch.cat((final_ViB_encoder_output_seq, T5_encoder_output_seq["last_hidden_state"]),1)
                final_encoder_output_seq_list.append(tmp_encoder_output_seq)
            final_encoder_output_seq = torch.cat(final_encoder_output_seq_list,1)
        my_order_dict=T5_encoder_output_seq 
        my_order_dict.last_hidden_state=final_encoder_output_seq

        if train:
            outputs = self.T5model(encoder_outputs=my_order_dict, labels=T5_target_ids, decoder_attention_mask=T5_target_masks)
            return outputs
        else:
            if torch.cuda.device_count() > 1:
                pred = self.T5model.generate(encoder_outputs=my_order_dict)
            else:
                pred = self.T5model.generate(encoder_outputs=my_order_dict)
            return pred


    def forward(self,  train=None, LXM_source_ids=None, LXM_source_masks=None,T5_source_ids=None, T5_source_masks=None,token_type_ids=None, visual_features=None, spatial_features=None,T5_target_ids=None,T5_target_masks=None):
        return self.LXMT5end2T5dec(train, LXM_source_ids, LXM_source_masks, T5_source_ids, T5_source_masks, token_type_ids, visual_features, spatial_features, T5_target_ids, T5_target_masks)
