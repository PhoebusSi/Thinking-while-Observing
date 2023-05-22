from model_LXM2T5 import T5tokenizer, LXMT52T5, LXMtokenizer
import tqdm
from dataset_val4LXMT5 import KgDatasetVal
model = LXMT52T5()
model.module.load_state_dict(torch.load("xxxx.pth"))
test_dataset = KgDatasetVal(val=False)


test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=my_val_collate)

model.eval()  
answers = []  # [batch_answers,...]
preds = []  # [batch_preds,...]
preds_list = []
answers_list = []
id2pred_list = {}
for i, batch_data in enumerate(tqdm(test_dataloader)):
    with torch.no_grad():
        val_T5_input_id = torch.stack(batch_data['T5_input_ids']).to(device)
        val_T5_input_mask = torch.stack(batch_data['T5_input_masks']).to(device)
        val_visual_faetures = torch.tensor(np.array(batch_data['img'])).float().to(device)           
        val_spatial_features = torch.tensor(np.array(batch_data['spatial'])).float().to(device)
 
        val_LXM_input_id = torch.stack(batch_data['LXM_input_ids']).to(device)
        val_LXM_input_mask = torch.stack(batch_data['LXM_input_masks']).to(device)
        val_LXM_token_type_ids = torch.stack(batch_data['LXM_token_type_ids']).to(device)
                    


        val_outputs = model(train=False, LXM_source_ids=val_LXM_input_id, LXM_source_masks=val_LXM_input_mask,T5_source_ids=val_T5_input_id, T5_source_masks=val_T5_input_mask,token_type_ids=val_LXM_token_type_ids, visual_features=val_visual_faetures, spatial_features=val_spatial_features,T5_target_ids=None)
                        

                
        val_list_predict = T5tokenizer.batch_decode(val_outputs, skip_special_tokens=True)


 
        for i, pre in enumerate(batch_data['ans']):
                        
            preds_list.append(val_list_predict[i])
                        
            answers_list.append(batch_data['ans'][i])
                         
            id2pred_list[str(batch_data['id'][i])]=val_list_predict[i]




f=open("file_to_save.json", 'w')
json.dump(id2pred_list, f)
f.close()
                        