import json
import string
import regex

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def cal_acc_multi(ground_truth, preds, return_id = False):
    all_num = len(ground_truth)
    acc_num = 0
    ids = []
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
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


def ensemble(a):
    return max(a[::-1], key = a.count)

# Ground Truth Answers
f=open("/root/okvqa/data/okvqa_val.json", "r")
answer_dict=json.load(f)
f.close()
for k in answer_dict.keys():
    for a_ind, a in enumerate(answer_dict[k]['multi_answers']):
        answer_dict[k]['multi_answers'][a_ind] = normalize_answer(answer_dict[k]['multi_answers'][a_ind])


# Load Predictions (for example, ensemble of three models' predictions)
f1=open("/mnt/bn/qingyi-hl/finetunedModelonOKVQA/1e-41e-5FTwiki25-From-1e-41e-5PretrainWiki25Epo0/FTwiki25FromPretrainWiki25Epo0-1e41e5/predictions.json", "r")
predict0_dict=json.load(f1)
for p in predict0_dict.keys():
    predict0_dict[p]=normalize_answer(predict0_dict[p])
f1.close()
f2=open("/mnt/bn/qingyi-hl/finetunedModelonOKVQA/1e-41e-5FTwiki25-From-1e-41e-5PretrainWiki25Epo1/predictions.json", "r")
predict1_dict=json.load(f2)
for p in predict1_dict.keys():
    predict1_dict[p]=normalize_answer(predict1_dict[p])
f2.close()
f3=open("/mnt/bn/qingyi-hl/finetunedModelonOKVQA/1e-41e-5FTwiki25-From-1e-41e-5PretrainWiki25Epo2/predictions.json", "r")
predict2_dict=json.load(f3)
for p in predict2_dict.keys():
    predict2_dict[p]=normalize_answer(predict2_dict[p])
f3.close()



answer_list=[]
predict0_list=[]
predict1_list=[]
predict2_list=[]
emsemble_predict=[]
for k in answer_dict.keys():
    answer_list.append( answer_dict[k]['multi_answers'])
    predict0_list.append( predict0_dict[k])
    predict1_list.append( predict1_dict[k])
    predict2_list.append( predict2_dict[k])
    
    emsemble_predict.append(ensemble([predict0_dict[k], predict1_dict[k], predict2_dict[k])
    


acc_n0,all_n0=cal_acc_multi(answer_list,predict0_list)
acc_n1,all_n1=cal_acc_multi(answer_list,predict1_list)
acc_n2,all_n2=cal_acc_multi(answer_list,predict2_list)

acc_ens,all_ens=cal_acc_multi(answer_list,emsemble_predict)

print("0-accuracy",acc_n0/all_n0)
print("1-accuracy",acc_n1/all_n1)
print("2-accuracy",acc_n2/all_n2)


print("ensemble-accuracy",acc_ens/all_ens)