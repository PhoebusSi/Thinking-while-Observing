#!/bin/bash
#!/usr/bin/env bash
#!/bin/sh
export load_pthpath=${10}
export pre_epo=${11}
export load_pthmodel=$load_pthpath/model_for_epoch_$pre_epo.pth
#要做的任务
export NCCL_P2P_LEVEL=NVL
cd /opt/tiger/okvqa
export dataset=$1
#要保存的模型的路径
export model_dir=$2
mkdir $model_dir
mkdir $load_pthpath
echo "hdfs get"
echo "$load_pthmodel"
# cd /opt/tiger/okvqa
# hdfs dfs -get hdfs:/home/byte_arnold_hl_mlnlc/user/sqy/data4OKVQA/data
# hdfs dfs -get hdfs:/home/byte_arnold_hl_mlnlc/user/sqy/data4OKVQA/model
# hdfs dfs -get hdfs:///home/byte_arnold_hl_mlnlc/user/sqy/data4OKVQA/2E1D_pretrain_checkpoint/$load_pthmodel $load_pthpath
echo "hdfs done"
echo "$1, $2, $3, $4, $5, $6, $7, $8, $9, ${10}, ${11}, ${12}"
echo "dataset $1, model dir $2, input type $3, describe $4, lr $5, lr_LXM $6, batch_size $7, wiki num $8, gpu_num $9, load path ${10}, pre_epo ${11}, seed ${12}"


#iput的类型
export input_type=$3
#model_name to save
export describe=$4
export lr=$5
export lr_LXM=$6
#fine-tune比较好的batchsize是32
export batch_size=$7
export wiki_num=$8
export gpu_num=$9
export seed=${12}
# echo "$1, $2, $3, $4, $5"
# export pretrain=$1
# #有预训练则设置1，没有则设置0
# export model_dir=$2
# export dataset=$3
# #先默认都是okvqa，后期可以看看krvqa上的性能
# export load_pthpath=$4
# #默认为空
# export describe=$5
# #对这个描述进行切割，对应的设置T5的input的类型，方便ablation study。
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"



export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
# echo "master ip: ${METIS_WORKER_0_HOST}"
# ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# port=${ports[0]}
# echo "master port: ${port}"
#这套batchsize=128-4卡A100-fine-tune的参数80g*4,利用率100%
# python3 -m torch.distributed.launch --nproc_per_node $gpu_num  \
#     --nnodes=3 --node_rank=0 --master_addr="10.136.197.78" --master_port "10259" \
python3 -m torch.distributed.launch --nproc_per_node $gpu_num  \
    --nnodes=${ARNOLD_WORKER_NUM} --node_rank=${ARNOLD_ID} --master_addr=${METIS_WORKER_0_HOST} --master_port ${port} \
    train4LXMT5_jiqun_wiki_DDP_multiVal_GPT3.py \
    --dataset $dataset \
    --model_dir $model_dir \
    --input_type $input_type \
    --describe $describe \
    --learning_rate $lr \
    --learning_rate_LXM $lr_LXM \
    --validate \
    --gpt3 \
    --ofa finetune \
    --batch_size $batch_size \
    --load_pthpath $load_pthmodel \
    --num_wiki $wiki_num \
    --seed $seed \
    --visualBERT
    # 32
    # 128
    # --dataset okvqa --model_dir finetune-model  --input_type 1 --describe finetune-model --validate --batch_size 128

# python3 train4LXMT5_jiqun.py \
#     --dataset $dataset \
#     --model_dir $model_dir \
#     --input_type $input_type \
#     --describe $describe \
#     --validate
mv $model_dir /mnt/bn/qingyi-bn-lq/finetunedModelonOKVQA/visualBERT_seed${seed}-${describe}-4Kstep_multiVal_GPT3_finetuneOFA_${lr}${lr_LXM}FTwiki${wiki_num}-From-1e-41e-5PretrainWiki25Epo${pre_epo}/


# export final_describe=input_type$input_type-$describe-loadPretrain$load_pthpath-$lr-$batch_size
# export final_describe=PreFT-input_type$input_type-$describe-loadPretrain-$load_pthpath-$pre_epo-$lr-$batch_size-wiki-$wiki_num
# mv $model_dir /mnt/bn/qingyi-bn-lq/okvqa-output/$final_describe
# hdfs dfs -put $final_describe hdfs:/home/byte_arnold_hl_mlnlc/user/sqy/data4OKVQA/2E1D_loadPretrain_checkpoint
# """
#!/usr/bin/env bash

# CUR_DIR=$(cd $(dirname $0); pwd)
# cd $CUR_DIR

# # 取 worker0 第一个 port
# ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# port=${ports[0]}

# echo "total workers: ${ARNOLD_WORKER_NUM}"
# echo "cur worker id: ${ARNOLD_ID}"
# echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
# echo "master ip: ${METIS_WORKER_0_HOST}"
# echo "master port: ${port}"

# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
# export NCCL_SOCKET_IFNAME=eth0
# # export NCCL_DEBUG=INFO

#原始的脚本启动方式
#python3 train.py $@

# # 使用 torch.distributed.launch 启动
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
# --nnodes $ARNOLD_WORKER_NUM \
# --node_rank $ARNOLD_ID \
# --nproc_per_node $ARNOLD_WORKER_GPU \
# --master_addr $METIS_WORKER_0_HOST \
# --master_port $port \
# train4LXMT5_jiqun_DDP.py --dataset okvqa --model_dir finetune-model  --input_type 1 --describe finetune-model --validate --batch_size 256

#统计
#finetune的程序： batchsize=16 4卡A100 显存占35G 2.5min/epoch ！(CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4  --master_port 4309 train4LXMT5_jiqun_DDP.py --dataset okvqa --model_dir finetune-model  --input_type 1 --describe finetune-model --validate --batch_size 16 )
#finetune的程序： batchsize=32 4卡A100 显寸占45G 1.5min/epoch! （CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4  --master_port 4309 train4LXMT5_jiqun_DDP.py --dataset okvqa --model_dir finetune-model  --input_type 1 --describe finetune-model --validate --batch_size 32）（有时会报错，可能是线程太多，cpu或内存不够）
#finetune的程序： batchsize=32 1卡A100 单卡显存占41G 5min/epoch ！(CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4  --master_port 4309 train4LXMT5_jiqun_DDP.py --dataset okvqa --model_dir finetune-model  --input_type 1 --describe finetune-model --validate --batch_size 32 )



#pretrain的程序： batchsize=128 4卡A100 显存占29G 10.5min/epoch !(CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 4  --master_port 4009 train4LXMT5_jiqun_DDP.py --dataset okvqa --model_dir pretrain-model  --input_type 1 --describe pretrain-model --validate --batch_size 128  --pretrain)
#pretrain的程序： batchsize=256 4卡A100 显存占44G 9.5min/epoch !(CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 4  --master_port 4009 train4LXMT5_jiqun_DDP.py --dataset okvqa --model_dir pretrain-model  --input_type 1 --describe pretrain-model --validate --batch_size 256  --pretrain)