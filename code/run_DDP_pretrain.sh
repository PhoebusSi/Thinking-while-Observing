#!/bin/bash


export NCCL_P2P_LEVEL=NVL
echo "dataset $1, model dir $2, input type $3, describe $4, lr $5, lr_LXM $6, batch size $7, wiki num $8, gpu_num $9 "

export dataset=$1
export model_dir=$2
mkdir $model_dir
export input_type=$3
#model_name to save
export describe=$4
export lr=$5
export lr_LXM=$6
export batch_size=$7
# export port=$7
export wiki_num=$8
export gpu_num=$9
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

python3 -m torch.distributed.launch --nproc_per_node $gpu_num  \
    --nnodes=${ARNOLD_WORKER_NUM} --node_rank=${ARNOLD_ID} --master_addr=${METIS_WORKER_0_HOST} --master_port ${port} \
    train4LXMT5_DDP.py \  
    --dataset $dataset \
    --model_dir $model_dir \
    --input_type $input_type \
    --describe $describe \
    --learning_rate $lr \
    --learning_rate_LXM $lr_LXM \
    --validate \
    --batch_size $batch_size \
    --num_wiki $wiki_num \
    --pretrain 
    