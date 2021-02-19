#! /bin/bash
export PATH_POSTFIX=$1
export MISC_ARGS=$2

export DATA_ROOT="./outputs/Experiments"
export DATASET=${DATASET:-ArgoverseMapDataset} #KITTINMPairDataset}
export TRAINER=${TRAINER:-HardestContrastiveLossTrainer}
export MODEL=${MODEL:-ResUNetBN2C}
export MODEL_N_OUT=${MODEL_N_OUT:-32}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-1}
export MAX_EPOCH=${MAX_EPOCH:-200}
export BATCH_SIZE=${BATCH_SIZE:-1}
export ITER_SIZE=${ITER_SIZE:-1}
export VOXEL_SIZE=${VOXEL_SIZE:-0.3}
export POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER=${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER:-0.5} 
export CONV1_KERNEL_SIZE=${CONV1_KERNEL_SIZE:-7}
export EXP_GAMMA=${EXP_GAMMA:-0.99}
export RANDOM_SCALE=${RANDOM_SCALE:-False} #scale doesn't work for the local map
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export ARGO_PATH=${ARGO_PATH:-/home/chrischoy/datasets/KITTI_FCGF}
export VERSION=$(git rev-parse HEAD)

export OUT_DIR=${DATA_ROOT}/${DATASET}-v${VOXEL_SIZE}/${TRAINER}/${MODEL}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}i${ITER_SIZE}-modelnout${MODEL_N_OUT}${PATH_POSTFIX}/${TIME}

export PYTHONUNBUFFERED="True"

echo $OUT_DIR

mkdir -m 755 -p $OUT_DIR

LOG=${OUT_DIR}/log_${TIME}.txt

#echo "Host: " $(hostname) | tee -a $LOG
#echo "Conda " $(which conda) | tee -a $LOG
#echo $(pwd) | tee -a $LOG
#echo "Version: " $VERSION | tee -a $LOG
#echo "Git diff" | tee -a $LOG
#echo "" | tee -a $LOG
#git diff | tee -a $LOG
#echo "" | tee -a $LOG
#nvidia-smi | tee -a $LOG

# Training
python train.py \
	--dataset ${DATASET} \
	--trainer ${TRAINER} \
	--model ${MODEL} \
	--model_n_out ${MODEL_N_OUT} \
	--conv1_kernel_size ${CONV1_KERNEL_SIZE} \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--batch_size ${BATCH_SIZE} \
	--iter_size ${ITER_SIZE} \
	--max_epoch ${MAX_EPOCH} \
	--voxel_size ${VOXEL_SIZE} \
	--out_dir ${OUT_DIR} \
	--use_random_scale ${RANDOM_SCALE} \
	--positive_pair_search_voxel_size_multiplier ${POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER} \
	--kitti_root ${ARGO_PATH} \
	--hit_ratio_thresh 0.2 \
	--nn_max_n  100 #otherwise the gpu memory of my laptop is not enough

	$MISC_ARGS 2>&1 | tee -a $LOG

# Test
#python -m scripts.test_argo_save_results \
#	--kitti_root ${ARGO_PATH} \
#	--save_dir "outputs/Experiments/KITTIMapDataset-v0.3/HardestContrastiveLossTrainer/ResUNetBN2C/SGD-lr1e-1-e200-b2i1-modelnout32/2021-02-11_22-35-59"
#