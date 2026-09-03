#!/bin/bash
#SBATCH --job-name=joint_seg
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=100:00:00
#SBATCH --output=./output/joint_seg_%A_%a.out
#SBATCH --error=./error/joint_seg_%A_%a.err
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --array=0-0

set -euo pipefail

module purge
module load arch/h100
module load pytorch-gpu

PROJECT_DIR=/lustre/fswork/projects/rech/crp/ulq73oz/IABM2025/JointSegmentationRegistration
cd "$PROJECT_DIR"

lambda_reg_values=(1)
lambda_reg=${lambda_reg_values[$SLURM_ARRAY_TASK_ID]}
lambda_jac_values=(20)
lambda_jac=${lambda_jac_values[$SLURM_ARRAY_TASK_ID]}
pretrain_registration_epochs=400
segmentation_method=temporal_neighbors

set -x
srun python -u src/train.py \
    --dataset "$PROJECT_DIR/data/babofet.yaml" \
    --segmentation_checkpoint "$PROJECT_DIR/last_segmentation_new.pt" \
    --lambda_reg "$lambda_reg" \
    --lambda_jac "$lambda_jac" \
    --lambda_seg 1 \
    --lambda_sim 0.0005 \
    --lambda_anchor 1.0 \
    --pretrain_registration_epochs "$pretrain_registration_epochs" \
    --segmentation_method "$segmentation_method" \
    --max_epochs 5000 \
    --check_val_every_n_epoch 5
