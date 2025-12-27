#!/bin/bash
#SBATCH --job-name=kg_run

#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -euo pipefail

echo "========== SLURM JOB INFO =========="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node(s):       $SLURM_NODELIST"
echo "GPUs:          ${SLURM_GPUS:-N/A}"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "===================================="

# Make sure log directory exists
mkdir -p logs

# ---- Modules / environment (edit to match your cluster) ----
# module purge
# module load cuda
# module load python

# If you use conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate YOUR_ENV

# If you use venv:
# source /path/to/venv/bin/activate

# ---- Sanity check that GPU is visible ----
echo "---- nvidia-smi ----"
nvidia-smi || true

# ---- Recommended performance env vars (safe defaults) ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# (Optional) for PyTorch + NCCL stability even on 1 GPU
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

# ---- Run your command ----
# Replace this with your actual training/inference command

python run_project.py \
    --max_kg_triples 50000 \
    --embed_epochs 1 \
    --embed_batch_size 256 \
    --classifier_epochs 1 \
    --classifier_batch_size 128 \
    --force


# Example:
# srun python -u train.py --config config.yaml

echo "End time: $(date)"
