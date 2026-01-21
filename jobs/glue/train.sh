#!/bin/bash
#SBATCH --job-name=glue-detach-PosNeoBERT
#SBATCH --time=0:30:00
#SBATCH -C h100
#SBATCH -A znb@h100
#SBATCH --qos qos_gpu_h100-dev
#SBATCH --nodes=1                    # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --gpus-per-task=1           # number of gpus per node
#SBATCH --cpus-per-task=24           # number of cpus per nod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

# Get run parameters
export meta_task=$1
export task=$2
export model=$3
export ckpt=$4
export batch_size=$5
export lr=$6
export seed=$7
export transfer_from_task=$8
export num_ckpt=$9
export from_hub=${10}
export max_length=${11}

# Define run directory
export run_dir=${12}

export mixed_precision=${13}
export early_stopping=${14}
export wd=${15}

# Load modules, python environment and install local package
# source activate base
# conda init
# conda activate .venv

# cd $HOME/neo-bert
# pip install -e .

# Error handling for missing parameters
if [[ -z "$task" || -z "$batch_size" || -z "$lr" || -z "$seed" || -z "$model" || -z "$ckpt"  || -z "$run_dir"  || -z "$transfer_from_task"  || -z "$num_ckpt" ]]; then
    echo "Missing arguments."
    exit 1
fi

# Log parameters
echo "Running $meta_task task: $task"
echo "Logging in directory $run_dir"
echo "Model: $model, Checkpoint: $ckpt"
echo "Batch size: $batch_size, Learning rate: $lr, Seed: $seed"
echo "Transfering from matching task if available: $transfer_from_task"
echo "Number of checkpoints to merge: $num_ckpt"
echo "Weight decay: $wd"

module load arch/h100
module load pytorch-gpu/py3/2.5.0
export PYTHONPATH=$WORK/repos/PosNeoBERT/src:$PYTHONPATH

uv run $WORK/repos/PosNeoBERT/scripts/evaluation/run_glue.py \
    dataset=$meta_task \
    task=$task \
    hydra.run.dir=$run_dir/hydra \
    wandb.mode=disabled \
    model.name=$model \
    model.from_hub=$from_hub \
    model.pretrained_config_path=$SCRATCH/logs/$model/hydra/.hydra/config.yaml \
    model.pretrained_checkpoint_dir=$SCRATCH/logs/$model \
    model.pretrained_checkpoint=$ckpt \
    model.num_checkpoints_to_merge=$num_ckpt \
    model.transfer_from_task=$transfer_from_task \
    trainer.dir=$run_dir \
    trainer.train_batch_size=$batch_size \
    trainer.max_ckpt=0 \
    optimizer.hparams.lr=$lr \
    optimizer.hparams.weight_decay=$wd \
    id=$model_$ckpt_$task_$batch_size_$lr_$seed \
    seed=$seed \
    tokenizer.max_length=$max_length \
    trainer.early_stopping=$early_stopping \
    trainer.mixed_precision=$mixed_precision \
    +flash_attention=false \