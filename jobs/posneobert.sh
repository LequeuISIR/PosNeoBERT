#!/bin/bash
#SBATCH --job-name=entropy-cos-untie-softpick-swiglu-PosNeoBERT
#SBATCH --time=48:00:00
#SBATCH --partition=hard   
#SBATCH --exclude=top 
#SBATCH --nodes=1                    # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --gpus-per-task=2           # number of gpus per node
#SBATCH --cpus-per-task=16           # number of cpus per nod
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo Launching

echo "Allocated nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo $MASTER_ADDR
echo $MASTER_PORT
# Maximum number of threads in the OpenMP parallel region (defaults to 1)
# (called by `torch.distributed.run`, called by `accelerate launch`)
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE))

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Define the command to run on each node
cmd=(
    accelerate launch \
    --config_file=$HOME/repos/PosNeoBERT/conf/accelerate_deepspeed_zero2.yaml \
    --machine_rank=\$SLURM_NODEID \
    --num_cpu_threads_per_process=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE)) \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE)) \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --gradient_clipping=1.0 \
    $HOME/repos/PosNeoBERT/scripts/pretraining/pretrain.py \
    wandb.name=$SLURM_JOB_NAME \
    wandb.mode=offline \
    wandb.dir=/data/lequeu/logs/$SLURM_JOB_NAME/wandb \
    wandb.mode=offline \
    trainer.dir=/data/lequeu/logs/$SLURM_JOB_NAME \
    trainer.entropy_regularization_lambda=0.01 \
    hydra.run.dir=/data/lequeu/logs/$SLURM_JOB_NAME/hydra \
    dataset=wikibook \
    dataset.path_to_disk=/data/lequeu/PosNeoBERT/tokenized_datasets/new_padded_wikibook_google_512/ \
    tokenizer=google \
    model=[posneobert] \
    model.positional_embed_init=2dim_cosine \
    model.mix_attentions=sum \
    model.untie_cls=true \
    model.attention_ativation=softpick \
    model.hidden_act=swiglu \
    model.random_offset=true \
    datacollator=mlm_20 \
    optimizer=adamw \
    scheduler=cosine_decay \
    trainer.gradient_accumulation_steps=4 \
    dataloader.train.batch_size=16 \
    tokenizer.max_length=512
)

# Load python environment
source .venv/bin/activate


bash -c "$(for a in "${cmd[@]}" ; do echo -n \"$a\" "" ; done)"
