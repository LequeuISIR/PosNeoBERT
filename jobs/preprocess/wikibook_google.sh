#!/bin/bash
#SBATCH --job-name=preprocess-wikibook-google
#SBATCH --output=logs/%x_output.txt
#SBATCH --error=logs/%x_error.txt
#SBATCH --time=24:00:00
#SBATCH --partition=funky
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --cpus-per-task=32               # number of cpus per node
#SBATCH --mem=128G                       # memory per node


# Launch the tokenization
uv run $HOME/repos/PosNeoBERT/scripts/pretraining/preprocess.py \
    wandb.mode=disabled \
    trainer.dir=$HOME/repos/PosNeoBERT/logs/$SLURM_JOB_NAME \
    hydra.run.dir=$HOME/repos/PosNeoBERT/logs/$SLURM_JOB_NAME/hydra \
    tokenizer=google \
    tokenizer.max_length=512 \
    dataset=wikibook \
    dataset.path_to_disk=/data/lequeu/PosNeoBERT/tokenized_datasets/new_padded_wikibook_google_512 \