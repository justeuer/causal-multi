#!/bin/bash

#! /usr/bin/env bash
#SBATCH --job-name=mgpt
#SBATCH --output=/hits/basement/nlp/steuerjs/out/slurm/%j.out
#SBATCH --error=/hits/basement/nlp/steuerjs/out/slurm/%j.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=genoa-hopper.p

. activate /home/steuerjs/miniconda3/envs/causal

MODEL="ai-forever/mgpt"
# MODEL="EleutherAI/pythia-70m" 

NUM_BATCHES=${2:-25}
BATCH_SIZE=${3:-16}

EVAL_SEEDS=(41)

# de -> es
python generalization.py --model_id $MODEL --eval_seeds $EVAL_SEEDS --source_set subject_verb_agreement/3rd_masc_de --eval_set subject_verb_agreement/3rd_masc_es --batch_size $BATCH_SIZE --num_batches $NUM_BATCHES

# es -> de
python generalization.py --model_id $MODEL --eval_seeds $EVAL_SEEDS --source_set subject_verb_agreement/3rd_masc_es --eval_set subject_verb_agreement/3rd_masc_de --batch_size $BATCH_SIZE --num_batches $NUM_BATCHES
