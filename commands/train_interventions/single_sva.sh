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


python causalgym/test_all.py --model $MODEL --only-das --dataset subject_verb_agreement/3rd_masc_es
python causalgym/test_all.py --model $MODEL --only-das --dataset subject_verb_agreement/3rd_masc_de

