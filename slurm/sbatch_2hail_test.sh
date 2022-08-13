#!/bin/bash

#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
# memory in MB
#SBATCH --mem=4096
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_2022_08_09/slurm_output/out/test_patch_run_%04a_stdout.txt
#SBATCH --error=/ourdisk/hpc/ai2es/severe_nowcasting/hail_nowcasting/trained_at_init_time_2022_08_09/slurm_output/err/test_patch_run_%04a_stderr.txt
#SBATCH --time=05:00:00
#SBATCH --job-name=generate_patches
#SBATCH --mail-user=tobias.schmidt@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/tgschmidt/auto-patcher/src
#SBATCH --array=0-49
#
#################################################
/home/tgschmidt/sn_env/bin/python -u patcher.py --config_path /home/tgschmidt/auto-patcher/configs/2hail_test.cfg --run_num $SLURM_ARRAY_TASK_ID


