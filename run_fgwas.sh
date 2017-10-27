#!/bin/bash
#SBATCH -J run_fgwas
#SBATCH -o run_fgwas.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p normal
#SBATCH -t 23:59:59 
#SBATCH -A BIGS2-Community-Det
# SLURM email notifications are now working on Lonestar 5 
#SBATCH --mail-user=username@tacc.utexas.edu
#SBATCH --mail-type=begin   # email me when the job starts
#SBATCH --mail-type=end     # email me when the job finishes
module load python 
cd /work/04333/chaoh/FGWAS/
export PATH=/work/04333/chaoh/software/anaconda2/bin:$PATH
python2 ./test.py ./data/ ./result/
