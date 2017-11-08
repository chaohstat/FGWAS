#!/bin/bash
#SBATCH --job-name=run_fgwas
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000
module load anaconda 
cd /nas/longleaf/home/chaoh/FGWAS/
export PATH=/nas/longleaf/apps/anaconda/4.3.0/anaconda/bin:$PATH
python ./test.py ./data/ ./result/
