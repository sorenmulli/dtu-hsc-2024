#!/bin/sh
#BSUB -J trainSpecDec
#BSUB -o logs/dccrn-train_%J.out
#BSUB -e logs/dccrn-train_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 960
####BSUB -u mariejuhljorgensen@gmail.com
####BSUB -B
####BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment	
# module load scipy/VERSION

nvidia-smi
module load cuda/11.6

# activate the virtual environment 
# NOTE: needs to have been built with the same SciPy version above!
source /zhome/e3/b/155491/miniconda3/etc/profile.d/conda.sh
conda activate hsc

python -m dtu_hsc_solutions.ml_models.huggingface_model_training --task 2 --model dccrnet --epochs 100 --k-folds 1 --ir True  --freeze-encoder True --loss spec
