#!/bin/sh
#BSUB -J createSpecDec
#BSUB -o logs/create_%J.out
#BSUB -e logs/create_%J.err
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

python -m dtu_hsc_solutions data dccrnet-tuned --level Task_2_Level_1 --overwrite-output --dir-postfix _aligned_ir_spec_dec

python -m hsc_given_code.evaluate --text_file data/Task_2_Level_1/Task_2_Level_1_text_samples.txt --model_path data/deepspeech-0.9.3-models.pbmm --scorer_path data/deepspeech-0.9.3-models.scorer --audio_dir data/output/dccrnet-tuned/Task_2_Level_1_aligned_ir_spec_dec
