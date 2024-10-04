#!/bin/sh
#BSUB -J train_all
#BSUB -o logs/0_TrCrEv_%J.out
#BSUB -e logs/0_TrCrEv_%J.err
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

levels=(1 2 3)

name="t2-ir-all"    
    echo "Training"
    python -m dtu_hsc_solutions.ml_models.huggingface_model_training\
        --data-path data\
        --model dccrnet\
        --task 2\
        --level all\
        --k-folds 1\
        --epochs 50\
        --loss spec\
        --name $name\
	--ir True\
    
for level in "${levels[@]}"; do
    name="t1l$level-ir-all"
    echo "Predicting on level $level"
    python -m dtu_hsc_solutions data dccrnet-tuned\
        --level Task_2_Level_$level\
        --overwrite-output\
        --dir-postfix $name\
        --weights-dir "ml_models/$name/load_dccrnet_model_fold_1_model.pth"
    echo "Evaluating on level $level"

    python -m hsc_given_code.evaluate\
        --text_file data/Task_2_Level_$level/Task_2_Level_${level}_text_samples.txt\
        --model_path data/deepspeech-0.9.3-models.pbmm\
        --scorer_path data/deepspeech-0.9.3-models.scorer\
        --audio_dir data/output/dccrnet-tuned/Task_2_Level_$level$name

    echo "Predicting on level $level with ir"
    python -m dtu_hsc_solutions data linear-to-dccrnet-tuned\
        --level Task_2_Level_$level\
        --overwrite-output\
        --dir-postfix $name\
        --weights-dir "ml_models/$name/load_dccrnet_model_fold_1_model.pth"
        
    echo "Evaluating on level $level with ir"
    python -m hsc_given_code.evaluate\
        --text_file data/Task_2_Level_$level/Task_2_Level_${level}_text_samples.txt\
        --model_path data/deepspeech-0.9.3-models.pbmm\
        --scorer_path data/deepspeech-0.9.3-models.scorer\
        --audio_dir data/output/linear-to-dccrnet-tuned/Task_2_Level_$level$name
done

