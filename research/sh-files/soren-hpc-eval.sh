#!/bin/sh
#BSUB -q p1
#BSUB -gpu "num=1::mode=exclusive_process"
#BSUB -R "select[gpu80gb]"

#BSUB -n 4
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"

#BSUB -W 24:00

#BSUB -J "hsc-train"
#BSUB -N
#BSUB -u swiho@dtu.dk
#BSUB -oo /zhome/ac/c/137651/joblogs/stdout_%J.out
#BSUB -eo /zhome/ac/c/137651/joblogs/stderr_%J.out


echo "Starting job on GPU $CUDA_VISIBLE_DEVICES ..."

export HSC=/dtu/p1/swiho/hsc
source /zhome/ac/c/137651/setup-hsc.sh


levels=(4 5 6 7)

for level in "${levels[@]}"; do
    echo "Predicting on level $level"
    name="t1l$level-no-ir"
    python -m dtu_hsc_solutions $HSC dccrnet-tuned\
        --level Task_1_Level_$level\
        --overwrite-output\
        --dir-postfix $name\
        --weights-dir "ml_models/$name/load_dccrnet_model_fold_1_model.pth"
    echo "Evaluating on level $level"
    python -m hsc_given_code.evaluate\
        --text_file $HSC/Task_1_Level_$level/Task_1_Level_${level}_text_samples.txt\
        --model_path $HSC/deepspeech-0.9.3-models.pbmm\
        --scorer_path $HSC/deepspeech-0.9.3-models.scorer\
        --audio_dir $HSC/output/dccrnet-tuned/Task_1_Level_$level$name
done
