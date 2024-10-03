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
    python -m dtu_hsc_solutions.ml_models.huggingface_model_training\
        --data-path $HSC\
        --model dccrnet\
        --task 1\
        --level $level\
        --k-folds 1\
        --epochs 30\
        --loss spec\
        --name t1l$level-no-ir
done
