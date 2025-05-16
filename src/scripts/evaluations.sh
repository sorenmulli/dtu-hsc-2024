HSC=$1

echo "Running in dir $HSC"

LEVELS=(
    Task_1_Level_1
    Task_1_Level_2
    Task_1_Level_3
    Task_1_Level_4
    Task_1_Level_5
    Task_1_Level_6
    Task_1_Level_7
    Task_2_Level_1
    Task_2_Level_2
    Task_2_Level_3
)

SOLUTIONS=(
    spectral-subtract
    linear-filter
    reg-linear-filter
    voicefixer
    linear-to-voicefixer
    dccrnet-tuned
    linear-to-dccrnet-tuned
    noop
)
for SOLUTION in "${SOLUTIONS[@]}"
do
    echo "> Starting solution $SOLUTION"
    for LEVEL in "${LEVELS[@]}"
    do
        echo ">> Evaluating $SOLUTION on level $LEVEL"
        python -m hsc_given_code.evaluate \
            --text_file $HSC/Test/$LEVEL/*.txt \
            --model_path $HSC/deepspeech-0.9.3-models.pbmm \
            --scorer_path $HSC/deepspeech-0.9.3-models.scorer \
            --audio_dir $HSC/output/$SOLUTION/Test_$LEVEL \
            --output_csv $HSC/output/$SOLUTION/Test_$LEVEL/results.csv

        python -m hsc_given_code.evaluate \
            --text_file $HSC/$LEVEL/${LEVEL}_text_samples.txt \
            --model_path $HSC/deepspeech-0.9.3-models.pbmm \
            --scorer_path $HSC/deepspeech-0.9.3-models.scorer \
            --audio_dir $HSC/output/$SOLUTION/$LEVEL \
            --output_csv $HSC/output/$SOLUTION/$LEVEL/results.csv

    done
done
