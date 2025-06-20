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
# Comment in for task 3
#    Task_3_Level_1
#    Task_3_Level_2
)

SOLUTIONS=(
    spectral-subtract
    linear-filter
    reg-linear-filter
    voicefixer
    linear-to-voicefixer
    noop
# Comment in for task 3
#    combined-linear
#    combined-linear-to-voicefixer
)

python -m dtu_hsc_solutions.linear_filter.compute_filter $HSC

for SOLUTION in "${SOLUTIONS[@]}"
do
    echo "> Starting solution $SOLUTION"
    for LEVEL in "${LEVELS[@]}"
    do
        echo ">> Trying $SOLUTION on level $LEVEL"
        LEVEL_CSV="$HSC/output/$SOLUTION/$LEVEL/results.csv"
        if [ -f "$LEVEL_CSV" ]; then
            echo ">>> Skipping Level evaluation - results already exist at $LEVEL_CSV"
        else
            python -m dtu_hsc_solutions $HSC $SOLUTION --level $LEVEL --overwrite-output
        fi

        TEST_CSV="$HSC/output/$SOLUTION/Test_$LEVEL/results.csv"
        if [ -f "$TEST_CSV" ]; then
            echo ">>> Skipping Test evaluation - results already exist at $TEST_CSV"
        else
            python -m dtu_hsc_solutions $HSC $SOLUTION --level $LEVEL --overwrite-output --test-split
        fi
    done
done
