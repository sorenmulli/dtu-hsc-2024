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

python -m dtu_hsc_solutions.linear_filter.compute_filter $HSC

for SOLUTION in "${SOLUTIONS[@]}"
do
    echo "> Starting solution $SOLUTION"
    for LEVEL in "${LEVELS[@]}"
    do
        echo ">> Trying $SOLUTION on level $LEVEL"
        python -m dtu_hsc_solutions $HSC $SOLUTION --level $LEVEL --overwrite-output
        python -m dtu_hsc_solutions $HSC $SOLUTION --level $LEVEL --overwrite-output --test-split
    done
done
