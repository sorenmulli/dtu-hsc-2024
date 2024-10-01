### Initial finetuning
From a small training of 10 epochs on non-aligned data, these losses where computed:
![img](ml-models\results\losses_per_fold_10epochs.png)

From a small training of 100 epochs on naive-aligned data, these losses where computed. The mean CER was 0.7
![img](ml-models\results\losses_per_fold_100epochs_naivepad.png)

From a small training of 100 epochs on data run through the linear filter first (is the alignment then crosscorrelation or still naive?), these losses where computed. The mean CER was 0.72. Seems the sentences are good, but cut off to soon. Maybe issue with alignment:
![img](ml-models\results\losses_per_fold_100epochs_linfilter_naivepad.png)

### Out of the box DCCRNet
DCCRN out of the box on task_1_level_4 has mean CER 0.86.
DCCRN out of the box on task_1_level_7 has mean CER 0.95.
DCCRN out of the box on task_2_level_2 has mean CER 0.63.
DCCRN out of the box on task_2_level_3 has mean CER 0.69.
DCCRN out of the box on task_3_level_1 has mean CER 0.99.


Just IR on task_2_level_1 has a mean CER of 0.47

### Finetune with aligned data
DCCRN finetuned for 100 epochs on task_2_level_1 with aligned data has mean CER of 0.53
![img](ml-models\results\losses_per_fold_100epochs_aligned.png)



