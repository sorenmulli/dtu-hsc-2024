### Out of the box DCCRNet
DCCRN out of the box on task_1_level_4 has mean CER 0.86.

DCCRN out of the box on task_1_level_7 has mean CER 0.95.

DCCRN out of the box on task_2_level_2 has mean CER 0.63.

DCCRN out of the box on task_2_level_3 has mean CER 0.69.

DCCRN out of the box on task_3_level_1 has mean CER 0.99.


Task_2_level_1: Just IR-reconstructed data from task_2_level_1 has a mean CER of 0.47

Task_2_level_1: Using IR-reconstructions as input to the DCCRNet (out of the box) gets mean CER of 0.41.

### Initial finetuning
From a small training of 10 epochs on non-aligned data, these losses where computed:
![img](img/losses_per_fold_10epochs.png)

From a small training of 100 epochs on naive-aligned data, these losses where computed. The mean CER was 0.7
![img](img/losses_per_fold_100epochs_naivepad.png)

From a small training of 100 epochs on data run through the linear filter first (is the alignment then crosscorrelation or still naive?), these losses where computed. The mean CER was 0.72. Seems the sentences are good, but cut off to soon. Maybe issue with alignment:
![img](img/losses_per_fold_100epochs_linfilter_naivepad.png)

### Finetune with aligned data
DCCRN finetuned for 100 epochs on task_2_level_1 with aligned data (through cross-correlation) has mean CER of 0.53
![img](img/losses_per_fold_100epochs_aligned.png)

DCCRN finetuned for 100 epochs on task_2_level_1 with aligned data (through cross-correlation) run through IR has mean CER of 0.81
![img](img/losses_per_fold_100epochs_aligned.png)

DCCRN finetuned for 100 epochs on task_2_level_1 with aligned + IR data using a Scale-Invariant Signal-to-Distortion-Ratio (SI-SDR) as loss function has mean CER of 0.52

![img](img/losses_per_fold_aligned_ir_sdr.png)

DCCRN finetuned for 100 epochs on task_2_level_1 with aligned + IR data using a Scale-Invariant Signal-to-Distortion-Ratio (SI-SDR) as loss function and only training the decoder of the model has mean CER of 0.5221230

![img](img/losses_per_fold_aligned_ir_sdr_dec.png)

DCCRN finetuned for 100 epochs on task_2_level_1 with aligned + IR data using a spectral convergence loss as loss function has mean CER of 0.12

![img](img/losses_per_fold_aligned_ir_spec.png)

DCCRN finetuned for 100 epochs on task_2_level_1 with aligned + IR data using a combination of Scale-Invariant Signal-to-Distortion-Ratio (SI-SDR) and spectral convergence as loss function has mean CER of 0.52

![img](img/losses_per_fold_aligned_ir_comb.png)