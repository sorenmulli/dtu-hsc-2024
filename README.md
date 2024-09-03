# DTU: HSC 2024

Notes and code for the Helsinki Speech Challenge 2024 work on the Technical University of Denmark.

## 1. Resources

- Challenge description: [arxiv.org/abs/2406.04123](https://arxiv.org/abs/2406.04123)
- Challenge website: [blogs.helsinki.fi/helsinki-speech-challenge](https://blogs.helsinki.fi/helsinki-speech-challenge/)
- Challenge data: [zenodo.org/records/11380835](https://zenodo.org/records/11380835)

## 2. Software Setup
Python 3.9 is used because of challenge dependencies (see 5.3 in the challenge description).

The code is divided into different modules which can all be installed by running
```bash
pip install -e .
```
This will also install requirements set by challenge organizers.


The modules are
- `hsc_given_code`: Evaluation code provided for the challenge.
- `dtu_hsc_data` : Data loading, visualization and preprocessing code.
- `dtu_hsc_solutions` : Implementations of the solutions to the challenge.

## 3. Data and Model Setup

1. Select Download all in Zenodo to download all data, then unzip parent and child directories to some directory.

2. Download DeepSpeech model files (for evaluation) in same data path.
```bash
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```


## 4. Running the Code
You can check the evaluation on the original audio running something like:
```bash
# Handy short-hand to the top-level path of the challenge data and models
export HSC=~/Downloads/hsc
python -m hsc_given_code.evaluate \
    --text_file $HSC/Task_1_Level_1/Task_1_Level_1_text_samples.txt \
    --model_path $HSC/deepspeech-0.9.3-models.pbmm \
    --scorer_path $HSC/deepspeech-0.9.3-models.scorer \
    --audio_dir $HSC/Task_1_Level_1/Recorded
```
Here, I see an average CER of 4.3%.

and you can try an example solution (Wiener filtering):
```bash
# Runs the speech enhancement "solution"
python -m dtu_hsc_solutions $HSC wiener
# and then evaluate
python -m hsc_given_code.evaluate \
    --text_file $HSC/Task_1_Level_1/Task_1_Level_1_text_samples.txt \
    --model_path $HSC/deepspeech-0.9.3-models.pbmm \
    --scorer_path $HSC/deepspeech-0.9.3-models.scorer \
    --audio_dir $HSC/output/wiener/Task_1_Level_1
```
It displays mean CER = 10.3%: even worse than simply running the original audio - there is work to do!
