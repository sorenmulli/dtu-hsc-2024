# DTU: HSC 2024

Notes and code for the Helsinki Speech Challenge 2024 work on the Technical University of Denmark.

## 0. Authors

The reposistory was created as a collaboration between the following people of the technical university of Denmark. Ordering is random and not indicative of contribution

- Marie Juhl Jørgensen
- Christian Deding
- Søren Vejlgaard Holm
- Karl Meisner-Jensen
- Yue Chang
- Asger Dyregaard
- Kim Knudsen
- Martin Carsten Nielsen

## 1. Description of Methods

...

## 2. Running the submission
 1. Install the package `dtu_hsc_solutions` from this repository with `pip install -e .` from top-level. 
    This installs many dependencies as well.

 2. Have a local directory called `models` containing downloaded model weights. The script assumes 
    that this is placed in in the repo root path but you can also customize it with the `--models-path` argument.
    This can be downloaded from the MISSING LINK.

 3. Run e.g. `python main.py path/to/input/files path/to/output/files T1L3`

## 3. Resources

- Challenge description: [arxiv.org/abs/2406.04123](https://arxiv.org/abs/2406.04123)
- Challenge website: [blogs.helsinki.fi/helsinki-speech-challenge](https://blogs.helsinki.fi/helsinki-speech-challenge/)
- Challenge data: [zenodo.org/records/11380835](https://zenodo.org/records/11380835)

## 4. Software Setup
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

## 5. Data and Model Setup

1. Select Download all in Zenodo to download all data, then unzip parent and child directories to some directory.

2. Download DeepSpeech model files (for evaluation) in same data path.
```bash
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

<!--
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
Or in a powershell terminal:
```python -m hsc_given_code.evaluate --text_file data/Task_2_Level_1/Task_2_Level_1/Task_2_Level_1_text_samples.txt --model_path data/deepspeech-0.9.3-models.pbmm --scorer_path data/deepspeech-0.9.3-models.scorer --audio_dir data/Task_2_Level_1/Task_2_Level_1/Recorded```

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
-->
