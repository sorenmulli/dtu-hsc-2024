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

## 3. Data and Model Setup
...

## 4. Running the Code
Important commands include
```bash
python -m hsc_given_code.evaluate --help
```
