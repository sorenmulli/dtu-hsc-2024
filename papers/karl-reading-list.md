# Karl's paper reading list HSC 2024

## Speech enhancement overview:

[Overview](https://github.com/WenzheLiu-Speech/awesome-speech-enhancement?tab=readme-ov-file)

## DOSE: 

- SE using diffusio   n model to generate denoised text by conditioning on noised text
- Fixes 'condition collapse' by dropout and 'Adaptive prior'
- Compares to other Dropout SE methods:
  - DiffWave
  - DiffuSE
  - CDiffuSE
  - SGMSE
  - DR-DiffuSE

[DOSE](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)
[CODE](https://github.com/ICDM-UESTC/DOSE?tab=readme-ov-file)
[PRETRAINED MODELS](https://github.com/ICDM-UESTC/DOSE/releases)


## DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement

- The words: **Phase aware** (useful, reverb?)
- Shows good [samples](https://huyanxin.github.io/DeepComplexCRN/)
  - Many samples sound like they're simply nonstationary noise laid on top of speech
  - It does have reverb as well though                  
- Uses complex information (impossible to train further?)
- Came #1 in a competition for real-time denoising
- Looks (sounds) quite promising
- Does *not* appear to have a "clean method of running" 
- Does not link directly to data

[DCCRN](https://arxiv.org/abs/2008.00264)
[CODE](https://github.com/huyanxin/DeepComplexCRN?tab=readme-ov-file)

Another, more point-and-click (though not reviewed) implementation appears to be found [here](https://github.com/maggie0830/DCCRN.git)


## A selection of datasets for [noise](https://github.com/Yuan-ManX/ai-audio-datasets.git)
