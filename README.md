# CA-NoiseGAN
The implementation of ECCV 2020 paper *"Learning Camera-Aware Noise Models"*

- ***Arxiv:*** [TODO]
- ***Project Website: https://arcchang1236.github.io/CA-NoiseGAN/***
- ***Abstract Video: https://youtu.be/_VWN8oLk68Q*** 
- ***Long Video: https://youtu.be/_gScv9bAdTE*** 

## Highlights
##### Modeling imaging sensor noise is a fundamental problem for image processing and computer vision applications. While most previous works adopt statistical noise models, real-world noise is far more complicated and beyond what these models can describe. To tackle this issue, we propose a data-driven approach, where a generative noise model is learned from real-world noise. The proposed noise model is camera-aware, that is, different noise characteristics of different camera sensors can be learned simultaneously, and a single learned noise model can generate different noise for different camera sensors. Experimental results show that our method quantitatively and qualitatively outperforms existing statistical noise models and learning-based methods.
##### Our main contributions are summarized as follows:
- Propose a learning-based generative model for camera sensor noise.
- Achieve camera awareness by leveraging camera-specific Poisson-Gaussian noise and a camera characteristics encoding network.
- Design a novel feature matching loss for signal-dependent patterns, whichleads to significant improvement of visual quality.
- Outperform state-of-the-art noise modeling methods and improve image de-noising performance.

## Prerequisities
- Pytorch >= 1.1.0 
- [Noise Flow] Tensorflow 1.13.0, tensorflow-probability >= 0.5.0
- Python >= 3.6, Ubuntu 16.04, cuda-10.1

## Quick Start

#### <Step 1> Download test data

You need to download our test data ***[link]*** and unzip them first.
Then, you can change the *data_dir* in **config.yml** into your data path.

#### <Step 2> Download the checkpoints of denoisers and noise models

You should download the checpoints file ***[link]*** and unzip them into the root directory.

#### <Step 3> Test the denoisers and noise models

For denoisers, 
```bash
python test_denoisers.py --config config.yml
```

For noise models,

```bash
python test_noise_models.py --config config.yml
```

You need to check the correctness of each path in config.yml. Moreover, you can modify the amount of samples and patch size. See config.yml for more detail.

#### <Step 4> Visualization

The results will be saved in **samples/** .

## Citation
```
@inproceedings{chang2020canoisegan,
    author    = {Chang, Ke-Chi and Wang, Ren and Lin, Hung-Jin and Liu, Yu-Lun and Chen, Chia-Ping and Chang, Yu-Lin and Chen, Hwann-Tzong},
    title     = {Learning Camera-Aware Noise Models},
    booktitle = {European Conference on Computer Vision},
    year      = {2020}
  }
```

## Contact
If you find any problem, please feel free to contact me. (sky1236@gapp.nthu.edu.tw)