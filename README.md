# [ECCV'20] CA-NoiseGAN

![Python version support](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch version support](https://img.shields.io/badge/pytorch-1.1.0-red.svg)

***(News!)*** *We will release all test data and results before Aug.23, please watch our repository for newest update*  

**[Project](https://arcchang1236.github.io/CA-NoiseGAN/)** | **Paper** | **[Abstract](https://youtu.be/_VWN8oLk68Q)** | **[Long Video](https://youtu.be/_gScv9bAdTE)**

## Overview

***CA-NoiseGAN*** is a **PyTorch** implementation of  
*"Learning Camera-Aware Noise Models"*,  
[Ke-Chi Chang](http://arcchang1236.github.io/), Ren Wang, Hung-Jin Lin, [Yu-Lun Liu](http://www.cmlab.csie.ntu.edu.tw/~yulunliu/), Chia-Ping Chen, Yu-Lin Chang, [Hwann-Tzong Chen](https://htchen.github.io/)  
in **European Conference on Computer Vision (ECCV) 2020** conference.

<img src='imgs/archi.jpg' width='95%' />

**Modeling imaging sensor noise** is a fundamental problem for image processing and computer vision applications. While most previous works adopt statistical noise models, real-world noise is far more complicated and beyond what these models can describe. To tackle this issue, we propose a data-driven approach, where **a generative noise model is learned from real-world noise**. The proposed noise model is camera-aware, that is, **different noise characteristics of different camera sensors can be learned** simultaneously, and a **single learned noise model can generate different noise for different camera sensors**. Experimental results show that our method quantitatively and qualitatively outperforms existing statistical noise models and learning-based methods.

<img src='imgs/results.jpg' width='100%' />


## Requirements
This test code is implemented under **Python3**.  
Following libraries are required:

- [PyTorch](https://pytorch.org/) >= 1.1
- [scipy](https://www.scipy.org/)
- [scikit-image](https://scikit-image.org/)

If you want to visualize the results of Noise Flow, the libraries are also required:

- [TensorFlow](https://www.tensorflow.org/) = 1.13.0
- [tensorflow-probability](https://pypi.org/project/tensorflow-probability/) >= 0.5.0



## Usage

1. **Prepare Data**  
   We prepare our test data in *[Google Drive]* and they are totally derived from SIDD dataset.  
   Then, you can change the `data_dir` in `config.yml` into your data path.  

2. **Download Pretrained Models**  
   We provide pretrained baseline models of noise models and denoisers in *[Google Dirve]*.  
   Please unzip them under the root directory.

3. **Prepare Runtime Environment**  
   ```shell
   pip install -r requirements.txt
   ```
4. **Test the Noise Models and Denoisers**  
   You need to check the correctness of each path in `config.yml`.  
   Moreover, you can modify the amount of samples and patch size. See config.yml for more detail.
	- **Noise Models**
	  ```shell
	  python test_noise_models.py --config config.yml
	  ```

    - **Denoisers**
	  ```shell
	  python test_denoisers.py --config config.yml
	  ```

5. **Visual Results**  
   The results will be saved in `./samples/`.

## Resources
- [SIDD Dataset](https://www.eecs.yorku.ca/~kamel/sidd/)
- [Noise Flow](https://github.com/BorealisAI/noise_flow) (Tensorflow)
- [Simple Camera Pipeline](https://github.com/AbdoKamel/simple-camera-pipeline) (Python, MATLAB)


## Citation
```
@inproceedings{chang2020canoisegan,
    author    = {Chang, Ke-Chi and Wang, Ren and Lin, Hung-Jin and Liu, Yu-Lun and Chen, Chia-Ping and Chang, Yu-Lin and Chen, Hwann-Tzong},
    title     = {Learning Camera-Aware Noise Models},
    booktitle = {European Conference on Computer Vision},
    year      = {2020}
  }
```

## Acknowledgement
- [Mediatek Inc.](https://www.mediatek.tw/)