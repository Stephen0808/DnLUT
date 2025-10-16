<p align="center">
  <img src="./DnLUT/DnLUT_logo.png" height=110>
</p>

# [CVPR25] DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables

[![arXiv](https://img.shields.io/badge/arXiv-2503.15931-b31b1b.svg)](https://arxiv.org/abs/2503.15931)


:construction: This repo is still under construction. Thanks for your waiting!

:star: If DnLUT is helpful to your images or projects, please help star this repo. Thanks! :hugs: 

## TODO
- [x] Add training codes and config files
- [x] Add LUT transferring and inference codes

## Installation

```python
git clone https://github.com/Stephen0808/DnLUT.git
pip install -r requirements.txt
```
## Dataset

We build our training and evaluation dataset by following [Restormer](https://github.com/swz30/Restormer/blob/main/Denoising/README.md#training). You could build other datasets with their repo as reference. If you want to download the correponding dataset for our task in huggingface, I will later upload them if you put forward your issue.

### Gaussian Denoising
+ Download training (DIV2K, Flickr2K, WED, BSD) and testing datasets, run
```python
cd build_data
python download_data.py --data train-test --noise gaussian
```

+ Generate image patches from full-resolution training images, run
```python
python generate_patches_dfwb.py 
```

### Real Image Denoising
+ Download SIDD training data, run
```python
cd build_data
python download_data.py --data train --noise real
```

+ Generate image patches from full-resolution training images, run

```python 
python generate_patches_sidd.py 
```

## Training

```python
cd ~/DnLUT/dn
python 1_train_model_dnlut.py
```

## Transferring
```python
cd ~/DnLUT/dn
python 2_transfer_to_lut_dn.py
```

## Citation
```
@article{yang2025dnlut,
  title={DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables},
  author={Yang, Sidi and Huang, Binxiao and Zhang, Yulun and Yu, Dahai and Yang, Yujiu and Wong, Ngai},
  journal={arXiv preprint arXiv:2503.15931},
  year={2025}
}
```

## Acknowledgement
Our codes are built upon [SRLUT](https://github.com/yhjo09/SR-LUT) and [MuLUT](https://github.com/ddlee-cn/MuLUT). Thanks for their great works.
