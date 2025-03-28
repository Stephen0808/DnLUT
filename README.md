<p align="center">
  <img src="./DnLUT/DnLUT_logo.png" height=110>
</p>

# [CVPR25] DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables

[paper](https://arxiv.org/pdf/2503.15931)

:star: If DnLUT is helpful to your images or projects, please help star this repo. Thanks! :hugs: 

## TODO
- [x] Add training code and config files
- [ ] Add LUT and inference code

## Installation

```python
git clone https://github.com/Stephen0808/DnLUT.git
pip install -r requirements.txt
```
## Dataset

We build our training and evaluation dataset by following [Restormer](https://github.com/swz30/Restormer/blob/main/Denoising/README.md#training). You could build other datasets with their repo as reference. If you want to download the correponding dataset for our task in huggingface, I will later upload them if you put forward your issue.

### Gaussian Noise
+ Download training (DIV2K, Flickr2K, WED, BSD) and testing datasets, run
```python
python download_data.py --data train-test --noise gaussian
```

+ Generate image patches from full-resolution training images, run
```python
python generate_patches_dfwb.py 
```

### Real Image Denoising
+ Download SIDD training data, run
```python
python download_data.py --data train --noise real
```

+ Generate image patches from full-resolution training images, run

```python 
python generate_patches_sidd.py 
```

## Training

```python
cd /home/styan/DnLUT/dn
CUDA_VISIBLE_DEVICES=4 python 1_train_model_dnlut.py
```
