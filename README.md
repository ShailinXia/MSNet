# MSNet: Multi-Scale Network for Object Detection in Remote Sensing Images

[![Paper](https://img.shields.io/badge/Paper-PR-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0031320323009020)  [![Project](https://img.shields.io/badge/Project-GitHub-gold)](https://github.com/ShailinXia/MSNet)

## Abstract

Remote sensing object detection (RSOD) encounters challenges in effectively extracting features of small objects in remote sensing images (RSIs). To alleviate these problems, we proposed a Multi-Scale Network for Object Detection in Remote Sensing Images (MSNet) with multi-dimension feature information. Firstly, we design a Partial and Pointwise Convolution Extraction Module (P$^2$CEM) to capture feature of object in spatial and channel dimension simultaneously. Secondly, we design a Local and Global Information Fusion Module (LGIFM), designed local information stack and context modeling module to capture texture information and semantic information within the multi-scale feature maps respectively. Moreover, the LGIFM enhances the ability of representing features for small objects and objects within complex backgrounds by allocating weights between local and global information. Finally, we introduce Local and Global Information Fusion Pyramid (LGIFP). With the aid of the LGIFM, the LGIFP enhances the feature representation of small object information, which contributes to dense connection across the multi-scale feature maps. Extensive experiments validate that our proposed method outperforms state-of-the-art performance. Specifically, MSNet achieves mean average precision (mAP) scores of 75.3\%, 93.39\%, 96.00\%, and 95.62\% on the DIOR, HRRSD, NWPU VHR-10, and RSOD datasets, respectively.

<div style="text-align: center;">
	<img src="image/structure.png" alt="structure" width="800">
</div>

## Dependencies
```python
torch
torchvision
tensorboard
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv-python==4.1.2.30
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0
```

## Train and test
MSNet is trained on NWPU, RSOD, DIOR, HRRSD datasets, prepare them for training and testing. You can also use your datasets, meanwhile change the path to yours.

After that, you can train and test the MSNet by

```shell
cd SourceFile

python train.py [--parameters]
```

## Results
If you want to see the results of the model, you can run the following command:

```shell
cd SourceFile

python get_map.py [--parameters]
```

## Quick test
pre-trained weights can be found at SourceFile/logs, meanwhile change the path to yours. you can test the MSNet by

```shell
cd SourceFile

python predict.py
```

## Citation
If you find this project useful in your research, please consider citing:

```

```

## Contact us
Please contact us if there are any questions or suggestions (shailinxia666@gmail.com).