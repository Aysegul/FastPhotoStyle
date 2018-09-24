[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)

## Domain Stylization

PyTorch implementation of synthetic to real domain adaptation with stylization algorithm.
[Domain Stylization: A Strong, Simple Baseline for Synthetic to Real Image Domain Adaptation](https://arxiv.org/abs/1807.09384) <br>
Aysegul Dundar (NVIDIA), [Ming-Yu Liu (NVIDIA)](http://mingyuliu.net/), [Ting-Chun Wang (NVIDIA)](https://tcwang0509.github.io/), John Zedlewski (NVIDIA), [Jan Kautz (NVIDIA)](http://jankautz.com/) <br>In arXiv, 2018 <br>


### Algorithm

Domain Stylization is an effective approach of using the existing FastPhotoStyle algorithm to generate stylized synthetic image datasets.
To create the stylized synthetic dataset, for every image–label pair in source domain, we randomly sample N image–label pairs in target domain to generate N stylized synthetic images.


The algorithm iterates between two steps: stylization and semantic segmentation learning.
First, we generate a stylized synthetic dataset without using segmentation masks.
We then train a semantic segmentation network with the stylized images.
The segmentation network predicts semantic labels for the real image dataset.
The synthetic dataset is then stylized using the real image dataset with the estimated segmentation masks for creating a new stylized synthetic dataset.


<img src="./domain_stylization.gif" width="800" title="GIF">




### Examples


1. First iteration stylize with no label map

```
python demo_domain_stylization.py --content_list PATH-TO-YOUR-CONTENT \
--out_path PATH-TO-YOUR-OUTPUT
```

2. Train a segmentation network, and dump the prediction from the training set.
The training code for DRN26 that is used in the paper can be found [here](https://github.com/fyu/drn).

3. Second iteration stylize with the estimated label maps

```
python demo_domain_stylization.py --content_list PATH-TO-YOUR-CONTENT \
--style_seg_path PATH-TO-YOUR-SEG-MASKS \
--content_seg_path PATH-TO-YOUR-CONTENT-MASKS --out_path PATH-TO-YOUR-OUTPUT
```
