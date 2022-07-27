# Uni3DA

This repo contains the source code and dataset for our Uni3DA work(under review):

[**Uni3DA: Universal 3D Domain Adaptation for Object Recognition**]()



## Introduction
Traditional 3D point cloud classification task aims to train a classifier in the closed-set world, where training and test data share the same label set and the same data distribution. In this paper, we target a more challenging and realistic setting in 3D point cloud classification task: universal domain adaptation (UniDA), where 1) data distributions for training and test data are
different; and 2) for given label sets of training data and test data, they may contain a shared label set and hold a private label set
respectively, bringing up an additional category discrepancy. To solve UniDA problem, researchers have designed many methods
based on 2D image datasets. However, existing methods based on 2D image datasets cannot be directly applied to the 3D
scenarios, due to the difficulty in capturing discriminative local geometric structures brought by the unordered and irregular 3D
point cloud data. To address UniDA in 3D scenarios, we develop a 3D universal domain adaptation framework, which consists
of three modules: Self-Constructed Geometric (SCG) module, Local-to-Global Hypersphere Reasoning (LGHR) module and
Self-Supervised Boundary Adaptation (SBA) module. SCG and LGHR generate the discriminative representation, which is used
to acquire domain-invariant knowledge for training and test data. SBA is designed to automatically recognize whether a given label is from the shared label set or private label set, and adapts training and test data from the shared label set. To our best
knowledge, this is the first work about UniDA for 3D scenarios. Extensive experiments on public 3D point cloud datasets illustrate
the superiority of our method over existing UniDA methods.

## Dataset

The Uni3DA_data dataset is extracted from three popular 3D object/scene datasets (i.e., [ModelNet](https://modelnet.cs.princeton.edu/), [ShapeNet](https://shapenet.cs.stanford.edu/iccv17/) and [ScanNet](http://www.scan-net.org/)) for cross-domain 3D objects classification.

## Requirements
- Python 3.6
- PyTorch 1.2
- ......

To build env, just run: 

``
pip3 install -r requirements.txt
``

## File Structure
```
.
├── README.md
├── logs                            
├── dataset
│   └──Uni3DA_data                              
├── dataloader.py
├── data_utils.py
├── main.sh
├── mmd.py
├── Model.py
├── model_utils.py
├── train.py            
└── train_gan.py                                   
```

## Data Download
Download the [Uni3DA_data]() and extract it as the dataset fold.

## Train & Test
If you run the experiment on one adaptation scanerio, like modelnet to shapenet,
```
python train_gan.py -source modelnet -target shapenet
```

## Citation

TBD....

## Contact
If you have any questions, please contact [Yu Ren](renyu0414@gmail.com). 
