# UniUSNet: A Promptable Framework for Universal Ultrasound Disease Prediction and Tissue Segmentation

UniUSNet is a universal framework for ultrasound image classification and segmentation, featuring:

- A novel promptable module for incorporating detailed information into the model's learning process.
- Versatility across various ultrasound natures, anatomical positions, and input types. Proficiency in both segmentation and classification tasks
- Strong generalization capabilities demonstrated through zero-shot and fine-tuning experiments on new datasets.

For more details, see the accompanying paper,

> [**UniUSNet: A Promptable Framework for Universal Ultrasound Disease Prediction and Tissue Segmentation**](https://arxiv.org/abs/2406.01154)<br/>
  Zehui Lin, Zhuoneng Zhang, Xindi Hu, Zhifan Gao, Xin Yang, Yue Sun, Dong Ni, Tao Tan. <b>Arxiv</b>, Jun 3, 2024. https://arxiv.org/abs/2406.01154

## Installation
- Clone this repository.
```
git clone https://github.com/Zehui-Lin/UniUSNet.git
cd UniUSNet
```
- Create a new conda environment.
```
conda create -n UniUSNet python=3.10
conda activate UniUSNet
```
- Install the required packages.
```
pip install -r requirements.txt
```

## Data

- BroadUS-9.7K consists of ten publicly-available datasets, including [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), [BUSIS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9025635/), [UDIAT](https://ieeexplore.ieee.org/abstract/document/8003418), [BUS-BRA](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16812), [Fatty-Liver](https://link.springer.com/article/10.1007/s11548-018-1843-2#Sec8), [kidneyUS](http://rsingla.ca/kidneyUS/), [DDTI](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images/data), [Fetal HC](https://hc18.grand-challenge.org/), [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html) and [Appendix](https://zenodo.org/records/7669442).
- You can prepare the data by downloading the datasets and organizing them as follows:

```
data
├── classification
│   └── UDIAT
│       ├── 0
│       │   ├── 000001.png
│       │   ├── ...
│       ├── 1
│       │   ├── 000100.png
│       │   ├── ...
│       ├── config.yaml
│       ├── test.txt
│       ├── train.txt
│       └── val.txt
│   └── ...
└── segmentation
    └── BUSIS
        ├── config.yaml
        ├── imgs
        │   ├── 000001.png
        │   ├── ...
        ├── masks
        │   ├── 000001.png
        │   ├── ...
        ├── test.txt
        ├── train.txt
        └── val.txt
    └── ...
```
- Please refer to the `data_demo` folder for examples.

## Training
We use `torch.distributed` for multi-GPU training (also supports single GPU training). To train the model, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 omni_train.py --output_dir exp_out/trial_1 --prompt
```

## Testing
To test the model, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 omni_test.py --output_dir exp_out/trial_1 --prompt
```


## Checkpoints
- You can download the pre-trained checkpoints from [BaiduYun](https://pan.baidu.com/s/1uciwM5K4wRiMWnrAsB4qMQ?pwd=x390).

## Citation
If you find this work useful, please consider citing:

```
@article{lin2024uniusnet,
  title={UniUSNet: A Promptable Framework for Universal Ultrasound Disease Prediction and Tissue Segmentation},
  author={Lin, Zehui and Zhang, Zhuoneng and Hu, Xindi and Gao, Zhifan and Yang, Xin and Sun, Yue and Ni, Dong and Tan, Tao},
  journal={arXiv preprint arXiv:2406.01154},
  year={2024}
}
```

## Acknowledgements
This repository is based on the [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) repository. We thank the authors for their contributions.