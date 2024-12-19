# Disentangling, Amplifying, and Debiasing: Learning Disentangled Representations for Fair Graph Neural Networks

This repository provides the official PyTorch implementation of **DAB-GNN**, as introduced in the following paper:

> **Disentangling, Amplifying, and Debiasing: Learning Disentangled Representations for Fair Graph Neural Networks**
> [Yeon-Chang Lee](mailto:yeonchang@unist.ac.kr), [Hojung Shin](mailto:hojungshin@hanyang.ac.kr), [Sang-Wook Kim](mailto:wook@hanyang.ac.kr)
> *AAAI 2025 (full paper)*
> [[arXiv Preprint](https://arxiv.org/abs/2408.12875)]

## Requirements

- Python 3.7
- Windows 10
- Torch 1.7.0
- CUDA 11.0
- torch_geometric 1.6.0
  - torch_scatter 2.0.5
  - torch_sparse 0.6.8
- GPU: NVIDIA RTX 3060 12GB
- Additional dependencies listed in `dependencies/requirements.txt`

## Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Create and Activate Conda Environment

```bash
conda create -n <env_name> python=3.7
conda activate <env_name>

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install torch_geometric==1.6.0

pip install ./dependencies/torch_1.7.0/torch_scatter-2.0.5-cp37-cp37m-win_amd64.whl
pip install ./dependencies/torch_1.7.0/torch_sparse-0.6.8-cp37-cp37m-win_amd64.whl

pip install -r ./dependencies/requirements.txt

pip install wandb
```

## Usage

You can run the model with:

```bash
python basic_model.py --dataset=<dataset>
```

### Dataset Download

You can download the **bail**, **credit**, **nba**, **pokec_z**, and **pokec_n** datasets from:

- **bail**, **credit**: [NIFTY GitHub](https://github.com/chirag-agarwall/nifty/tree/main/dataset)
- **nba**, **pokec_z**, **pokec_n**: [FairGNN GitHub](https://github.com/EnyanDai/FairGNN/tree/main/dataset)

### Hyperparameter Configuration

Create a YAML config file named `<dataset>.yml` in the `./config` directory to configure hyperparameters. You can refer to the example file `bail.yml` for guidance.

## Example Results

### DAB-GNN

```bash
python basic_model.py --dataset=bail
```

```bash
Loading config from configs/bail.yml
Epoch 0: disentanglement_loss_train: 0.0000
Epoch 0: disentanglement_loss_val: 0.0000
Epoch 0: structural_loss_train: 0.6931 | attribute_loss_train: 0.7419 | structural_roc_train: 0.5504 | attribute_roc_train: 0.5368
Epoch 0: structural_loss_val: 0.6933 | attribute_loss_val: 0.7659 | structural_roc_val: 0.5749 | attribute_roc_val: 0.5859
Epoch 100: disentanglement_loss_train: 0.0000
Epoch 100: disentanglement_loss_val: 0.0000
Epoch 100: structural_loss_train: 0.6193 | attribute_loss_train: 0.4865 | structural_roc_train: 0.9940 | attribute_roc_train: 0.8572
Epoch 100: structural_loss_val: 0.6648 | attribute_loss_val: 0.5694 | structural_roc_val: 0.7618 | attribute_roc_val: 0.7620
Epoch 200: disentanglement_loss_train: 0.0000
Epoch 200: disentanglement_loss_val: 0.0000
Epoch 200: structural_loss_train: 0.1187 | attribute_loss_train: 0.2692 | structural_roc_train: 1.0000 | attribute_roc_train: 0.9800
Epoch 200: structural_loss_val: 0.6685 | attribute_loss_val: 0.3905 | structural_roc_val: 0.7755 | attribute_roc_val: 0.9025
Epoch 300: disentanglement_loss_train: 0.0001
Epoch 300: disentanglement_loss_val: 0.0000
Epoch 300: structural_loss_train: 0.0183 | attribute_loss_train: 0.1338 | structural_roc_train: 1.0000 | attribute_roc_train: 0.9956
Epoch 300: structural_loss_val: 0.9839 | attribute_loss_val: 0.3632 | structural_roc_val: 0.7805 | attribute_roc_val: 0.9302
Epoch 400: disentanglement_loss_train: 0.0001
Epoch 400: disentanglement_loss_val: 0.0001
Epoch 400: structural_loss_train: 0.0067 | attribute_loss_train: 0.0715 | structural_roc_train: 1.0000 | attribute_roc_train: 1.0000
Epoch 400: structural_loss_val: 1.1229 | attribute_loss_val: 0.4111 | structural_roc_val: 0.7818 | attribute_roc_val: 0.9416
Epoch 500: disentanglement_loss_train: 0.0001
Epoch 500: disentanglement_loss_val: 0.0001
Epoch 500: structural_loss_train: 0.0034 | attribute_loss_train: 0.0395 | structural_roc_train: 1.0000 | attribute_roc_train: 1.0000
Epoch 500: structural_loss_val: 1.1895 | attribute_loss_val: 0.5079 | structural_roc_val: 0.7825 | attribute_roc_val: 0.9440
Epoch 600: disentanglement_loss_train: 0.0002
Epoch 600: disentanglement_loss_val: 0.0002
Epoch 600: structural_loss_train: 0.0022 | attribute_loss_train: 0.0225 | structural_roc_train: 1.0000 | attribute_roc_train: 1.0000
Epoch 600: structural_loss_val: 1.2242 | attribute_loss_val: 0.6365 | structural_roc_val: 0.7829 | attribute_roc_val: 0.9419
Epoch 700: disentanglement_loss_train: 0.0002
Epoch 700: disentanglement_loss_val: 0.0002
Epoch 700: structural_loss_train: 0.0014 | attribute_loss_train: 0.0135 | structural_roc_train: 1.0000 | attribute_roc_train: 1.0000
Epoch 700: structural_loss_val: 1.2419 | attribute_loss_val: 0.7743 | structural_roc_val: 0.7830 | attribute_roc_val: 0.9396
Epoch 800: disentanglement_loss_train: 0.0003
Epoch 800: disentanglement_loss_val: 0.0002
Epoch 800: structural_loss_train: 0.0011 | attribute_loss_train: 0.0088 | structural_roc_train: 1.0000 | attribute_roc_train: 1.0000
Epoch 800: structural_loss_val: 1.2494 | attribute_loss_val: 0.9012 | structural_roc_val: 0.7832 | attribute_roc_val: 0.9376
Epoch 900: disentanglement_loss_train: 0.0003
Epoch 900: disentanglement_loss_val: 0.0003
Epoch 900: structural_loss_train: 0.0008 | attribute_loss_train: 0.0063 | structural_roc_train: 1.0000 | attribute_roc_train: 1.0000
Epoch 900: structural_loss_val: 1.2508 | attribute_loss_val: 1.0186 | structural_roc_val: 0.7832 | attribute_roc_val: 0.9362
Epoch 1000: disentanglement_loss_train: 0.0003
Epoch 1000: disentanglement_loss_val: 0.0003
Epoch 1000: structural_loss_train: 0.0006 | attribute_loss_train: 0.0044 | structural_roc_train: 1.0000 | attribute_roc_train: 1.0000
Epoch 1000: structural_loss_val: 1.2487 | attribute_loss_val: 1.1257 | structural_roc_val: 0.7830 | attribute_roc_val: 0.9352
==========structural==========
The AUCROC of estimator: 0.7906
Parity: 0.07446610812847049 | Equality: 0.050270723709296083
F1-score: 0.675451439726696
acc: 0.7181606272515363
best loss: 0.5908668637275696
best epoch: 162
==========attribute==========
The AUCROC of estimator: 0.9425
Parity: 0.023855114118354115 | Equality: 0.02720201326927474
F1-score: 0.888145896656535
acc: 0.9220173765628311
best loss: 0.5436404347419739
best epoch: 530
attribute_best_embedding tensor([[ 0.5164,  0.7160, -0.0107,  ..., -0.1818, -0.0984, -1.2288],
        [ 0.6523,  0.6456,  0.0538,  ..., -0.2951, -0.2874, -0.9294],
        [-0.1445,  2.5998, -0.7883,  ...,  0.5370,  1.2287, -1.9153],
        ...,
        [ 3.5530, -6.9330,  3.9255,  ..., -2.1567, -6.9366,  5.0597],
        [-1.0065,  2.0721, -0.9699,  ...,  0.5313,  2.3818, -2.0274],
        [ 0.4665, -0.1582, -1.1230,  ...,  0.1956, -0.4106, -0.7524]],
       device='cuda:1', grad_fn=<AddBackward0>)
structural_best_embedding tensor([[ 0.0321,  0.0291, -0.0355,  ..., -0.0339, -0.0201,  0.0517],
        [-0.0420,  0.0530, -0.3459,  ..., -0.3186,  0.0798,  0.4339],
        [-0.0231,  0.0451, -0.2423,  ..., -0.2265,  0.0541,  0.3080],
        ...,
        [ 0.0662,  0.0246,  0.0507,  ...,  0.0537, -0.0584, -0.0699],
        [-0.0167,  0.0439, -0.2475,  ..., -0.2269,  0.0451,  0.3147],
        [-0.0347,  0.0481, -0.3116,  ..., -0.2831,  0.0732,  0.3935]],
       device='cuda:1', grad_fn=<AddBackward0>)
[Train] Epoch 0:train_loss: 1.0527 | train_auc_roc: 0.0420
[val] Epoch 0:val_loss: 1.1946 | val_auc_roc: 0.0914
[Train] Epoch 100:train_loss: 0.1036 | train_auc_roc: 1.0000
[val] Epoch 100:val_loss: 0.4073 | val_auc_roc: 0.8910
[Train] Epoch 200:train_loss: 0.0341 | train_auc_roc: 1.0000
[val] Epoch 200:val_loss: 0.5082 | val_auc_roc: 0.8807
[Train] Epoch 300:train_loss: 0.0136 | train_auc_roc: 1.0000
[val] Epoch 300:val_loss: 0.6020 | val_auc_roc: 0.8778
[Train] Epoch 400:train_loss: 0.0061 | train_auc_roc: 1.0000
[val] Epoch 400:val_loss: 0.6865 | val_auc_roc: 0.8744
[Train] Epoch 500:train_loss: 0.0021 | train_auc_roc: 1.0000
[val] Epoch 500:val_loss: 0.7586 | val_auc_roc: 0.8721
[Train] Epoch 600:train_loss: -0.0002 | train_auc_roc: 1.0000
[val] Epoch 600:val_loss: 0.8194 | val_auc_roc: 0.8708
[Train] Epoch 700:train_loss: -0.0013 | train_auc_roc: 1.0000
[val] Epoch 700:val_loss: 0.8726 | val_auc_roc: 0.8700
[Train] Epoch 800:train_loss: -0.0024 | train_auc_roc: 1.0000
[val] Epoch 800:val_loss: 0.9219 | val_auc_roc: 0.8693
[Train] Epoch 900:train_loss: -0.0036 | train_auc_roc: 1.0000
[val] Epoch 900:val_loss: 0.9635 | val_auc_roc: 0.8690
[Train] Epoch 1000:train_loss: -0.0042 | train_auc_roc: 1.0000
[val] Epoch 1000:val_loss: 0.9949 | val_auc_roc: 0.8693
==========non_linear==========
The AUCROC of estimator: 0.9063
Parity: 0.0454376022636756 | Equality: 0.04086784107374364
F1-score: 0.8161975875933372
acc: 0.864378046196228
best loss: 0.3886524736881256
best epoch: 70
non_linear_best_embedding tensor([[-0.2934, -0.2937,  0.0462,  ...,  0.2341,  0.6235,  1.1771],
        [ 0.3325, -0.6986,  0.4676,  ...,  0.6137,  0.7085,  0.4955],
        [ 1.2068, -2.6448,  1.7221,  ..., -0.3105, -0.4625,  1.6073],
        ...,
        [-3.6192,  7.1882, -3.9762,  ...,  4.3250,  6.9951, -2.4747],
        [ 1.5286, -2.1160,  2.7418,  ..., -0.3044, -1.5535,  1.7127],
        [ 1.5361,  0.1102,  1.8410,  ...,  0.0875,  1.0526,  0.3589]],
       device='cuda:1', grad_fn=<SubBackward0>)
[Train] Epoch 0:train_loss: 0.8155 | train_auc_roc: 0.0364
[val] Epoch 0:val_loss: 0.8043 | val_auc_roc: 0.0861
[Train] Epoch 100:train_loss: 0.3692 | train_auc_roc: 1.0000
[val] Epoch 100:val_loss: 0.4571 | val_auc_roc: 0.9307
[Train] Epoch 200:train_loss: 0.1638 | train_auc_roc: 1.0000
[val] Epoch 200:val_loss: 0.3358 | val_auc_roc: 0.9341
[Train] Epoch 300:train_loss: 0.0829 | train_auc_roc: 1.0000
[val] Epoch 300:val_loss: 0.3415 | val_auc_roc: 0.9336
[Train] Epoch 400:train_loss: 0.0465 | train_auc_roc: 1.0000
[val] Epoch 400:val_loss: 0.3784 | val_auc_roc: 0.9330
[Train] Epoch 500:train_loss: 0.0286 | train_auc_roc: 1.0000
[val] Epoch 500:val_loss: 0.4220 | val_auc_roc: 0.9325
[Train] Epoch 600:train_loss: 0.0190 | train_auc_roc: 1.0000
[val] Epoch 600:val_loss: 0.4652 | val_auc_roc: 0.9322
[Train] Epoch 700:train_loss: 0.0134 | train_auc_roc: 1.0000
[val] Epoch 700:val_loss: 0.5061 | val_auc_roc: 0.9320
[Train] Epoch 800:train_loss: 0.0098 | train_auc_roc: 1.0000
[val] Epoch 800:val_loss: 0.5442 | val_auc_roc: 0.9319
[Train] Epoch 900:train_loss: 0.0075 | train_auc_roc: 1.0000
[val] Epoch 900:val_loss: 0.5797 | val_auc_roc: 0.9319
[Train] Epoch 1000:train_loss: 0.0058 | train_auc_roc: 1.0000
[val] Epoch 1000:val_loss: 0.6129 | val_auc_roc: 0.9318
==========classifier==========
The AUCROC of estimator: 0.9295
Parity: 0.0002623988865660398 | Equality: 0.01731869137497133
F1-score: 0.8634598147594861
acc: 0.9031574486119941
best loss: 0.48205164074897766
best epoch: 90
```

### Vanilla GCN


```bash
python basic_model.py --model=vanilla --no-wd_loss
```

```bash
Loading config from configs/bail.yml
[Train] Epoch 0:train_loss: 0.7551 | train_auc_roc: 0.5456
[val] Epoch 0:val_loss: 0.6616 | val_auc_roc: 0.5804
[Train] Epoch 100:train_loss: 0.4339 | train_auc_roc: 0.9240
[val] Epoch 100:val_loss: 0.5152 | val_auc_roc: 0.8389
[Train] Epoch 200:train_loss: 0.2967 | train_auc_roc: 0.9476
[val] Epoch 200:val_loss: 0.4531 | val_auc_roc: 0.8652
[Train] Epoch 300:train_loss: 0.2398 | train_auc_roc: 0.9556
[val] Epoch 300:val_loss: 0.4578 | val_auc_roc: 0.8724
[Train] Epoch 400:train_loss: 0.2110 | train_auc_roc: 0.9612
[val] Epoch 400:val_loss: 0.4809 | val_auc_roc: 0.8745
[Train] Epoch 500:train_loss: 0.1937 | train_auc_roc: 0.9652
[val] Epoch 500:val_loss: 0.5162 | val_auc_roc: 0.8743
[Train] Epoch 600:train_loss: 0.1827 | train_auc_roc: 0.9704
[val] Epoch 600:val_loss: 0.5560 | val_auc_roc: 0.8744
[Train] Epoch 700:train_loss: 0.1758 | train_auc_roc: 0.9716
[val] Epoch 700:val_loss: 0.5905 | val_auc_roc: 0.8758
[Train] Epoch 800:train_loss: 0.1712 | train_auc_roc: 0.9740
[val] Epoch 800:val_loss: 0.6154 | val_auc_roc: 0.8785
[Train] Epoch 900:train_loss: 0.1679 | train_auc_roc: 0.9752
[val] Epoch 900:val_loss: 0.6342 | val_auc_roc: 0.8821
[Train] Epoch 1000:train_loss: 0.1654 | train_auc_roc: 0.9752
[val] Epoch 1000:val_loss: 0.6468 | val_auc_roc: 0.8855
==========vanilla==========
The AUCROC of estimator: 0.8724
Parity: 0.08000344353607303 | Equality: 0.05177304964539009
F1-score: 0.7927461139896373
acc: 0.8474253019707565
best loss: 0.4503420293331146
best epoch: 232
```

## Cite

```
@article{DBLP:journals/corr/abs-2408-12875,
  author       = {Yeon{-}Chang Lee and
                  Hojung Shin and
                  Sang{-}Wook Kim},
  title        = {Disentangling, Amplifying, and Debiasing: Learning Disentangled Representations
                  for Fair Graph Neural Networks},
  journal      = {CoRR},
  volume       = {abs/2408.12875},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2408.12875},
  doi          = {10.48550/ARXIV.2408.12875},
  eprinttype    = {arXiv},
  eprint       = {2408.12875},
  timestamp    = {Sat, 28 Sep 2024 18:01:42 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2408-12875.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```