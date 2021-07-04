





## Graph Dynamic Autoencoder for Fault Detection

This repository includes our model code and dataset for fault detection.

#### Model

This structure of our model looks just like the following illustration:

<img src="D:\Github\Graph-Dynamic-Autoencoder\img\Structure of GDAE.svg" alt="Structure of GDAE" style="zoom:67%;" />

The experiments were performed on a computer with the following specifications:

- CPU: AMD Ryzen 5 2600x; RAM: 8.0 GB;
- GPU: NVIDIA GeForce GTX 970;
- Operating System: Ubuntu 18.04 LTS 64-bit computing (Python 3.8).

dataset:  Tennessee Eastman process (TEP)

#### Requirements

- torch==1.6.0 (cuda 10.2)
- dgl==0.6.0 (cuda 10.2)
- numpy==1.18.5
- scikit-learn==0.23.1
- scipy==1.5.0

#### Train

```python
python train.py
```

If you want to train your own weights, you can do so with this command.

#### Test

```python
python test.py
```

The params folder has the weights we trained and also includes the weights for the ablation experiments. You can use this command to directly reproduce the results obtained in our paper.

#### Note

If you don't have the GPU version of torch installed, you can install it by visiting [Pytorch](https://pytorch.org/).

It is okay to use the CPU directly, but you will need to make some changes to the code.





