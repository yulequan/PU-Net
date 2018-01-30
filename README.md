# PU-Net: Point Cloud Upsampling Network
by [Lequan Yu](http://appsrv.cse.cuhk.edu.hk/~lqyu/), Xianzhi Li, [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

This repository is for '[PU-Net: Point Cloud Upsampling Network](https://arxiv.org/abs/1801.06761)'. The code is modified from [PointNet++](https://github.com/charlesq34/pointnet2) and [PointSetGeneration](https://github.com/fanhqme/PointSetGeneration). 

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under TF1.3 (higher version should also work) and Python 2.7 on Ubuntu 16.04.

For compiling TF operators, please check `tf_xxx_compile.sh` under each ops subfolder in `code/tf_ops` folder. Note that you need to update `nvcc`, `python` and 'tensoflow include library' if necessary. You also need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.


### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/hszhao/PSPNet.git
   ```

2. To train the model:
  First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1te8d1y2BTFBL_3CB1jpqbOFzkkjvtKsE/view?usp=sharing) and put it in folder `h5_data`.
  Then run:
   ```shell
   cd code
   python main.py --phase train
   ```

3. To evaluate the model:
    First, you need to download the pretrained model from [GoogleDrive](https://drive.google.com/file/d/1c1oYNwIzKxCOF_6bqm3HmwYcCZv1230Z/view?usp=sharing), extract it and put it in folder 'model'.
    Then run:
   ```shell
   cd code
   python main.py --phase test --log_dir ../model/generator2_new6
   ```
   You will see the input and output results in the folder `../model/generator2_new6/result`.


## Citation

If PU-Net is useful for your research, please consider citing:
    @article{yu2018pu,
  title={PU-Net: Point Cloud Upsampling Network},
  author={Yu, Lequan and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:1801.06761},
  year={2018}
}

### Questions

Please contact 'lqyu@cse.cuhk.edu.hk'

