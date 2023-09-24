<br />
<p align="center">
  <h1 align="center">Explore In-Context Learning for 3D Point Cloud Understanding</h1>
  <p align="center">
    NeurIPS, 2023
    <br />
    <a href="https://github.com/fanglaosi/"><strong>Zhongbin Fang</strong></a>
    ·
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    ·
    <a href="https://xialipku.github.io/"><strong>Xia Li</strong></a>
    <br />
    <a href="ml.inf.ethz.ch/"><strong>Joachim M. Buhmann</strong></a>
    .
    <a href="https://www.mmlab-ntu.com/person/ccloy/"><strong>Chen Change Loy</strong></a>
    .
    <a href="https://www.ece.pku.edu.cn/info/1046/2596.htm"><strong>Mengyuan Liu*</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2306.08659'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/fanglaosi/Point-In-Context' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
<br />

<div  align="center">    
 <img src="./assets/imgs/teaser_00.jpg" width = 1000  align=center />
</div>

❗❗❗ Our work is the **_first_** to explore in-context learning in 3D point clouds, including task definition, benchmark, and baseline models.

[//]: # (## ☀️Abstract)

[//]: # ()
[//]: # (With the rise of large-scale models trained on broad data, in-context learning has become a new learning paradigm that has demonstrated significant potential in natural language processing and computer vision tasks. Meanwhile, in-context learning is still largely unexplored in the 3D point cloud domain. Although masked modeling has been successfully applied for in-context learning in 2D vision, directly extending it to 3D point clouds remains a formidable challenge. In the case of point clouds, the tokens themselves are the point cloud positions &#40;coordinates&#41; that are masked during inference. Moreover, position embedding in previous works may inadvertently introduce information leakage. To address these challenges, we introduce a novel framework, named Point-In-Context, designed especially for in-context learning in 3D point clouds, where both inputs and outputs are modeled as coordinates for each task. Additionally, we propose the Joint Sampling module, carefully designed to work in tandem with the general point sampling operator, effectively resolving the aforementioned technical issues. We conduct extensive experiments to validate the versatility and adaptability of our proposed methods in handling a wide range of tasks. Furthermore, with a more effective prompt selection strategy, our framework surpasses the results of individually trained models.)

[//]: # ()
[//]: # (## ⚡Features)

[//]: # ()
[//]: # (### $In-context\ learning\ for\ 3D\ understanding$)

[//]: # ()
[//]: # (- The first work to explore the application of in-context learning in the 3D domain.)

[//]: # (- A new framework for tackling multiple tasks &#40;four tasks&#41;, which are unified into the same input-output space.)

[//]: # (- Can improve the performance of our Point-In-Context &#40;Sep & Cat&#41; by selecting higher-quality prompts.)

[//]: # ()
[//]: # (### $New\ benchmark$)

[//]: # ()
[//]: # (- A new multi-task benchmark for evaluating the capability of processing multiple tasks, including reconstruction, denoising, registration, and part segmentation.)

[//]: # ()
[//]: # (### $Strong\ performance$)

[//]: # ()
[//]: # (- Surpasses classical models &#40;PointNet, DGCNN, PCT, PointMAE&#41;, which are equipped with multi-task heads.)

[//]: # (- Surpasses even task-specific models &#40;PointNet, DGCNN, PCT&#41; on registration when given higher-quality prompts.)

## ✋Run

### 1. Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```

Chamfer Distance & embedding
```
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
```

Pytorch3d
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
export CUB_HOME=/usr/local/cuda/include/
FORCE_CUDA=1 python setup.py install
```

### 2. Dataset Generation

You can preprocess the dataset yourself, see the [data_processing](./data/DATASET.md).

Alternatively, we have provided the [pre-processed_datasets](https://drive.google.com/file/d/10z9s3S_r_HEckWZXHnIQkvDF6BdbNOXs/view?usp=sharing) (recommend). Please download it and unzip it in ```data/```

### 3. Training Point-In-Context
To train Point-In-Context on our dataset, run the following command:

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/PIC_Sep.yaml --exp_name exp/training/PIC_Sep
```

### 4. Evaluation
To evaluate the performance on Part Segmentation task, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python eval_seg.py --config cfgs/PIC_Sep.yaml --exp_name exp/evaluate/PIC_Sep --ckpts exp/training/PIC_Sep/ckpt-best.pth --data_path <path_to_data>
```
To evaluate the performance on Reconstruction, Denoising, Registration tasks, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python eval_cd.py --config cfgs/PIC_Sep.yaml --exp_name exp/evaluate/PIC_Sep --ckpts exp/training/PIC_Sep/ckpt-best.pth --data_path <path_to_data>
```

## 😃Visualization

<div  align="center">    
 <img src="./assets/imgs/visualization_main_00.jpg" width = 1000  align=center />
</div>

## License
MIT License

## Citation
If you find our work useful in your research, please consider citing: 
```
@article{fang2023explore,
  title={Explore In-Context Learning for 3D Point Cloud Understanding},
  author={Fang, Zhongbin and Li, Xiangtai and Li, Xia and Buhmann, Joachim M and Loy, Chen Change and Liu, Mengyuan},
  journal={arXiv preprint arXiv:2306.08659},
  year={2023}
}
```

## Acknowledgement

This work is built upon the [Point-MAE](https://github.com/Pang-Yatian/Point-MAE).