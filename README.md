<p align="center">
  <h1 align="center">Novel Part Segmentation in Point Cloud with Adaptive Prompts</h1>
  <p align="center">
    Arxiv, 2024
    <br />
    <a href="https://www.ece.pku.edu.cn/info/1046/2596.htm"><strong>Mengyuan Liu</strong></a>
    ·
    <a href="https://github.com/fanglaosi/"><strong>Zhongbin Fang📧</strong></a>
    ·
    <a href="https://xialipku.github.io/"><strong>Xia Li📧</strong></a>
    .
    <a href="ml.inf.ethz.ch/"><strong>Joachim M. Buhmann</strong></a>
    <br />
    <a href="https://scholar.google.com/citations?user=jz5XKuQAAAAJ&hl=zh-CN&oi=ao"><strong>Deheng Ye</strong></a>
    .
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    .
    <a href="https://www.mmlab-ntu.com/person/ccloy/"><strong>Chen Change Loy</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2404.12352'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv PDF'>
    </a>
    <a href='https://fanglaosi.github.io/Point-In-Context_Pages/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>

> ❗❗❗ Please click the <a href='https://fanglaosi.github.io/Point-In-Context_Pages/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a> button to view our new journal paper
    <a href='https://arxiv.org/abs/2404.12352'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv PDF'>
    </a>: **Novel Part Segmentation in Point Cloud with Adaptive Prompts, 2024.** 
    We propose **Adaptive Prompting Model (APM)**. APM possesses strong performance and 
**generalization ability** in segmentation tasks. It can **generalize to unseen datasets** and can also **perform part-specific segmentation** by customizing unique prompts.

> Ps: the code of APM will be available soon.

<br />
<p align="center">
  <h1 align="center">Explore In-Context Learning for 3D Point Cloud Understanding</h1>
  <p align="center">
    NeurIPS (Spotlight), 2023
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
    <a href="https://www.ece.pku.edu.cn/info/1046/2596.htm"><strong>Mengyuan Liu📧</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2306.08659'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://fanglaosi.github.io/Point-In-Context_Pages/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
<br />



<div  align="center">    
 <img src="./assets/imgs/teaser_00.jpg" width = 1000  align=center />
</div>

⭐ Our work is the **_first_** to explore in-context learning in 3D point clouds, including task definition, benchmark, and baseline models.

[//]: # (# ☀Abstract)

[//]: # ()
[//]: # ()
[//]: # (With the rise of large-scale models trained on broad data, in-context learning has become a new learning paradigm that has demonstrated significant potential in natural language processing and computer vision tasks. Meanwhile, in-context learning is still largely unexplored in the 3D point cloud domain. Although masked modeling has been successfully applied for in-context learning in 2D vision, directly extending it to 3D point clouds remains a formidable challenge. In the case of point clouds, the tokens themselves are the point cloud positions &#40;coordinates&#41; that are masked during inference. Moreover, position embedding in previous works may inadvertently introduce information leakage. To address these challenges, we introduce a novel framework, named Point-In-Context, designed especially for in-context learning in 3D point clouds, where both inputs and outputs are modeled as coordinates for each task. Additionally, we propose the Joint Sampling module, carefully designed to work in tandem with the general point sampling operator, effectively resolving the aforementioned technical issues. We conduct extensive experiments to validate the versatility and adaptability of our proposed methods in handling a wide range of tasks. Furthermore, with a more effective prompt selection strategy, our framework surpasses the results of individually trained models.)


# 🙂News
- [2024.4.19] Our extended journal paper, [Point-In-Context-Segmenter](https://arxiv.org/abs/2404.12352), is released on Arxiv. ⭐⭐⭐
- [2023.9.24] Training and testing code is released, and [PIC-Sep](https://drive.google.com/file/d/1Dkq5V9LNNGBgxWcPo8tkWC05Yi7DCre3/view?usp=sharing) and [PIC-Cat](https://drive.google.com/file/d/1Dkq5V9LNNGBgxWcPo8tkWC05Yi7DCre3/view?usp=sharing) is released
- [2023.9.22] Our [Point-In-Context](https://arxiv.org/abs/2306.08659) is accepted at NeurIPS 2023 as a spotlight! 🎉🎉🎉
- [2023.6.16] Our [Point-In-Context](https://arxiv.org/abs/2306.08659) is released and GitHub repo is created.

# ⚡Features


## In-context learning for 3D understanding


- The first work to explore the application of in-context learning in the 3D domain.
- A new framework for tackling multiple tasks (four tasks), which are unified into the same input-output space.
- Can improve the performance of our Point-In-Context (Sep & Cat) by selecting higher-quality prompts.

## New benchmark

- A new multi-task benchmark for evaluating the capability of processing multiple tasks, including reconstruction, denoising, registration, and part segmentation.

## Strong performance

- Surpasses classical models (PointNet, DGCNN, PCT, PointMAE), which are equipped with multi-task heads.
- Surpasses even task-specific models (PointNet, DGCNN, PCT) on registration when given higher-quality prompts.

# ✋Run

## 1. Requirements
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

## 2. Dataset Generation

You can preprocess the dataset yourself, see the [data_processing](./data/DATASET.md).

Alternatively, we have provided the [pre-processed_datasets](https://drive.google.com/file/d/10z9s3S_r_HEckWZXHnIQkvDF6BdbNOXs/view?usp=sharing) (recommend). Please download it and unzip it in ```data/```

## 3. Training Point-In-Context
To train Point-In-Context on our dataset, run the following command:

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/PIC_Sep.yaml --exp_name exp/training/PIC_Sep
```

## 4. Evaluation
To evaluate the performance on Part Segmentation task, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python eval_seg.py --config cfgs/PIC_Sep.yaml --exp_name exp/evaluate/PIC_Sep --ckpts exp/training/PIC_Sep/ckpt-best.pth --data_path <path_to_data>
```
To evaluate the performance on Reconstruction, Denoising, Registration tasks, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python eval_cd.py --config cfgs/PIC_Sep.yaml --exp_name exp/evaluate/PIC_Sep --ckpts exp/training/PIC_Sep/ckpt-best.pth --data_path <path_to_data>
```

# 📚Released Models


| Name                                  | Params | Rec. (CD↓) | Deno. (CD↓) | Reg. (CD↓) | Part Seg. (mIOU↑) |
|---------------------------------------|:------:|:----------:|:----------:|:---------:|:-----------------:|
| [PIC-Sep](https://drive.google.com/file/d/1Dkq5V9LNNGBgxWcPo8tkWC05Yi7DCre3/view?usp=sharing)     | **28.9M**  |  **4.4**   |    **7.5**     |    **8.6**    |     **78.60**     |
| [PIC-Cat](https://drive.google.com/file/d/1Dkq5V9LNNGBgxWcPo8tkWC05Yi7DCre3/view?usp=sharing) | **29.0M**  |  **4.9**   |    **6.0**     |   **14.4**    |     **79.75**     |

> The above results are reimplemented  and are basically consistent with the results reported in the paper.

# 😃Visualization
In-context inference demo (part segmentation, denoising, registration). Our Point-In-Context is designed to perform various tasks on a given query point cloud, adapting its operations based on different prompt pairs. Notably, the PIC has the ability to accurately predict the correct point cloud, even when provided with a clean input point cloud for the denoising task.

![in-context_demo](./assets/gifs/in-context_demo.gif)

Visualization of predictions obtained by our PIC-Sep and their corresponding targets in different point cloud tasks.

![visual](./assets/imgs/visualization_main_00.jpg)

# License
MIT License

# Citation
If you find our work useful in your research, please consider citing: 
```
@article{liu2024pointincontext,
  title={Point-In-Context: Understanding Point Cloud via In-Context Learning}, 
  author={Liu, Mengyuan and Fang, Zhongbin and Li, Xia and Buhmann, Joachim M and Li, Xiangtai and Loy, Chen Change},
  journal={arXiv preprint arXiv:2401.08210},
  year={2024}
}
@article{fang2024explore,
  title={Explore in-context learning for 3d point cloud understanding},
  author={Fang, Zhongbin and Li, Xiangtai and Li, Xia and Buhmann, Joachim M and Loy, Chen Change and Liu, Mengyuan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

# Acknowledgement

This work is built upon the [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Visual Prompt](https://github.com/amirbar/visual_prompting), [Painter](https://github.com/baaivision/Painter).

Thanks to the following excellent works: [PointNet](https://github.com/fxia22/pointnet.pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn), [PCT](https://github.com/MenghaoGuo/PCT), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [ACT](https://github.com/RunpeiDong/ACT), [I2P-MAE](https://github.com/ZrrSkywalker/I2P-MAE), [ReCon](https://github.com/qizekun/ReCon); 
