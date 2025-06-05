<div align="center">

<h1>SFDM: Robust Decomposition of Geometry and Reflectance for Realistic Face Rendering from Sparse-view Images</h1>

<h4 align="center">
  <a href="https://kingjg.github.io/SFDM.github.io/" target='_blank'>[Project Page]</a> •
  <a href="https://arxiv.org/abs/2312.06085" target='_blank'>[arXiv]</a>
  <br>
</h4>

---
</div>

Official repository for "SFDM: Robust Decomposition of Geometry and Reflectance for Realistic Face Rendering from Sparse-view Images". 

>In this study, we introduce a novel two-stage technique for decomposing and reconstructing facial features from sparse-view images, a task made challenging by the unique geometry and complex skin reflectance of each individual. To synthesize 3D facial models more realistically, we endeavor to decouple key facial attributes from the RGB color, including geometry, diffuse reflectance, and specular reflectance. Specifically, we design a Sparse-view Face Decomposition Model (SFDM): 1) In the first stage, we create a general facial template from a wide array of individual faces, encapsulating essential geometric and reflectance characteristics. 2) Guided by this template, we refine a specific facial model for each individual in the second stage, considering the interaction between geometry and reflectance, as well as the effects of subsurface scattering on the skin. With these advances, our method can reconstruct high-quality facial representations from as few as three images. The comprehensive evaluation and comparison reveal that our approach outperforms existing methods by effectively disentangling geometric and reflectance components, significantly enhancing the quality of synthesized novel views, and paving the way for applications in facial relighting and reflectance editing.

## Installation

```shell
# Create Conda Environment
conda create -n sfdm python=3.8 
conda activate sfdm

# Install requirements
pip install -r requirement.txt
```

## Data Preparation

We mainly utilize <a href="https://facescape.nju.edu.cn/" target="_blank">FaceScape Dataset</a> for the evaluation. Please kindly refer <a href="https://github.com/lioryariv/volsdf" target="_blank">NeuFace</a> for the preprocessing of FaceScape dataset and downloading the pretrained ImFace model.

## Individual Reconstruction

### Train on Facescape

```shell
python tools/train.py --cfg ./config/train_sfdm_st2.yaml
```

### Evaluate on Facescape

```shell
# Evaluation
# change eval_type in the config file to 'evalset'
python tools/eval.py --cfg ./config/train_sfdm_st2.yaml --out_dir ./exp_pl/results --ckpt $PATH_TO_CKPT
```

### Test on Facescape

```shell
# Evaluation
# change eval_type in the config file to 'testset'
python tools/eval.py --cfg ./config/train_sfdm_st2.yaml --out_dir ./exp_pl/results --ckpt $PATH_TO_CKPT
```

## Citation

```
@article{jin2023robust,
  title={Robust Geometry and Reflectance Disentanglement for 3D Face Reconstruction from Sparse-view Images},
  author={Jin, Daisheng and Hu, Jiangbei and Xu, Baixin and Dai, Yuxin and Qian, Chen and He, Ying},
  journal={arXiv preprint arXiv:2312.06085},
  year={2023}
}
```

## Acknowledgement

The codebase is developed based on <a href="https://github.com/aejion/NeuFace" target="_blank">NeuFace</a> and <a href="https://github.com/xubaixinxbx/3dheads" target="_blank">3dheads</a>. Many thanks for these great projects!