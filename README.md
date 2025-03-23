# Removing Structured Noise with Diffusion Models

### [Code](https://github.com/tristan-deep/joint-diffusion) | [Paper](https://arxiv.org/abs/2302.05290) | [Blog](https://tristan-deep.github.io/posts/2023-02-11-diffusion-models/) | [Weights](https://huggingface.co/collections/tristan-deep/removing-structured-noise-using-diffusion-models-67802b581433e4ecf48e49bc)

[Tristan Stevens](https://tristan-deep.github.io/),
[Hans van Gorp](https://www.tue.nl/en/research/researchers/hans-van-gorp/),
[Can Meral](https://scholar.google.com/citations?user=kX0KaeUAAAAJ&hl=en&oi=sra),
[Junseob Shin](https://scholar.google.com/citations?user=R2cdBMsAAAAJ),
[Jason Yu](https://ieeexplore.ieee.org/author/37088639661),
[Jean-Luc Robert](https://scholar.google.com/citations?hl=en&user=BrY9ygYAAAAJ),
[Ruud van Sloun](https://www.tue.nl/en/research/researchers/ruud-van-sloun/)<br>

> [!TIP]
> Weights are now hosted on [Hugging Face](https://huggingface.co/collections/tristan-deep/removing-structured-noise-using-diffusion-models-67802b581433e4ecf48e49bc) 🤗.

> [!NOTE]
> Our paper got accepted to [TMLR](https://openreview.net/forum?id=BvKYsaOVEn) 🎉!

Official repository of the Removing Structured Noise with Diffusion Models [paper](https://arxiv.org/abs/2302.05290).
The joint posterior sampling functions for diffusion models proposed in the paper can be found in [sampling.py](./generators/SGM/sampling.py) and [guidance.py](./generators/SGM/guidance.py). For the interested reader, a more in depth explanation of the method and underlying principles can be found [here](https://tristan-deep.github.io/posts/2023/03/diffusion-models/). Any information on how to setup the code and run inference can be found in the [getting started](#getting-started) section.

If you find the code useful for your research, please cite [the paper](https://arxiv.org/abs/2302.05290):
```bib
@article{
  stevens2025removing,
  title={Removing Structured Noise using Diffusion Models},
  author={Stevens, Tristan SW and van Gorp, Hans and Meral, Faik C and Shin, Junseob and Yu, Jason and Robert, Jean-Luc and van Sloun, Ruud JG},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=BvKYsaOVEn},
}
```
<p align="center">
<figure align="center">
  <img src="./images/joint-diffusion-diagram.png" alt="diagram" style="width: 500;"/>
  <figcaption><strong>Overview of the proposed joint posterior sampling method for removing structured noise using diffusion models.</strong></figcaption>
</figure>
</p>

## Table of contents
- [Removing Structured Noise with Diffusion Models](#removing-structured-noise-with-diffusion-models)
    - [Code | Paper | Project Page | Blog | Weights](#code--paper--project-page--blog--weights)
  - [Table of contents](#table-of-contents)
  - [Structured denoising](#structured-denoising)
      - [Structured denoising on CelebA with MNIST corruption](#structured-denoising-on-celeba-with-mnist-corruption)
      - [Projection, DPS, PiGDM and Flow](#projection-dps-pigdm-and-flow)
  - [Getting started](#getting-started)
    - [Install environment](#install-environment)
    - [Download weights](#download-weights)
    - [Run inference](#run-inference)
    - [Inference configs](#inference-configs)
    - [Datasets](#datasets)
  - [References](#references)

## Structured denoising

Run the following command with  `keep_track` set to `true` in the [config](./configs/inference/paper/celeba_mnist_pigdm.yaml) to run the structured denoising and generate the animation.
```shell
python inference.py -e paper/celeba_mnist_pigdm -t denoise -m sgm
```
<p align="center">
<figure align="center">
  <img src="./images/celeba_mnist_sgm_animation.gif" alt="CelebA structured denoising" style="width: 400;"/>
  <figcaption><strong>Structured denoising with the joint diffusion method.</strong></figcaption>
</figure>
</p>


#### Structured denoising on CelebA with MNIST corruption

<table>
  <tr>
    <td><img src="./images/comparison_celeba_mnist.png" alt="CelebA" style="width: 500;"></td>
    <td><img src="./images/comparison_celeba_mnist_ood.png" alt="OOD"  style="width: 500;"></td>
  </tr>
  <tr>
    <td align="center"><strong>CelebA</strong></td>
    <td align="center"><strong>Out-of-distribution dataset</strong></td>
  </tr>
</table>

#### Projection, DPS, PiGDM and Flow

<table>
  <tr>
    <td><img src="./images/comparison_celeba_mnist_proj_dps_pidgm.png" alt="CelebA" style="width: 500;"></td>
    <td><img src="./images/comparison_celeba_mnist_proj_dps_pidgm_ood.png" alt="OOD"  style="width: 500;"></td>
  </tr>
  <tr>
    <td align="center"><strong>CelebA</strong></td>
    <td align="center"><strong>Out-of-distribution dataset</strong></td>
  </tr>
</table>


## Getting started
### Install environment
Although manuall installation is possible, we recommend using the provided Dockerfile to build the environment. First, clone the repository and build the Docker image.

```shell
git clone git@github.com:tristan-deep/joint-diffusion.git
cd joint-diffusion
docker build . -t joint-diffusion:latest
```

This will build the image `joint-diffusion:latest` with all the necessary dependencies. To run the image, use the following command:

```shell
docker run -it --gpus all --user "$(id -u):$(id -g)" -v $(pwd):/joint-diffusion --name joint-diffusion joint-diffusion:latest
```

For manual installation one can check the [requirements.txt](./requirements/requirements.txt) file for dependencies as well as install CUDA enabled [Tensorflow](https://www.tensorflow.org/install/pip)(2.9) and [PyTorch](https://pytorch.org/get-started/locally/)(1.12) (latter only needed for the GLOW baseline).


### Download weights
Pretrained weights should be automatically downloaded by the Hugging Face API, please create your access token [here](https://huggingface.co/docs/hub/en/security-tokens). However, they can also be manually downloaded [here](https://huggingface.co/collections/tristan-deep/removing-structured-noise-using-diffusion-models-67802b581433e4ecf48e49bc). The `run_id` model in the config either points to the Hugging Face repo using `hf://` (default) or to a local folder with the checkpoints manually saved. In those folders, besides the checkpoint, a training config `.yaml` file is provided for each trained model (necessary for inference, to build the model again).

### Datasets
Make sure to set the `data_root` parameter in the inference config (for instance this [config](./configs/inference/paper/celeba_mnist_pigdm.yaml)). It is set to the working directory as default. All datasets (for instance CelebA and MNIST) should be (automatically) downloaded and put as a subdirectory to the specified `data_root`. More information can be found in the [datasets.py](datasets.py) docstrings.

### Run inference
Use the [inference.py](inference.py) script for inference.
```shell
usage: inference.py [-h]
                    [-e EXPERIMENT]
                    [-t {denoise,sample,evaluate,show_dataset}]
                    [-n NUM_IMG]
                    [-m [MODELS ...]]
                    [-s SWEEP]

options:
  -h, --help            show this help message and exit
  -e EXPERIMENT, --experiment EXPERIMENT
                        experiment name located at ./configs/inference/<experiment>)
  -t {denoise,sequence_denoise,sample,evaluate,show_dataset,plot_results,run_metrics}, --task
                        which task to run
  -n NUM_IMG, --num_img NUM_IMG
                        number of images
  -m [MODELS ...], --models [MODELS ...]
                        list of models to run
  -s SWEEP, --sweep SWEEP
                        sweep config located at ./configs/sweeps/<sweep>)
  -ef EVAL_FOLDER, --eval_folder EVAL_FOLDER
                        eval file located at ./results/<eval_folder>)
```

Example:
Main experiment with CelebA data and MNIST corruption:
```shell
python inference.py -e paper/celeba_mnist_pigdm -t denoise -m bm3d nlm gan sgm
```
Denoising comparison with multiple models:
```shell
python inference.py -e paper/celeba_denoising -t denoise -m bm3d nlm gan sgm
```
Or to run a sweep
```shell
python inference.py -e paper/celeba_denoising -t denoise -m sgm -s sgm_sweep
```

### Inference configs
All working inference configs are found in the [./configs/inference/paper](./configs/inference/paper) folder. Path to those inference configs (or just the name of them) should be provided to the `--experiment` flag when calling the [inference.py](inference.py) script.

## References
- Our paper https://arxiv.org/abs/2302.05290
- Diffusion model implementation adopted from https://github.com/yang-song/score_sde_pytorch
- Glow and GAN implementations from https://github.com/CACTuS-AI/GlowIP
