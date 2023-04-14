# Local-Global MCMC kernels: the bost of both worlds (NeurIPS 2022) [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/21c86d5b10cdc28664ccdadf0a29065a-Abstract-Conference.html)

[[ArXiv]](https://arxiv.org/abs/2111.02702)

Authors: Sergey Samsonov, Evgeny Lagutin, Marylou Gabrié, Alain Durmus, Alexey Naumov, Eric Moulines.

> **Abstract:** *In the present paper we study an Explore-Exploit Markov chain Monte Carlo strategy (Ex2MCMC) that combines local and global samplers showing that it enjoys the advantages of both approaches. We prove V-uniform geometric ergodicity of Ex2MCMC without requiring a uniform adaptation of the global sampler to the target distribution. We also compute explicit bounds on the mixing rate of the Explore-Exploit strategy under realistic conditions. Moreover, we also analyze an adaptive version of the strategy (FlEx2MCMC) where a normalizing flow is trained while sampling to serve as a proposal for global moves. We illustrate the efficiency of Ex2MCMC and its adaptive version on classical sampling benchmarks as well as in sampling high-dimensional distributions defined by Generative Adversarial Networks seen as Energy Based Models.*
> 
<!-- This repository contains Python code to reproduce experiments from [**Local-Global MCMC kernels: the bost of both worlds**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/21c86d5b10cdc28664ccdadf0a29065a-Abstract-Conference.html) (NeurIPS'22). -->


- [Local-Global MCMC kernels: the bost of both worlds (NeurIPS 2022) \[Paper\]](#local-global-mcmc-kernels-the-bost-of-both-worlds-neurips-2022-paper)
  - [Single chain mixing](#single-chain-mixing)
  - [Sampling from GAN as Energy-Based Models with MCMC](#sampling-from-gan-as-energy-based-models-with-mcmc)
  - [Algorithms](#algorithms)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Experiments with synthetic distributions:](#experiments-with-synthetic-distributions)
    - [Experiments with GANs on MNIST dataset](#experiments-with-gans-on-mnist-dataset)
    - [Experiments with GANs on CIFAR10 dataset](#experiments-with-gans-on-cifar10-dataset)
    - [Sampling and FID computation](#sampling-and-fid-computation)
  - [Results](#results)
    - [FID and Inception Score (CIFAR10)](#fid-and-inception-score-cifar10)
    - [Sampling trajectories (CIFAR10)](#sampling-trajectories-cifar10)
    - [Energy landscape approximation (MNIST)](#energy-landscape-approximation-mnist)
  - [Citation](#citation)

## Single chain mixing

<img src="./imgs/gaussian_mixture.png" alt="i-SIR" width="900"/>

## Sampling from GAN as Energy-Based Models with MCMC


<img src="./imgs/fid_flex.png" alt="FID" width="385"/> <img src="./imgs/is_flex.png" alt="Inception Score" width="400"/> 
<!-- <img src="./imgs/energy_flex.png" alt="Energy" width="270"/>  -->


## Algorithms 
<!-- **i-SIR:**

<img src="./algs/isir.png" alt="i-SIR" width="600"/> -->

**Ex2MCMC:**

<img src="./imgs/ex2.png" alt="Ex2MCMC" width="600"/>

**FlEx2MCMC:**

<img src="./imgs/flex.png" alt="FlEx2MCMC" width="600"/>

## Installation

Create environment:

```bash
conda create -n ex2mcmc python=3.8
conda activate ex2mcmc
```

Install poetry (if absent):
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create false
```

Install the project:
```bash
poetry install
```

Download checkpoints:

CIFAR10:

| GAN   |     Steps     |  Path, G |  Path, D |
|:----------|:-------------:|:------:|:------:|
| DCGAN NS  | 100k      |   [netG_100000_steps.pth](https://drive.google.com/file/d/1gv8_qr_xa8hJzdJpBXiKr8v922EqcE-E/view?usp=share_link) |   [netD_100000_steps.pth](https://drive.google.com/file/d/1u1sPUmlvyhcbNDX2DVsR-mGOzqQ6U8sh/view?usp=share_link) |
| SNGAN, Hinge  | 100k      |   [netG.pth](https://drive.google.com/file/d/118zC_iEkN27jGLVNmDuQpMeyw7BKOUra/view?usp=share_link) |   [netD.pth](https://drive.google.com/file/d/1xU5FV59TLhAlkFubJGmJVS87HnZZ2xHT/view?usp=share_link) |

MNIST:

| GAN      |  Path |
|:----------|:-------------:|
| Vanilla  |   [vanilla_gan.pth](https://drive.google.com/file/d/1xa1v4hPQQdU2RkhjMn5sFZCITxTJ5Dhj/view?usp=share_link) |
| WGAN CP  |   [wgan.pth](https://drive.google.com/file/d/17nQJnfs2_T6kyahnkW3fu8AVY54kmRmw/view?usp=share_link) |

You also can run script to download checkpoints:

```bash
chmod +x get_ckpts.sh
./get_ckpts.sh
```

Download statistics for FID cimputation for CIFAR10 dataset:

```bash
gdown 1jjgB_iuvmoVAXPRvVTI_hBfuIz7mQgOg -O stats/fid_stats_cifar10.npz
```

<!-- | WGAN GP   | --        |   [TBD]() |   [TBD]() | -->

## Usage

 ### Experiments with synthetic distributions:
  
| Experiment | Path | Colab |
|:----------|:-------|:-----:|
| Toyish Gaussian   |     ```experiments/exp_synthetic/toyish_gaussian.ipynb``` | [TBD]() |
| Gaussian mixture  |     ```experiments/exp_synthetic/gaussian_mixture.ipynb``` | [TBD]() |
| FlEx for banana-shaped distribution   |     ```experiments/exp_synthetic/flex_banana.ipynb``` | [TBD]() |
| FlEx for Neal's funnel distribution   |     ```experiments/exp_synthetic/flex_funnel.ipynb``` | [TBD]() |

To reproduce the experimets on banana-shaped and funnel distributions:

```bash
python experiments/exp_synthetic/banana_funnel_metrics.py --distribution {banana,funnel} --device cuda:0
```

 ### Experiments with GANs on MNIST dataset
 
 ```experiments/exp_mnist/JSGAN_samples.ipynb```

 ```experiments/exp_mnist/WGAN_samples.ipynb```

 ### Experiments with GANs on CIFAR10 dataset

```experiments/exp_cifar10/DCGAN_samples.ipynb```

```experiments/exp_cifar10/SNGAN_samples.ipynb```

### Sampling and FID computation

```bash
python experiments/exp_fid/run_mmc_dcgan.py configs/mcmc_configs/{ula,mala,isir,ex2mcmc,flex2mcmc}.yml configs/mmc_dcgan.yml
```

## Results

### FID and Inception Score (CIFAR10)
| GAN | MCMC | steps | Inception Score | FID  |
|:----|:-----|:------|:---------------:|:----:|
|DCGAN| none | 0     |                 |      |
|DCGAN| i-SIR  | 1k     |              |      |
|DCGAN| MALA  | 1k      |               |      |
|DCGAN| Ex2MCMC  | 1k   |              |      |
|DCGAN| FlEx2MCMC  | 1k |              |      |


### Sampling trajectories (CIFAR10)
Generation trajectories for DCGAN, top to bottom: i-SIR, MALA, Ex2MCMC, FlEx2MCMC:

<img src="./imgs/cifar10_dcgan_gen.png" alt="CIFAR10 generations" width="600"/> 


### Energy landscape approximation (MNIST)

Projection of GAN samples onto the energy landsape when trained on MNIST dataset:

<img src="./imgs/energy_landscape.png" alt="energy landscape" width="600"/> 

## Citation

```bibtex
@article{samsonov2022local,
  title={Local-Global MCMC kernels: the best of both worlds},
  author={Samsonov, Sergey and Lagutin, Evgeny and Gabri{\'e}, Marylou and Durmus, Alain and Naumov, Alexey and Moulines, Eric},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={5178--5193},
  year={2022}
}
```




