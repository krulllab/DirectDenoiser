# DirectDenoiser
Code for [Direct Unsupervised Denoising](https://openaccess.thecvf.com/content/ICCV2023W/BIC/papers/Salmon_Direct_Unsupervised_Denoising_ICCVW_2023_paper.pdf)<br>
<sup>1</sup>Benjamin Salmon and <sup>2</sup>Alexander Krull<br>
<sup>1, 2</sup>University of Birmingham<br>
<sup>1</sup>brs209@student.bham.ac.uk, <sup>2</sup>a.f.f.krull@bham.ac.uk<br>
This project includes code from the Hierarchical DivNoising project, which is licensed under the MIT License - [HDN project](https://github.com/juglab/HDN).


![Inference time vs PSNR](https://github.com/krulllab/DirectDenoising/tree/main/resources/inference_time.pdf)

Unsupervised deep learning-based denoisers like [Hierarchical DivNoising](https://github.com/juglab/HDN) are trained to denoise images without examples of clean, noise-free images. They achieve excellent results, but are limited in application by prolonged inference times. This stems from the fact that unsupervised denoisers do not learn a direct mapping from noisy to clean, but instead learn to produce random samples from a posterior distribution over the clean images that could underlie a given noisy image. Practitioners are often interested in the mean of this distribution, which is typically estimated by averaging 100 to 1000 random samples, incurring prohibitively computation times.<br>
In this project, we instead train an additional network to estimate this mean directly - the Direct Denoiser. This increases training time by around 1.25x (using our hardware) but reduces inference time by over 2000x by estimating the mean of the posterior distribution in a single pass. This turns hours of inference time into seconds and days into minutes.

### BibTeX
```
@inproceedings{salmon2023direct,
  title={Direct Unsupervised Denoising},
  author={Salmon, Benjamin and Krull, Alexander},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3838--3845},
  year={2023}
}
```

### Dependenceis
We recommend installing the dependencies in a conda environment. If you haven't already, install miniconda on your system by following this [link](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).<br>
Once conda is installed, create and activate an environment by entering these lines into a command line interface:<br>
1. `conda create --name directdenoiser`
2. `conda activate directdenoiser`


Next, install PyTorch and torchvision for your system by following this [link](https://pytorch.org/get-started/locally/).<br> 
After that, you're ready to install the dependencies for this repository:<br>
`pip install lightning jupyterlab matplotlib tifffile scikit-learn tensorboard`

### Example notebooks
This repository contains 3 notebooks that will first download then denoise the [C. Majalis dataset](https://ieeexplore.ieee.org/abstract/document/9098336?casa_token=ROPuswhAvi0AAAAA:BYQUOnGY51SEqy3CAe7ZTzoOpjjfq8oWrwcJF6KfF4KzIlrjpCL0mR7H7TjDV802pTiJfe0ufg).