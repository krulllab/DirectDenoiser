import random

import torch
import pytorch_lightning as pl
from torch import optim
import numpy as np


class Backbone(pl.LightningModule):
    """Module to unify the training of the VAE and direct denoiser.
    
    
    Parameters
    ----------
    vae : torch.nn.Module
        The Ladder VAE model from https://github.com/juglab/HDN/.
    direct_denoiser : torch.nn.Module, optional
        A deterministic network that will learn to predict the VAE's output.
        Omit for original HDN training.
    data_mean : float, optional
        The mean of the training data. Used to normalise the data before
        passing it to the VAE.
    data_std : float, optional
        The standard deviation of the training data. Used to normalise the data
        before passing it to the VAE.
    gaussian_noise_std : float, optional
        If a trained noise model is not used, this is the standard deviation
        of the Gaussian noise that is in the training data.
    n_grad_batches : int, optional
        The number of batches to accumulate gradients over before updating the
        weights.
    lr : float, optional
        The learning rate for the Adamax optimiser for both VAE and Direct Denoiser.
        """


    def __init__(
        self,
        vae,
        direct_denoiser=None,
        data_mean=0,
        data_std=1,
        gaussian_noise_std=None,
        n_grad_batches=8,
        lr=3e-4,
    ):
        self.save_hyperparameters()

        super().__init__()
        self.vae = vae
        self.direct_denoiser = direct_denoiser
        self.data_mean = data_mean
        self.data_std = data_std
        self.gaussian_noise_std = gaussian_noise_std
        self.n_grad_batches = n_grad_batches
        self.lr = lr

        self.automatic_optimization = False 

    def forward(self, x):
        x = (x - self.data_mean) / self.data_std

        vae_out = self.vae(x)

        if self.direct_denoiser is not None:
            s_direct = self.direct_denoiser(x)
        else:
            s_direct = None

        out = {
            "s_hat": vae_out["out_mean"],
            "s_direct": s_direct,
            "kl_loss": vae_out["kl_loss"],
            "ll": vae_out["ll"],
        }

        return out

    def configure_optimizers(self):
        optimizers = []
        schedulers = []

        vae_params = self.vae.parameters()
        vae_optimizer = optim.Adamax(vae_params, lr=self.lr)
        optimizers.append(vae_optimizer)
        vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, "min", patience=10, factor=0.5, min_lr=1e-12, verbose=True
        )
        schedulers.append(vae_scheduler)

        if self.direct_denoiser is not None:
            dd_params = self.direct_denoiser.parameters()
            dd_optimizer = optim.Adamax(dd_params, lr=self.lr)
            optimizers.append(dd_optimizer)
            dd_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                dd_optimizer, "min", patience=10, factor=0.5, min_lr=1e-12, verbose=True
            )
            schedulers.append(dd_scheduler)

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        self.vae.mode_pred = False
        x = batch[0]

        out = self(x)

        kl_loss = out["kl_loss"] / float(x.shape[2] * x.shape[3])

        reconstruction_loss = -out["ll"].mean()
        if self.gaussian_noise_std is not None:
            reconstruction_loss = reconstruction_loss / (
                (self.gaussian_noise_std / self.data_std) ** 2
            )

        elbo = kl_loss + reconstruction_loss
        self.manual_backward(elbo)
        self.log("train/elbo", elbo)
        self.log("train/kl_loss", kl_loss)
        self.log("train/reconstruction_loss", reconstruction_loss)

        if self.direct_denoiser is not None:
            dd_loss = self.direct_denoiser.loss(
                out["s_hat"].detach(), out["s_direct"]
            ).mean()
            self.manual_backward(dd_loss)
            self.log("train/dd_loss", dd_loss)

        if (batch_idx + 1) % self.n_grad_batches == 0:
            optimizers = self.optimizers()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        vae_scheduler = schedulers[0]
        vae_scheduler.step(self.trainer.callback_metrics["val/elbo"])

        if self.direct_denoiser is not None:
            dd_scheduler = schedulers[1]
            dd_scheduler.step(self.trainer.callback_metrics["val/dd_loss"])

    def log_tensorboard_images(self, img, img_name):
        img = img.cpu().numpy()
        normalised_img = (img - np.percentile(img, 1)) / (
            np.percentile(img, 99) - np.percentile(img, 1)
        )
        clamped_img = np.clip(normalised_img, 0, 1)
        self.trainer.logger.experiment.add_image(
            img_name, clamped_img, self.current_epoch
        )

    def validation_step(self, batch, batch_idx):
        self.vae.mode_pred = False
        x = batch[0]

        out = self(x)

        kl_loss = out["kl_loss"] / float(x.shape[2] * x.shape[3])

        reconstruction_loss = -out["ll"].mean()
        if self.gaussian_noise_std is not None:
            reconstruction_loss = reconstruction_loss / (
                (self.gaussian_noise_std / self.data_std) ** 2
            )

        elbo = kl_loss + reconstruction_loss
        self.log("val/elbo", elbo)
        self.log("val/kl_loss", kl_loss)
        self.log("val/reconstruction_loss", reconstruction_loss)

        if self.direct_denoiser is not None:
            dd_loss = self.direct_denoiser.loss(
                out["s_hat"].detach(), out["s_direct"]
            ).mean()
            self.log("val/dd_loss", dd_loss)

        if batch_idx == 0:
            idx = random.randint(0, x.shape[0] - 1)
            out = self.forward(x[idx : idx + 1].repeat_interleave(10, 0))
            mmse = torch.mean(out["s_hat"], 0, keepdim=True)
            self.log_tensorboard_images(x[idx], "inputs/noisy")
            self.log_tensorboard_images(out["s_hat"][0], "outputs/sample 1")
            self.log_tensorboard_images(out["s_hat"][1], "outputs/sample 2")
            self.log_tensorboard_images(mmse[0], "outputs/mmse (10 samples)")
            if self.direct_denoiser is not None:
                self.log_tensorboard_images(
                    out["s_direct"][0], "outputs/direct estimate"
                )

    @torch.no_grad()
    def predict_vae(self, x, n_samples, batch_size):
        x = (x - self.data_mean) / self.data_std
        samples = self.vae.predict(x, n_samples, batch_size)
        samples = samples * self.data_std + self.data_mean

        return samples

    @torch.no_grad()
    def predict_direct_denoiser(self, x):
        x = (x - self.data_mean) / self.data_std
        s_direct = self.direct_denoiser(x)
        s_direct = s_direct * self.data_std + self.data_mean

        return s_direct
    