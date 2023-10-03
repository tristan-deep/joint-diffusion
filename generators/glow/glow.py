"""GLOW normalizing flow model
Author(s): https://github.com/CACTuS-AI/GlowIP
Changes: Tristan Stevens
"""
import datetime
import time

import numpy as np
import skimage.io as sio
import torch
import torch.nn as nn

import wandb
from generators.glow.flow import Flow
from generators.glow.split import Split
from generators.glow.squeeze import Squeeze
from utils.utils import convert_torch_tensor, tf_tensor_to_torch

# device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class STEFloor(torch.autograd.Function):
    """Sraight through estimator of floor function

    see: https://www.hassanaskary.com/python/pytorch/deep%
    20learning/2020/09/19/intuitive-explanation-of-straight
    -through-estimators.html

    """

    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)


class Glow(nn.Module):
    def __init__(
        self,
        image_shape,
        K,
        L,
        coupling,
        device,
        n_bits_x=8,
        coupling_bias=0,
        squeeze_contig=False,
        nn_init_last_zeros=False,
    ):
        super(Glow, self).__init__()
        self.image_shape = image_shape
        self.K = K
        self.L = L
        self.coupling = coupling
        self.n_bits_x = n_bits_x
        self.device = device
        self.init_resizer = False
        self.nn_init_last_zeros = nn_init_last_zeros
        self.coupling_bias = coupling_bias
        self.squeeze_contig = squeeze_contig

        # setting up layers
        c, h, w = image_shape
        self.glow_modules = nn.ModuleList()

        for l in range(L - 1):
            # step of flows
            squeeze = Squeeze(factor=2, contiguous=self.squeeze_contig)
            if w == 1:
                c = c * 2
            else:
                c = c * 4
            self.glow_modules.append(squeeze)
            for k in range(K):
                flow = Flow(
                    c,
                    self.coupling,
                    device,
                    self.coupling_bias,
                    self.nn_init_last_zeros,
                )
                self.glow_modules.append(flow)
            split = Split()
            c = c // 2
            self.glow_modules.append(split)
        # L-th flow
        squeeze = Squeeze(factor=2)
        if w == 1:
            c = c * 2
        else:
            c = c * 4
        self.glow_modules.append(squeeze)
        flow = Flow(c, self.coupling, device, nn_init_last_zeros)
        self.glow_modules.append(flow)

        # at the end
        self.to(device)

    def forward(self, x, logdet=None, reverse=False, reverse_clone=True):
        if not reverse:
            n, c, h, w = x.size()
            Z = []
            if logdet is None:
                logdet = torch.tensor(
                    0.0, requires_grad=False, device=self.device, dtype=torch.float
                )
            for i in range(len(self.glow_modules)):
                module_name = self.glow_modules[i].__class__.__name__
                if module_name == "Squeeze":
                    x, logdet = self.glow_modules[i](x, logdet=logdet, reverse=False)
                elif module_name == "Flow":
                    x, logdet, actloss = self.glow_modules[i](
                        x, logdet=logdet, reverse=False
                    )
                elif module_name == "Split":
                    x, z = self.glow_modules[i](x, reverse=False)
                    Z.append(z)
                else:
                    raise ValueError("Unknown Layer")

            Z.append(x)

            if not self.init_resizer:
                self.sizes = [t.size() for t in Z]
                self.init_resizer = True
            return Z, logdet, actloss

        if reverse:
            if reverse_clone:
                x = [x[i].clone().detach() for i in range(len(x))]
            else:
                x = [x[i] for i in range(len(x))]
            x_rev = x[-1]  # here x is z -> latent vector
            k = len(x) - 2
            for i in range(len(self.glow_modules) - 1, -1, -1):
                module_name = self.glow_modules[i].__class__.__name__
                if module_name == "Split":
                    x_rev = self.glow_modules[i](x_rev, x[k], reverse=True)
                    k = k - 1
                elif module_name == "Flow":
                    x_rev = self.glow_modules[i](x_rev, reverse=True)
                elif module_name == "Squeeze":
                    x_rev = self.glow_modules[i](x_rev, reverse=True)
                else:
                    raise ValueError("Unknown Layer")
            return x_rev

    def nll_loss(self, x, logdet=None):
        n, c, h, w = x.size()
        z, logdet, actloss = self.forward(x, logdet=logdet, reverse=False)
        if not self.init_resizer:
            self.sizes = [t.size() for t in z]
            self.init_resizer = True
        z_ = [z_.view(n, -1) for z_ in z]
        z_ = torch.cat(z_, dim=1)
        mean = 0
        logs = 0
        logdet += float(-np.log(256.0) * h * w * c)
        logpz = -0.5 * (
            logs * 2.0
            + ((z_ - mean) ** 2) / np.exp(logs * 2.0)
            + float(np.log(2 * np.pi))
        ).sum(-1)
        nll = -(logdet + logpz).mean()
        nll = nll / float(np.log(2.0) * h * w * c)
        return (
            nll,
            -logdet.mean().item(),
            -logpz.mean().item(),
            z_.mean().item(),
            z_.std().item(),
        )

    def preprocess(self, x, clone=False):
        if clone:
            x = x.detach().clone()
        n_bins = 2**self.n_bits_x
        x = x / 2 ** (8 - self.n_bits_x)
        x = STEFloor.apply(x)
        x = x / n_bins - 0.5
        x = x + torch.tensor(
            np.random.uniform(0, 1 / n_bins, x.size()),
            dtype=torch.float,
            device=self.device,
        )
        return x

    def postprocess(self, x, floor_clamp=True):
        n_bins = 2**self.n_bits_x
        if floor_clamp:
            x = torch.floor((x + 0.5) * n_bins) * (1.0 / n_bins)
            x = torch.clamp(x, 0, 1)
        else:
            x = x + 0.5
        return x

    def generate_z(self, n, mu=0, std=1, to_torch=True):
        # a function to reshape z so that it can be fed to the reverse method
        z_np = [np.random.normal(mu, std, [n] + list(size)[1:]) for size in self.sizes]
        if to_torch:
            z_t = [
                torch.tensor(
                    t, dtype=torch.float, device=self.device, requires_grad=False
                )
                for t in z_np
            ]
            return z_np, z_t
        else:
            return z_np

    def flatten_z(self, z):
        n = z[0].size()[0]
        z = [z_.view(n, -1) for z_ in z]
        z = torch.cat(z, dim=1)
        return z

    def unflatten_z(self, z, clone=True):
        # z must be torch tensor
        n_elements = [np.prod(s[1:]) for s in self.sizes]
        z_unflatten = []
        start = 0
        for n, size in zip(n_elements, self.sizes):
            end = start + n
            z_ = z[:, start:end].view([-1] + list(size)[1:])
            if clone:
                z_ = z_.clone().detach()
            z_unflatten.append(z_)
            start = end
        return z_unflatten

    def set_actnorm_init(self):
        # a method to set actnorm to True to prevent re-initializing for resuming training
        for i in range(len(self.glow_modules)):
            module_name = self.glow_modules[i].__class__.__name__
            if module_name == "Flow":
                self.glow_modules[i].actnorm.initialized = True
                self.glow_modules[i].coupling.net.actnorm1.initialized = True
                self.glow_modules[i].coupling.net.actnorm2.initialized = True

    def cache_inv_conv(self):
        for i in range(len(self.glow_modules)):
            module_name = self.glow_modules[i].__class__.__name__
            if module_name == "InvertibleConvolution":
                self.glow_modules[i].cache_inv_conv()

    def reset_cache_conv(self):
        for i in range(len(self.glow_modules)):
            module_name = self.glow_modules[i].__class__.__name__
            if module_name == "InvertibleConvolution":
                self.glow_modules[i].reset_cache_conv()

    def compile(self, config=None, optimizer=None, loss_fn=None, run_eagerly=None):
        # setting up optimizer and learning rate scheduler
        self.config = config
        if optimizer:
            self.train()
            self.optimizer = optimizer
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.get("lr_factor", 0.99),
                patience=config.lr_patience,
                verbose=True,
                min_lr=config.lr_min,
            )
        if loss_fn:
            if loss_fn == "nll":
                self.monitor_loss = loss_fn
                self.loss_fn = self.nll_loss
            else:
                raise NotImplementedError

        if not run_eagerly:
            # requires PyTorch 2.0 (optional)
            if int(torch.__version__[0]) >= 2:
                try:
                    self = torch.compile(self)
                except Exception as e:
                    print(e)
            else:
                print(
                    f"Cannot compile torch model with Pytorch {torch.__version__}."
                    " Please upgrade to PyTorch 2.0 or higher"
                )

    def fit(self, dataset, epochs, callbacks=None, steps_per_epoch=None, **kwargs):
        # starting training code here
        if callbacks is None:
            callbacks = []

        for cb in callbacks:
            cb.on_train_begin()
        print(steps_per_epoch)
        if steps_per_epoch is None:
            n_batches = len(dataset)
        else:
            n_batches = steps_per_epoch

        print("+-" * 10, "starting training", "-+" * 10)
        global_step = 0
        global_loss = []
        warmup_completed = False
        for i in range(epochs):
            t0 = time.time()
            wandb.log({"epoch": i})
            for cb in callbacks:
                cb.on_epoch_begin(i)
            for j, data in zip(range(n_batches), dataset):
                for cb in callbacks:
                    cb.on_batch_begin(j)
                self.optimizer.zero_grad()
                self.zero_grad()
                # loading batch
                data = tf_tensor_to_torch(data)
                x = data.to(device=self.device)
                if self.config.image_range is not None:
                    # pre-processing data
                    x = self.preprocess(x * 255)
                # computing loss: "nll"
                n, c, h, w = x.size()
                nll, logdet, logpz, z_mu, z_std = self.loss_fn(x)
                # skipping first batch due to data dependant initialization (if not initialized)
                if global_step == 0:
                    global_step += 1
                    continue
                # backpropogating loss and gradient clipping
                nll.backward()
                torch.nn.utils.clip_grad_value_(self.parameters(), self.config.clipnorm)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
                # linearly increase learning rate till warmup_iter upto args.lr
                if global_step <= self.config.warmup_iter:
                    warmup_lr = self.config.lr / self.config.warmup_iter * global_step
                    for params in self.optimizer.param_groups:
                        params["lr"] = warmup_lr
                # taking optimizer step
                self.optimizer.step()
                # learning rate scheduling after warm up iterations
                if global_step > self.config.warmup_iter:
                    self.lr_scheduler.step(nll)
                    if not warmup_completed:
                        if self.config.warmup_iter == 0:
                            print("no model warming...")
                        else:
                            print("\nwarm up completed")
                    warmup_completed = True

                # eta
                torch.cuda.synchronize()
                t1 = time.time()
                elapsed = t1 - t0
                eta = int(elapsed / (j + 1) * (n_batches - (j + 1)))
                eta = str(datetime.timedelta(seconds=eta))

                # printing training metrics
                print(
                    f"\repoch={i}..nll={nll.item():.2f}.."
                    f"logdet={logdet:.2f}..logpz={logpz:.2f}.."
                    f"mu={z_mu:.2f}..std={z_std:.2f}.."
                    f"gradnorm={grad_norm:.2f}..eta={eta}",
                    end="\r",
                )

                if j % self.config.log_freq == 0:
                    wandb.log(
                        {
                            "nll": nll.item(),
                            "logdet": logdet,
                            "logpz": logpz,
                            "z_mu": z_mu,
                            "z_std": z_std,
                            "grad_norm": grad_norm,
                        }
                    )

                global_step = global_step + 1
                global_loss.append(nll.item())

                for cb in callbacks:
                    cb.on_batch_end(j)
            for cb in callbacks:
                cb.on_epoch_end(i)
        for cb in callbacks:
            cb.on_train_end()

    def sample(self, z=None, shape=None, **kwargs):
        if z is None:
            z = self.get_latent_vector(shape[0])
        with torch.no_grad():
            samples = self.forward(z, reverse=True)
            samples = self.postprocess(samples)
            if self.config.image_range is not None:
                samples = torch.clip(samples, *self.config.image_range)
            samples = convert_torch_tensor(samples)
        return samples

    def get_latent_vector(self, batch_size):
        std = 0.8
        if not hasattr(self, "sizes"):
            _ = self.forward(
                self.preprocess(
                    torch.zeros([batch_size, *self.image_shape]).to(self.device)
                )
            )

        _, z_sample = self.generate_z(n=batch_size, mu=0, std=std, to_torch=True)
        return z_sample

    def summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Total number of params: {total_params}")
        print(f"Total number of trainable params: {total_trainable_params}")


if __name__ == "__main__":
    size = (16, 3, 64, 64)
    images = sio.imread_collection("./images/*.png")
    x = np.array([img.astype("float") / 255 for img in images]).transpose([0, 3, 1, 2])
    x = torch.tensor(x, device=device, dtype=torch.float, requires_grad=True)
    logdet = torch.tensor(0.0, requires_grad=False, device=device, dtype=torch.float)

    with torch.no_grad():
        glow = Glow(
            (3, 64, 64),
            K=32,
            L=4,
            coupling="affine",
            nn_init_last_zeros=True,
            device=device,
        )
        z, logdet, actloss = glow(x, logdet=logdet, reverse=False)
        x_rev = glow(z, reverse=True)
    print(torch.norm(x_rev - x).item())
    reconstructed = x_rev.data.cpu().numpy().transpose([0, 2, 3, 1])
    sio.imshow_collection(images)
    sio.imshow_collection(reconstructed)
