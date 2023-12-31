import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class InvertibleConvolution(nn.Module):
    def __init__(
        self,
        channels,
        device,
    ):
        super(InvertibleConvolution, self).__init__()

        self.W_size = [channels, channels, 1, 1]
        W_init = np.random.normal(0, 1, self.W_size[:-2])
        W_init = np.linalg.qr(W_init)[0].astype(np.float32)
        W_init = W_init.reshape(self.W_size)
        self.W = nn.Parameter(
            torch.tensor(W_init, dtype=torch.float, device=device, requires_grad=True)
        )
        self.inv_w = None
        self.use_cached_inv_w = False

        self.to(device)

    def forward(self, x, logdet=None, reverse=False):
        n, c, h, w = x.size()
        if not reverse:
            x = F.conv2d(x, self.W)
            detW = torch.slogdet(self.W.squeeze())[1]
            logdet = logdet + h * w * detW
            return x, logdet

        if reverse:
            # if eval, try to use cached inv_w
            if not self.training or self.use_cached_inv_w:
                if self.inv_w is None:
                    self.inv_w = (
                        torch.inverse(self.W.squeeze().double())
                        .float()
                        .view(self.W_size)
                    )
                inv_w = self.inv_w
            else:
                inv_w = (
                    torch.inverse(self.W.squeeze().double()).float().view(self.W_size)
                )
            x = F.conv2d(x, inv_w)
            # assert not np.isinf(x.mean().item()), "inf after 1x1 conv in reverse"
            # assert not np.isnan(x.mean().item()), "nan after 1x1 conv in reverse"
            return x

    def cache_inv_conv(self):
        self.use_cached_inv_w = True
        self.inv_w = torch.inverse(self.W.squeeze().double()).float().view(self.W_size)

    def reset_cache_conv(self):
        self.use_cached_inv_w = False
        self.inv_w = None


if __name__ == "__main__":
    size = (16, 64, 32, 32)
    invconv = InvertibleConvolution(64, device)
    opt = torch.optim.Adam(invconv.parameters(), lr=0.05)
    for i in range(1000):
        opt.zero_grad()
        x = torch.tensor(np.random.normal(0, 1, size), dtype=torch.float, device=device)
        logdet = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)
        y_true = x * 13
        y, logdet = invconv(x, logdet=logdet, reverse=False)
        x_rev = invconv(y, reverse=True)
        mse = torch.norm(y_true - y)
        mse.backward()
        #        logdet.backward()
        opt.step()
        mu = y.mean().item()
        std = y.std().item()
        loss_rev = torch.norm(x_rev - x).item()
        print(
            "\r mse=%0.3f | mu=%0.3f | std=%0.3f | rloss=%0.3f | logdet=%0.3f"
            % (mse.item(), mu, std, loss_rev, logdet.item()),
            end="\r",
        )
