import math
from pathlib import Path

import numpy as np
import pywt
import scipy.fftpack as fftpack
from cvxopt import blas, div, lapack, matrix, mul, solvers, spdiag, sqrt
from sklearn.linear_model import Lasso
from tqdm import trange


def lasso_wavelet_estimator(hparams):
    """LASSO with Wavelet"""

    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        W = wavelet_basis(s=hparams.size, n_channels=hparams.n_channels)
        WA = np.dot(W, A_val)
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = solve_lasso(WA, y_val, hparams)
            x_hat = np.dot(z_hat, W)
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch

    return estimator


class lasso_dct_estimator:
    def __init__(self, hparams, A):
        self.hparams = hparams
        self.A_DCT = A
        for i in range(A.shape[1]):
            self.A_DCT[:, i] = vec(
                [dct2(channel) for channel in devec(self.A_DCT[:, i], s=hparams.size)],
                s=hparams.size,
            )

    def __call__(self, y_batch_val, hparams):
        x_hat_batch = []
        for j in trange(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = solve_lasso(self.A_DCT, y_val, hparams)
            x_hat = vec(
                [idct2(channel) for channel in devec(z_hat, s=hparams.size)],
                s=hparams.size,
            ).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch


def solve_lasso(A_val, y_val, hparams):
    if hparams.lasso_solver == "sklearn":
        lasso_est = Lasso(alpha=hparams.lambda_coeff)
        lasso_est.fit(A_val.T, y_val.reshape(-1))
        x_hat = lasso_est.coef_
        x_hat = np.reshape(x_hat, [-1])
    if hparams.lasso_solver == "cvxopt":
        A_mat = matrix(A_val.T)
        y_mat = matrix(y_val)
        x_hat_mat = l1regls(A_mat, y_mat)
        x_hat = np.asarray(x_hat_mat)
        x_hat = np.reshape(x_hat, [-1])
    return x_hat


def wavelet_basis(s=64, n_channels=3):
    assert s in [64, 128], "Size parameter must be either 64 or 128"
    assert n_channels in [1, 3], "n_channels must be 1 or 3"
    file = Path("./utils/wavelets") / f"wavelet_basis_{s}.npy"
    if file.exists():
        W_ = np.load(file)
    else:
        W_ = generate_wavelet_basis(s)

    # W_ initially has shape (4096,64,64), i.e. 4096 64x64 images
    # reshape this into 4096x4096, where each row is an image
    # take transpose to make columns images
    W_ = W_.reshape((s * s, s * s))
    if n_channels == 3:
        W = np.zeros((s * s * n_channels, s * s * n_channels))
        W[0::3, 0::3] = W_
        W[1::3, 1::3] = W_
        W[2::3, 2::3] = W_
    else:
        W = W_
    return W


def generate_wavelet_basis(s=64):
    """Generate wavelet basis"""
    x = np.zeros((s, s))
    coefs = pywt.wavedec2(x, "db1")
    n_levels = len(coefs)
    basis = []
    for i in range(n_levels):
        coefs[i] = list(coefs[i])
        n_filters = len(coefs[i])
        for j in range(n_filters):
            for m in range(coefs[i][j].shape[0]):
                try:
                    for n in range(coefs[i][j].shape[1]):
                        coefs[i][j][m][n] = 1
                        temp_basis = pywt.waverec2(coefs, "db1")
                        basis.append(temp_basis)
                        coefs[i][j][m][n] = 0
                except IndexError:
                    coefs[i][j][m] = 1
                    temp_basis = pywt.waverec2(coefs, "db1")
                    basis.append(temp_basis)
                    coefs[i][j][m] = 0

    basis = np.array(basis)
    folder = Path("./utils/wavelets")
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f"wavelet_basis_{s}.npy", basis)
    return basis


def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm="ortho").T, norm="ortho")


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm="ortho").T, norm="ortho")


def vec(channels, s=64):
    image = np.zeros((s, s, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector, s=64):
    image = np.reshape(vector, [s, s, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def l1regls(A, b):
    """Returns the solution of l1-norm regularized least-squares problem

    minimize || A*x - b ||_2^2  + || x ||_1.

    http://cvxopt.org/examples/mlbook/l1regls.html#
    """

    m, n = A.size
    q = matrix(1.0, (2 * n, 1))
    q[:n] = -2.0 * A.T * b

    def P(u, v, alpha=1.0, beta=0.0):
        """
        v := alpha * 2.0 * [ A'*A, 0; 0, 0 ] * u + beta * v
        """
        v *= beta
        v[:n] += alpha * 2.0 * A.T * (A * u[:n])

    def G(u, v, alpha=1.0, beta=0.0, trans="N"):
        """
        v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
        """

        v *= beta
        v[:n] += alpha * (u[:n] - u[n:])
        v[n:] += alpha * (-u[:n] - u[n:])

    h = matrix(0.0, (2 * n, 1))

    # Customized solver for the KKT system
    #
    #     [  2.0*A'*A  0    I      -I     ] [x[:n] ]     [bx[:n] ]
    #     [  0         0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
    #     [  I        -I   -D1^-1   0     ] [zl[:n]]     [bzl[:n]]
    #     [ -I        -I    0      -D2^-1 ] [zl[n:]]     [bzl[n:]]
    #
    # where D1 = W['di'][:n]**2, D2 = W['di'][:n]**2.
    #
    # We first eliminate zl and x[n:]:
    #
    #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] =
    #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] +
    #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
    #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:]
    #
    #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
    #         - (D2-D1)*(D1+D2)^-1 * x[:n]
    #
    #     zl[:n] = D1 * ( x[:n] - x[n:] - bzl[:n] )
    #     zl[n:] = D2 * (-x[:n] - x[n:] - bzl[n:] ).
    #
    # The first equation has the form
    #
    #     (A'*A + D)*x[:n]  =  rhs
    #
    # and is equivalent to
    #
    #     [ D    A' ] [ x:n] ]  = [ rhs ]
    #     [ A   -I  ] [ v    ]    [ 0   ].
    #
    # It can be solved as
    #
    #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
    #     x[:n] = D^-1 * ( rhs - A'*v ).

    S = matrix(0.0, (m, m))
    Asc = matrix(0.0, (m, n))
    v = matrix(0.0, (m, 1))

    def Fkkt(W):
        # Factor
        #
        #     S = A*D^-1*A' + I
        #
        # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**-2, D2 = d[n:]**-2.

        d1, d2 = W["di"][:n] ** 2, W["di"][n:] ** 2

        # ds is square root of diagonal of D
        ds = math.sqrt(2.0) * div(mul(W["di"][:n], W["di"][n:]), sqrt(d1 + d2))
        d3 = div(d2 - d1, d1 + d2)

        # Asc = A*diag(d)^-1/2
        Asc = A * spdiag(ds**-1)

        # S = I + A * D^-1 * A'
        blas.syrk(Asc, S)
        S[:: m + 1] += 1.0
        lapack.potrf(S)

        def g(x, y, z):
            x[:n] = 0.5 * (
                x[:n]
                - mul(d3, x[n:])
                + mul(d1, z[:n] + mul(d3, z[:n]))
                - mul(d2, z[n:] - mul(d3, z[n:]))
            )
            x[:n] = div(x[:n], ds)

            # Solve
            #
            #     S * v = 0.5 * A * D^-1 * ( bx[:n] -
            #         (D2-D1)*(D1+D2)^-1 * bx[n:] +
            #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
            #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:] )

            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)

            # x[:n] = D^-1 * ( rhs - A'*v ).
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans="T")
            x[:n] = div(x[:n], ds)

            # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]
            x[n:] = div(x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1 + d2) - mul(
                d3, x[:n]
            )

            # zl[:n] = D1^1/2 * (  x[:n] - x[n:] - bzl[:n] )
            # zl[n:] = D2^1/2 * ( -x[:n] - x[n:] - bzl[n:] ).
            z[:n] = mul(W["di"][:n], x[:n] - x[n:] - z[:n])
            z[n:] = mul(W["di"][n:], -x[:n] - x[n:] - z[n:])

        return g

    return solvers.coneqp(P, q, G, h, kktsolver=Fkkt)["x"][:n]
