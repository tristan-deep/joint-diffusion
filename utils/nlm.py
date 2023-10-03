"""Conventional denoising methods NLM and BM3D
"""
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from bm3d import BM3DStages, bm3d
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import resize
from skimage.util import random_noise


def NLM(image, t, f):
    """Non-local means denoising."""
    img = np.clip(img_as_float(image).astype(np.float64), a_min=0, a_max=1)
    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    print(f"estimated noise standard deviation = {sigma_est}")

    patch_kw = dict(
        patch_size=f,  # 5x5 patches
        patch_distance=t,  # 13x13 search area
        multichannel=True,
    )

    # fast algorithm, sigma provided
    img_denoised = denoise_nl_means(
        img, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw
    )

    return img_denoised.astype(np.float32)


def OBNLM(image, t, f, h):
    """Python Code for the Non local filter (Speckle)
    proposed for P. Coupe, P. Hellier, C. Kervrann and C. Barillot in
    "Nonlocal Means-Based Speckle Filtering for Ultrasound Images"

    Args:
        image (ndarray): Input image.
        t (int): Search window.
        f (int): Similarity window.
        h (float): Degree of filtering.

    Returns:
        img_denoised (ndarray): Denoised image.

    """

    img = np.clip(img_as_float(image).astype(np.float64), a_min=0, a_max=1)
    [m, n] = img.shape
    img_denoised = np.zeros((m, n), dtype=np.float32)
    h = h * h

    # Normalization
    kernel = np.zeros((2 * f + 1, 2 * f + 1))
    for d in range(1, f + 1):
        value = 1.0 / ((2 * d + 1) * (2 * d + 1))
        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                kernel[f - i, f - j] = kernel[f - i, f - j] + value

    kernel = kernel / f
    kernel = kernel / sum(sum(kernel))
    vkernel = np.reshape(kernel, (2 * f + 1) * (2 * f + 1))

    # padding
    pdimg = np.pad(img, ((f, f)), mode="symmetric")

    # Denoising
    for i in tqdm.tqdm(range(0, m), desc="obnlm"):
        for j in range(0, n):
            i1 = i + f
            j1 = j + f

            W1 = pdimg[range(i1 - f, i1 + f + 1), :]
            W1 = W1[:, range(j1 - f, j1 + f + 1)]

            wmax = 0
            average = 0
            sweight = 0

            rmin = max(i1 - t, f)
            rmax = min(i1 + t, m + f - 1)
            smin = max(j1 - t, f)
            smax = min(j1 + t, n + f - 1)

            # Find similarity between neighborhoods W1(center) and W2(surrounding)
            for r in range(rmin, rmax + 1):
                for s in range(smin, smax + 1):
                    if (r == i1) and (s == j1):
                        continue
                    W2 = pdimg[range(r - f, r + f + 1), :]
                    W2 = W2[:, range(s - f, s + f + 1)] + 1e-6
                    # Use Pearson Distance
                    temp = np.reshape(
                        ((np.square(W1 - W2)) / W2), (2 * f + 1) * (2 * f + 1)
                    )
                    d = np.dot(vkernel, temp)
                    # print(f'd: {d}, h: {h}')
                    w = np.exp(-d / h)
                    if w > wmax:
                        wmax = w

                    sweight = sweight + w
                    average = average + w * pdimg[r, s]

            average = average + wmax * pdimg[i1, j1]

            # Calculation of the weight
            sweight = sweight + wmax

            # Compute value of the denoised pixel
            if sweight > 0:
                # print(f'average: {average}, sweight: {sweight}')
                img_denoised[i, j] = average / sweight
            else:
                img_denoised[i, j] = img[i, j]

    return img_denoised.astype(np.float32)


def BM3D(image, stddev):
    """Block matching 3D denoising."""
    denoised_image = bm3d(
        image,
        stddev,
        stage_arg=BM3DStages.ALL_STAGES,
    )
    return denoised_image


if __name__ == "__main__":
    # For Testing
    img = data.camera()

    # Reduce the test runtime
    img = resize(img, (100, 100))

    img_noisy = random_noise(img, mode="speckle")

    # NLM Denoising
    img_denoised_nlm = NLM(img_noisy, 6, 5)

    # OBNLM Denoising
    img_denoised_obnlm = OBNLM(img_noisy, 5, 2, 0.1)

    # BM3D Denoising
    img_denoised_bm3d = BM3D(img_noisy, 0.1)

    images = [
        img,
        img_noisy,
        img_denoised_nlm,
        img_denoised_obnlm,
        img_denoised_bm3d,
    ]

    results = []
    for di in images:
        psnr = peak_signal_noise_ratio(img, di)
        results.append(psnr)

    # Show results
    titles = ["original", "noisy", "NLM", "OBNLM", "BM3D"]
    fig, axs = plt.subplots(1, len(images), figsize=(13, 3))

    for img, ax, title, result in zip(images, axs, titles, results):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{title}: PSNR={result:0.2f}")

    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
