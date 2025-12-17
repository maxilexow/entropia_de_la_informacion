import numpy as np


def hilbert_index_to_xy(t: int, n: int) -> tuple[int, int]:
    """
    Convert Hilbert curve index to (x, y).

    Parameters
    ----------
    t : int
        Hilbert index, 0 <= t < n*n
    n : int
        Grid size (must be power of 2)

    Returns
    -------
    (x, y) : tuple[int, int]
    """
    x = y = 0
    s = 1
    tt = t

    while s < n:
        rx = 1 & (tt // 2)
        ry = 1 & (tt ^ rx)

        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x

        x += s * rx
        y += s * ry

        tt //= 4
        s *= 2

    return x, y


def image_to_hilbert_timeseries(img: np.ndarray) -> np.ndarray:
    """
    Convert a square 2^k x 2^k image to a 1D time series
    using Hilbert curve ordering.

    Parameters
    ----------
    img : np.ndarray
        2D array (H x W), H == W == 2^k

    Returns
    -------
    ts : np.ndarray
        1D array of length H*W
    """
    assert img.ndim == 2
    n, m = img.shape
    assert n == m and (n & (n - 1)) == 0, "Size must be 2^k x 2^k"

    ts = np.empty(n * n, dtype=img.dtype)

    for t in range(n * n):
        x, y = hilbert_index_to_xy(t, n)
        ts[t] = img[y, x]  # row=y, col=x

    return ts

import itertools

def ordinal_patterns(x, emb_dim=3, emb_lag=1):
    """
    Calcula las permutaciones ordinales (Bandt & Pompe, 2002).

    Parámetros
    ----------
    x : array-like
        Serie temporal.
    emb_dim : int
        Dimensión del embedding (m).
    emb_lag : int
        Retardo entre muestras (τ).

    Retorna
    -------
    patterns : np.ndarray
        Índices de permutación observados (0 ... m!-1).
    counts : np.ndarray
        Frecuencia absoluta de cada patrón (longitud = m!).
    """
    x = np.asarray(x)
    n = len(x)
    if n < (emb_dim - 1) * emb_lag + 1:
        raise ValueError("Serie demasiado corta para el embedding especificado.")

    # Todas las permutaciones posibles
    perms = list(itertools.permutations(range(emb_dim)))
    perm_dict = {p: i for i, p in enumerate(perms)}
    patterns = np.zeros(n - (emb_dim - 1) * emb_lag, dtype=int)

    # Recorre la serie y asigna cada patrón
    for i in range(len(patterns)):
        window = x[i : i + emb_dim * emb_lag : emb_lag]
        ranks = tuple(np.argsort(window))
        patterns[i] = perm_dict[ranks]

    # Conteo de frecuencia
    counts = np.bincount(patterns, minlength=len(perms))
    return patterns, counts

from scipy.signal import welch

def autocorrelation(x, max_lag=None):
    """
    Autocorrelación normalizada hasta max_lag.
    """
    x = np.asarray(x)
    x -= np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:]
    corr /= corr[0]  # normaliza
    if max_lag is not None:
        corr = corr[:max_lag]
    return corr

def compute_psd(x, fs=1.0):
    """
    Densidad espectral de potencia (PSD) usando método de Welch.
    Retorna frecuencias y densidad en escala logarítmica.
    """
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)//2))
    return f, Pxx


from itertools import product
import pandas as pd 
import matplotlib.pyplot as plt
import math

def plot_entropy_complexity_planes(
    subset_df,
    m,
    tau,
    spec,
):
    H = subset_df[spec["H_col"]].values
    C = subset_df[spec["C_col"]].values

    fig, ax = plt.subplots(figsize=(6, 5))

    # ---- data first (sets zoom) ----
    ax.scatter(H, C, s=60, zorder=3, label="Images")

    # ---- bounds ----
    N = math.factorial(m)
    bounds = complexity_bounds(
        N=N,
        entropy_func=spec["H_func"],
        divergence_func=spec["D_func"],
    )

    plot_HC_bounds(ax, bounds)

    ax.set_xlabel("Normalized entropy H")
    ax.set_ylabel("Statistical complexity C")
    ax.set_title(f"{spec['label']} H–C plane (m={m}, τ={tau})")

    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()



def complexity_bounds(
    N,
    entropy_func,
    divergence_func,
    n_points=400,
    eps=1e-12
):
    # C_max
    Hmax, Cmax = [], []
    for p in np.linspace(1/N, 1, n_points):
        prob = np.full(N, eps)
        prob[0] = p
        prob[1:] = (1 - p) / (N - 1)
        prob /= prob.sum()

        H = entropy_func(prob)
        C = H * divergence_func(prob)
        Hmax.append(H)
        Cmax.append(C)

    # C_min
    Hmin, Cmin = [], []
    for p in np.linspace(0, 1, n_points):
        prob = np.full(N, eps)
        prob[0] = p
        prob[1] = 1 - p
        prob /= prob.sum()

        H = entropy_func(prob)
        C = H * divergence_func(prob)
        Hmin.append(H)
        Cmin.append(C)

    print("Cmin max:", np.max(Cmin))
    print("Cmax max:", np.max(Cmax))
    print("Cmin min:", np.min(Cmin))
    print("Cmax min:", np.min(Cmax))

    return (
        np.array(Hmin), np.array(Cmin),
        np.array(Hmax), np.array(Cmax),
    )


def plot_HC_bounds(ax, bounds):
    """
    Plot C_min and C_max bounds on an existing axis.
    """
    Hmin, Cmin, Hmax, Cmax = bounds

    ax.plot(Hmax, Cmax, color='red', lw=2, label=r"$C_{\max}$")
    ax.plot(Hmin, Cmin, color='blue',  lw=2, label=r"$C_{\min}$")
