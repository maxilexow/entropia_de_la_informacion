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
    bounds=None,
    labels=None
):
    H = subset_df[spec["H_col"]].values
    C = subset_df[spec["C_col"]].values

    fig, ax = plt.subplots(figsize=(6, 5))

    # ---- Data points
    ax.scatter(H, C, s=60, color="tab:blue", zorder=3, label="Images")
    
    # --- force limits from scatter data
    ax.update_datalim(ax.scatter(H, C, s=60, zorder=3).get_offsets())
    ax.autoscale_view()
        
    # --- freeze limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # labels
    for h, c, lbl in zip(H, C, labels):
        ax.annotate(
            lbl,
            xy=(h, c),
            xytext=(5, 5),          # offset in points
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )
        
    # ---- HC bounds (optional)
    if bounds is not None:
        ax.plot(
            bounds["Hmax"],
            bounds["Cmax"],
            color="red",
            lw=2,
            label=r"$C_{\max}$",
            zorder=2
        )
        ax.plot(
            bounds["Hmin"],
            bounds["Cmin"],
            color="blue",
            lw=2,
            label=r"$C_{\min}$",
            zorder=2
        )
    # ax.set_xlim(0.8, 1.0)
    # ax.set_ylim(0, 0.1)
    ax.set_xlabel("Normalized entropy H")
    ax.set_ylabel("Statistical complexity C")
    ax.set_title(f"{spec['label']} H–C plane (m={m}, τ={tau})")

    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()

def Cmin_curve(k, entropy_func, divergence_func, n_points=400):
    H, C = [], []
    for p in np.linspace(1/k, 1.0, n_points):
        prob = np.full(k, (1 - p) / (k - 1))
        prob[0] = p

        h = entropy_func(prob)
        c = h * divergence_func(prob)

        H.append(h)
        C.append(c)

    return np.array(H), np.array(C)

def Cmax_curve(k, entropy_func, divergence_func, n_points=400):
    H_all, C_all = [], []

    for j in range(1, k):
        p_vals = np.linspace(1/(j+1), 1/j, n_points)

        for p in p_vals:
            prob = np.zeros(k)
            prob[:j] = p
            prob[j] = 1 - j*p

            h = entropy_func(prob)
            c = h * divergence_func(prob)

            H_all.append(h)
            C_all.append(c)

    return np.array(H_all), np.array(C_all)

