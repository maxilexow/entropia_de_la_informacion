import numpy as np


def shannon_entropy(prob):
    prob = prob[prob > 0]
    return -np.sum(prob * np.log(prob))


def shannon_entropy_normalized(prob):
    H = shannon_entropy(prob)
    Hmax = np.log(len(prob))
    return H / Hmax

def renyi_entropy(prob, alpha=2.0):
    prob = prob[prob > 0]
    return (1.0 / (1.0 - alpha)) * np.log(np.sum(prob ** alpha))


def renyi_entropy_normalized(prob, alpha=2.0):
    H = renyi_entropy(prob, alpha)
    Hmax = np.log(len(prob))
    return H / Hmax

def tsallis_entropy(prob, q=2.0):
    prob = prob[prob > 0]
    return (1.0 - np.sum(prob ** q)) / (q - 1.0)


def tsallis_entropy_normalized(prob, q=2.0):
    H = tsallis_entropy(prob, q)
    Hmax = (1.0 - len(prob) ** (1.0 - q)) / (q - 1.0)
    return H / Hmax

def entropy_from_dataframe(df, m=3, tau=1, alpha=2.0, q=2.0):
    rows = []

    for col in df.columns:
        x = df[col].values

        _, counts = ordinal_patterns(x, emb_dim=m, emb_lag=tau)
        prob = counts / counts.sum()

        rows.append({
            "image": col,
            "H_shannon": shannon_entropy_normalized(prob),
            "H_renyi": renyi_entropy_normalized(prob, alpha),
            "H_tsallis": tsallis_entropy_normalized(prob, q)
        })

    return pd.DataFrame(rows)


def jensen_shannon_divergence(p, q):
    """
    Jensen-Shannon divergence between two PDFs.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)

    def shannon(x):
        x = x[x > 0]
        return -np.sum(x * np.log(x))

    return shannon(m) - 0.5 * shannon(p) - 0.5 * shannon(q)

def jensen_shannon_divergence_normalized(p):
    n = len(p)
    pe = np.full(n, 1.0 / n)
    D = jensen_shannon_divergence(p, pe)
    return D / np.log(2.0)

def jensen_renyi_divergence(prob, alpha=2.0):
    n = len(prob)
    pe = np.full(n, 1.0 / n)

    m = 0.5 * (prob + pe)

    Hm = renyi_entropy(m, alpha)
    Hp = renyi_entropy(prob, alpha)
    He = renyi_entropy(pe, alpha)

    return Hm - 0.5 * (Hp + He)

def jensen_renyi_divergence_normalized(prob, alpha=2.0):
    n = len(prob)
    D = jensen_renyi_divergence(prob, alpha)

    Dmax = (
        1.0 / (1.0 - alpha)
        * np.log(
            ((1.0 + n ** (1.0 - alpha)) ** alpha)
            / (2.0 ** alpha * n ** (1.0 - alpha))
        )
    )

    return D / Dmax

def jensen_tsallis_divergence(prob, q=2.0):
    n = len(prob)
    pe = np.full(n, 1.0 / n)

    m = 0.5 * (prob + pe)

    Sm = tsallis_entropy(m, q)
    Sp = tsallis_entropy(prob, q)
    Se = tsallis_entropy(pe, q)

    return Sm - 0.5 * (Sp + Se)

def jensen_tsallis_divergence_normalized(prob, q=2.0):
    n = len(prob)
    D = jensen_tsallis_divergence(prob, q)

    Dmax = (
        1.0 / (q - 1.0)
        * (
            1.0
            - ((1.0 + n ** (1.0 - q)) / 2.0) ** q
            - 0.5 * (1.0 - n ** (1.0 - q))
        )
    )

    return D / Dmax

def complexity_shannon(prob):
    H = shannon_entropy_normalized(prob)
    D = jensen_shannon_divergence_normalized(prob)
    return H * D

def complexity_renyi(prob, alpha=2.0):
    H = renyi_entropy_normalized(prob, alpha)
    D = jensen_renyi_divergence_normalized(prob, alpha)
    return H * D


def complexity_tsallis(prob, q=2.0):
    H = tsallis_entropy_normalized(prob, q)
    D = jensen_tsallis_divergence_normalized(prob, q)
    return H * D
