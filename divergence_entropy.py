import numpy as np

'''
defino entropias de Shannon, Renyi y Tsallis, y sus versiones normalizadas
(no puedo hacer una normalizada generalizada ya que se calculan distinto
'''
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


# def entropy_from_dataframe(df, m=3, tau=1, alpha=2.0, q=2.0):
#     rows = []

#     for col in df.columns:
#         x = df[col].values

#         _, counts = ordinal_patterns(x, emb_dim=m, emb_lag=tau)
#         prob = counts / counts.sum()

#         rows.append({
#             "image": col,
#             "H_shannon": shannon_entropy_normalized(prob),
#             "H_renyi": renyi_entropy_normalized(prob, alpha),
#             "H_tsallis": tsallis_entropy_normalized(prob, q)
#         })

#     return pd.DataFrame(rows)


def jensen_shannon_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    assert p.shape == q.shape
    
    m = 0.5 * (p + q)

    return shannon_entropy(m) - 0.5 * shannon_entropy(p) - 0.5 * shannon_entropy(q)

def jensen_shannon_divergence_normalized(prob):
    k = len(prob)

    # equilibrium
    pe = np.ones(k) / k

    # delta (perfect order)
    p_delta = np.zeros(k)
    p_delta[0] = 1.0

    D = jensen_shannon_divergence(prob, pe)
    Dmax = jensen_shannon_divergence(p_delta, pe)

    return D / Dmax


def renyi_kl_divergence(p, q, alpha):
    """
    Rényi–Kullback divergence K_q^(R)(p || q)
    """
    mask = (p > 0) & (q > 0)
    p = p[mask]
    q = q[mask]

    return (1.0 / (alpha - 1.0)) * np.log(
        np.sum(p**alpha * q**(1.0 - alpha))
    )

def jensen_renyi_divergence(prob, alpha=2.0):
    k = len(prob)
    pe = np.full(k, 1.0 / k)

    m = 0.5 * (prob + pe)

    return 0.5 * (
        renyi_kl_divergence(prob, m, alpha)
        + renyi_kl_divergence(pe, m, alpha)
    )

def jensen_renyi_divergence_normalized(prob, alpha=2.0):
    k = len(prob)

    # delta distribution
    p_delta = np.zeros(k)
    p_delta[0] = 1.0

    D = jensen_renyi_divergence(prob, alpha)
    Dmax = jensen_renyi_divergence(p_delta, alpha)

    return D / Dmax

def tsallis_kl_divergence(p, q, alpha):
    """
    Tsallis–Kullback divergence K_q^(T)(p || q)
    """
    mask = (p > 0) & (q > 0)
    p = p[mask]
    q = q[mask]

    return (1.0 / (alpha - 1.0)) * np.sum(
        p**alpha * (q**(1.0 - alpha) - p**(1.0 - alpha))
    )

def jensen_tsallis_divergence(prob, q=2.0):
    k = len(prob)
    pe = np.full(k, 1.0 / k)

    m = 0.5 * (prob + pe)

    return 0.5 * (
        tsallis_kl_divergence(prob, m, q)
        + tsallis_kl_divergence(pe, m, q)
    )

def jensen_tsallis_divergence_normalized(prob, q=2.0):
    k = len(prob)

    # delta distribution
    p_delta = np.zeros(k)
    p_delta[0] = 1.0

    D = jensen_tsallis_divergence(prob, q)
    Dmax = jensen_tsallis_divergence(p_delta, q)

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
