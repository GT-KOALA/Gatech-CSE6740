---
semester: fall-2026
lecture_number: "05"
slot: optional-2
slot_order: 4
role: optional_topic
role_label: Optional topic
tab_title: Extension II
assignee: ancasfiam
issue: 11
---


## Loading the data

Lets start by loading in the Wine dataset and investigating the information it provides:

```python
import numpy as np
from ucimlrepo import fetch_ucirepo
 
# UCI repository id 109 = Wine
wine = fetch_ucirepo(id=109)
 
X_df = wine.data.features     
y_df = wine.data.targets     
 
X_raw = X_df.to_numpy(dtype=float)
y_true = y_df.to_numpy().ravel()          # Not used in EM
 
print(X_raw.shape)
print(list(X_df.columns))
```
 
```
(178, 13) [1, 2, 3]
['Alcohol', 'Malicacid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue',
 '0D280_0D315_of_diluted_wines', 'Proline']
```


The dataset presents 178 wine samples with 13 features. The labels are not used to train the model in our case.  Gaussian mixture is an unsupervised
density model, so it sees only the feature matrix. 

```python
import pandas as pd
 
print(X_df[["Hue", "Malicacid", "Alcohol", "Magnesium", "Proline"]].head())
```

```
    Hue  Malicacid  Alcohol  Magnesium  Proline
0  1.04       1.71    14.23      127.0   1065.0
1  1.05       1.78    13.20      100.0   1050.0
2  1.03       2.36    13.16      101.0   1185.0
3  0.86       1.95    14.37      113.0   1480.0
4  1.04       2.59    13.24      118.0    735.0
```

## Standardizing

Looking at a couple features raises an obvious problem: the features are all in different measurement types which creates very different orders of magnitude. The next step is to standardize the numerical values with mean 0 and variance 1. 


```python
from sklearn.preprocessing import StandardScaler
 
X = StandardScaler().fit_transform(X_raw)      
```


## Reducing to two dimensions

Then, we turn to two principal components for vizualization:


```python
from sklearn.decomposition import PCA
 
pca = PCA(n_components=2)
Z = pca.fit_transform(X)
 
print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
```
 
```
[0.36198848 0.19207317] 0.5540616518332126
```


The first two components carry 36.20% and 19.21% of the total variance, 55.41% together. Nearly half the variance is discarded, so it must be noted that the two-dimensional picture is just a projection of the data. However, the alternative version proposed, fitting the GMM in the original standardized feature space, simply would not work on a dataset of this sample size. Each covariance matrix would need $91$ dimensions, so three components already need more covariance parameters than we have data points.




## Initializing


For the three component Gaussian Mixture implementation, note we have the following parameters to pick:

| parameter | shape | constraint | free parameters |
|---|---|---|---|
| mixing proportions $$\pi_k$$ | $$K$$ | $$\pi_k \ge 0$$, $$\sum_k \pi_k = 1$$ | $$K - 1$$ |
| means $$\mu_k$$ | $$K \times d$$ | none | $$Kd$$ |
| covariances $$\Sigma_k$$ | $$K \times d \times d$$ | symmetric positive definite | $$K \, d(d+1)/2$$ |

There are $17$ free parameters here, which is a much more comfortable ratio for the $178$ data points. EM is only guaranteed to reach a local maximum, so these starting values determine which solution we converge to. The initialization below is simple and fully explicit.

```python
SEED = 6740
K = 3
REG = 1e-6
 
def init_params(X, K, rng):
    n, d = X.shape
    pi = np.full(K, 1.0 / K)                          # uniform over components
    mu = X[rng.choice(n, size=K, replace=False)]      # K distinct data points
    Sigma = np.array([np.cov(X, rowvar=False) + REG * np.eye(d)
                      for _ in range(K)])             # global covariance
    return pi, mu, Sigma.copy()
 
rng = np.random.default_rng(SEED)
pi, mu, Sigma = init_params(Z, K, rng)
 
print("pi =", pi.round(4))
print("mu =\n", mu.round(4))
print("Sigma[0] =\n", Sigma[0].round(4))
```
 
```
pi = [0.3333 0.3333 0.3333]
mu =
 [[ 0.9575 -2.2235]
 [ 2.2248  1.8752]
 [ 2.5109  0.9181]]
Sigma[0] =
 [[ 4.7324 -0.    ]
 [-0.       2.5111]]
```



## The EM steps
 
`SEED = 6740` above fixes every number on this page. 
 
**E-step** 
 
$$\tau_k^i = p\left(z^i = k \mid D, \mu, \Sigma\right) = \frac{\pi_k \, \mathcal{N}(x^i \mid \mu_k, \Sigma_k)}{\sum_{k'=1}^{K} \pi_{k'} \, \mathcal{N}(x^i \mid \mu_{k'}, \Sigma_{k'})}$$
 
**M-step** 
 
$$\pi_k = \frac{\sum_i \tau_k^i}{n}, \qquad \mu_k = \frac{\sum_i \tau_k^i x^i}{\sum_i \tau_k^i}, \qquad \Sigma_k = \frac{\sum_i \tau_k^i \left(x^i - \mu_k\right)\left(x^i - \mu_k\right)^\top}{\sum_i \tau_k^i}$$
 
In code `Nk` is the shared denominator $$\sum_i \tau_k^i$$. These three lines are the closed-form
maximizers of the analytical lower bound,
 
$$l(\theta; D, \tau) = \sum_{i=1}^{n}\sum_{k=1}^{K} \tau_k^i \left[\log \pi_k - \frac{1}{2}\left(x^i - \mu_k\right)^\top \Sigma_k^{-1}\left(x^i - \mu_k\right) - \frac{1}{2}\log|\Sigma_k| - c\right] + H(q)$$
 
so `m_step` performs $$\theta^{t+1} = \arg\max_\theta l(\theta, D; q)$$.
 
```python
from scipy.special import logsumexp
 
def log_gaussian(X, mu, Sigma):
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)                  # Sigma = L L^T
    diff = X - mu
    sol = np.linalg.solve(L, diff.T)               # solve L s = (x - mu)
    maha = np.sum(sol ** 2, axis=0)                # (x-mu)^T Sigma^-1 (x-mu)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))     # log |Sigma|
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
 
 
def e_step(X, pi, mu, Sigma):
    n = X.shape[0]
    K = len(pi)
 
    # log of the numerator, pi_k * N(x^i | mu_k, Sigma_k), for every i and k
    log_w = np.zeros((n, K))
    for k in range(K):
        log_w[:, k] = np.log(pi[k]) + log_gaussian(X, mu[k], Sigma[k])
 
    # divide by the sum over k, in log space
    tau = np.zeros((n, K))
    log_likelihood = 0.0
    for i in range(n):
        total = logsumexp(log_w[i, :])             # log p(x^i)
        tau[i, :] = np.exp(log_w[i, :] - total)
        log_likelihood += total
 
    return tau, log_likelihood
 
 
def m_step(X, tau, reg=REG):
    n, d = X.shape
    K = tau.shape[1]
    pi = np.zeros(K)
    mu = np.zeros((K, d))
    Sigma = np.zeros((K, d, d))
 
    for k in range(K):
        Nk = tau[:, k].sum()                       # sum_i tau_k^i
        pi[k] = Nk / n
 
        for i in range(n):                         # weighted mean
            mu[k] += tau[i, k] * X[i]
        mu[k] /= Nk
 
        for i in range(n):                         # weighted scatter
            diff = X[i] - mu[k]
            Sigma[k] += tau[i, k] * np.outer(diff, diff)
        Sigma[k] /= Nk
        Sigma[k] += reg * np.eye(d)                # regulizer, see below
 
    return pi, mu, Sigma
```
 
The regularizer $$\varepsilon I$$ with $$\varepsilon = 10^{-6}$$ matters when a component's
responsibility concentrates on very few or near-collinear points. Since $$\Sigma_k$$ becomes singular,
inversion directly fails, and the unregularized likelihood diverges to $$+\infty$$ as the component
collapses onto a single point. Limiting every eigenvalue at $$\varepsilon$$ from below prevents both.

For initialization, random data points as means are fully explicit, but this seed puts two of three means in the same region, so k-means initialization could converge faster and more consistently. For numerical stability, the Cholesky factorization replaces the explicit $$\Sigma_k^{-1}$$ and $$|\Sigma_k|$$ above, and the E-step ratio is a logsumexp subtraction, so no exponential is ever formed at full scale.
 
**Convergence test** 

```python
TOL, MAX_ITER = 1e-6, 500
 
def fit_em(X, K, seed, max_iter=MAX_ITER, tol=TOL):
    rng = np.random.default_rng(seed)
    pi, mu, Sigma = init_params(X, K, rng)
    history, prev = [], -np.inf
 
    for it in range(1, max_iter + 1):
        tau, ll = e_step(X, pi, mu, Sigma)
        history.append(ll)
        pi, mu, Sigma = m_step(X, tau)
        if abs(ll - prev) < tol * abs(ll):            
            break
        prev = ll
 
    tau, ll = e_step(X, pi, mu, Sigma)
    history.append(ll)
    return dict(pi=pi, mu=mu, Sigma=Sigma, tau=tau, ll=ll,
                history=history, n_iter=it)
 
fit = fit_em(Z, K, SEED)
print(f"converged in {fit['n_iter']} iterations, log-likelihood {fit['ll']:.4f}")
print("pi  =", fit["pi"].round(4))
print("N_k =", fit["tau"].sum(axis=0).round(2))
```
 
```
converged in 27 iterations, log-likelihood -612.6255
pi  = [0.3775 0.2677 0.3548]
N_k = [67.18 47.65 63.17]
```


## Visuals

Each point is coloured by its three responsibilities:
<img src="https://github.com/user-attachments/assets/e06d7a19-f470-4208-ad74-d29e7bd2178b" alt="Scatter plot of the Wine data in the first two principal components, showing three separated groups coloured by mixture responsibility" width="700" />

 The X markers are the fitted means, and the rings are the one- and two-sigma contours of each
$$\Sigma_k$$. They come out different sizes and tilted differently, one of the advantages of a Gaussian model.
<img src="https://github.com/user-attachments/assets/b952e4e2-1ff1-4b98-9354-3c6415d2e6d5" alt="The same PCA scatter with three fitted Gaussian components, each marked by an X at its mean and surrounded by one- and two-sigma ellipses" width="700" />

On the left, the log-likelihood never decreases over the 27 iterations, as expected. On the
right, the gain per iteration flattens out around iteration 6 and then picks up again near
iteration 15.
<img src="https://github.com/user-attachments/assets/65c06936-081d-430c-86cf-083f7f6966fb" alt="Two panels: on the left the log-likelihood rises monotonically over 27 EM iterations with a plateau in the middle; on the right the per-iteration gain on a log scale dips near iteration 6, climbs again near iteration 15, then falls below the convergence threshold" width="900" />



### Sanity check against scikit-learn
 
```python
from sklearn.mixture import GaussianMixture
 
gm = GaussianMixture(n_components=K, covariance_type="full", reg_covar=REG,
                     tol=TOL, max_iter=MAX_ITER, n_init=10,
                     random_state=SEED).fit(Z)
 
print("ours    ll = %.4f" % fit["ll"])
print("sklearn ll = %.4f" % (gm.score(Z) * len(Z)))
print("same hard assignment:", (fit["tau"].argmax(1) == gm.predict(Z)).sum(), "/", len(Z))
```
 
```
ours    ll = -612.6255
sklearn ll = -612.6254
same hard assignment: 178 / 178
```
 
We notice the log-likelihoods agree to $$10^{-4}$$ and the two partitions are identical on all 178 points.



### Dataset
Used under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license, which permits sharing and adaptation provided appropriate credit is given.
Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.
 

 
