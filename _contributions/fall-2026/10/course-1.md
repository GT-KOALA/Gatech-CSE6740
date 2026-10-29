---
semester: fall-2026
lecture_number: "10"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: Roseburgendy
issue: 32
---
# Gaussian Process Regression: From Linear Regression to Kernels and Uncertainty

The lecture is one chain of ideas, and these notes follow it in five steps:
1. **Recap of linear regression:** the model, its likelihood, and the MLE solution.
2. **Limitations of linear regression:** one answer, no sense of confidence, and only straight lines.
3. **Bayesian linear regression:** keep a distribution over the weights, so predictions come with uncertainty (includes a Bayes' rule recap).
4. **Gaussian process regression (GPR):** apply the kernel trick (Lecture 9) to the Bayesian model, to get non-linear predictions and uncertainty.
5. **Interpretation of GPR:** what the mean, the variance, and the kernel mean.

Notation of this note:

| Symbol | Meaning |
|---|---|
| $$D=\{(x^{(i)},y^{(i)})\}_{i=1}^n$$ | Training data: $$n$$ input–target pairs |
| $$x^{(i)}\in\mathbb{R}^d$$ | The $$i$$-th input (feature vector) |
| $$X\in\mathbb{R}^{n\times d}$$ | Inputs stacked as rows |
| $$y\in\mathbb{R}^n$$ | Targets stacked into a vector |
| $$\theta\in\mathbb{R}^d$$ | Weight vector |
| $$\sigma^2$$ | Observation-noise variance |

A few slides write $$X^\top$$ where the rows-as-samples convention above gives $$X$$; these notes use the convention consistently.

## 1. Recap: Linear Regression

We start by defining a linear model with Gaussian noise $$\epsilon \sim N(0, \sigma^2)$$:

$$
y = \theta^\top x + \epsilon
\quad\Rightarrow\quad
y \mid x, \theta \sim N(\theta^\top x,\ \sigma^2)
$$

For a Gaussian random variable $$Y \sim N(\mu, \sigma^2)$$, the density is

$$
p(y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

Substituting $$\mu = \theta^\top x$$ gives the conditional density of $$y$$:

$$
p(y \mid x, \theta) = N(y;\ \theta^\top x,\ \sigma^2)
= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-\theta^\top x)^2}{2\sigma^2}\right)
$$

**Log-likelihood:** For i.i.d. data $$D = \{(x^{(i)}, y^{(i)})\}_{i=1}^n$$, the likelihood is the product $$\prod_{i=1}^n p(y^{(i)} \mid x^{(i)}, \theta)$$. Taking the log turns it into a sum:

$$
\ell(D \mid \theta)
= \sum_{i=1}^n \left[ -\log(\sqrt{2\pi}\sigma) - \frac{(y^{(i)}-\theta^\top x^{(i)})^2}{2\sigma^2} \right]
= \text{const} - \frac{1}{2\sigma^2} \sum_{i=1}^n (y^{(i)}-\theta^\top x^{(i)})^2
= \text{const} - \frac{1}{2\sigma^2} \|y - X\theta\|^2
$$

**Apply Maximum Likelihood Estimation (MLE):** Since the constant and the factor $$\frac{1}{2\sigma^2}$$ do not depend on $$\theta$$, maximizing the log-likelihood is equivalent to minimizing the squared error:

$$
\hat\theta_{MLE} = \arg\max_\theta\ \ell(D \mid \theta) = \arg\min_\theta\ \|y - X\theta\|^2
$$

Setting the gradient to zero:

$$
\frac{\partial}{\partial \theta} \|y - X\theta\|^2 = -2X^\top (y - X\theta) = 0
\ \Rightarrow\ X^\top X \theta = X^\top y
\ \Rightarrow\ \boxed{\hat\theta = (X^\top X)^{-1} X^\top y}
$$

For a new input $$x$$, we predict $$\hat y = \hat\theta^\top x$$.

Linear regression produces a **single weight vector** $$\theta^*$$, found by maximum likelihood, so all predictions rely on this one point estimate. Also, the model is **linear** in $$x$$. Ridge regression (Lecture 9) modifies the objective by adding a penalty $$\lambda\|\theta\|^2$$, which yields $$\theta^*=(X^\top X+\lambda I)^{-1}X^\top y$$. As we will see below, this penalty is secretly a Bayesian prior on $$\theta$$.

## 2. Limitations of Linear Regression

1. **No notion of confidence:** The prediction $$\hat y=\theta^{*\top}x$$ is just a number. A classifier that says "90% cat" and one that says "60% dog" both give an answer, but we would like to know how far to trust each. The same holds for regression: we would like a band around the prediction that is narrow where there is plenty of data and wide when we extrapolate.
2. **A single "best" answer ignores other explanations:** Maximum likelihood commits to one $$\theta^*$$. With few or noisy data, many different weight vectors explain the data almost equally well, and picking one throws that range of possible weight vectors away.
3. **Only linear relationships:** The model can only capture linear relationships. Lecture 9's fix was to replace $$x$$ with features $$\phi(x)$$ and use kernels, but that again produced a deterministic predictor.

## 3. Bayesian Linear Regression

### 3.1 Recap: Bayes' rule

For a hypothesis $$\theta$$ and observed data $$D$$:

$$
\underbrace{p(\theta\mid D)}_{\text{posterior}}=\frac{\overbrace{p(D\mid\theta)}^{\text{likelihood}}\ \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(D)}_{\text{evidence}}}
$$

$$
p(\theta\mid D) \propto p(D\mid\theta)\ p(\theta).
$$

- The **prior** $$p(\theta)$$ says what we believe before seeing data.
- The **likelihood** $$p(D\mid\theta)$$ says how well $$\theta$$ explains the data.
- The **posterior** $$p(\theta\mid D)$$ is the updated belief. It is large for $$\theta$$ that are both plausible a priori *and* fit the data.
- The **evidence** (total or marginal probability) $$p(D)=\int p(D\mid\theta)p(\theta)\ d\theta$$ does not depend on $$\theta$$. So for the *shape* of the posterior we can ignore it.

### 3.2 The Bayesian linear model (slides 6–7)

To address these limitations, we keep the same likelihood as in Section 1 and **additionally assume a Gaussian prior on the weights**:

$$
\theta\sim\mathcal{N}(0,S).
$$

($$S$$ is a covariance matrix; $$S=\tau^2I$$ says "weights are probably small, with no preferred direction.")

### 3.3 Training: computing the posterior (slides 7–8)

Take logs of Bayes' rule, $$p(\theta\mid D) \propto p(D\mid\theta)\ p(\theta)$$, and drop constants:

$$
\log p(\theta\mid D)= \log p(\theta) + \log p(D\mid\theta) + \text{const}= -\frac12\theta^\top S^{-1}\theta-\frac{1}{2\sigma^2}\|y-X\theta\|^2+\text{const}.
$$

Expand $$\|y-X\theta\|^2=y^\top y-2\theta^\top X^\top y+\theta^\top X^\top X\theta$$ and collect the quadratic and linear terms in $$\theta$$:

$$
\log p(\theta\mid D)=-\frac12\ \theta^\top\Big(\frac{1}{\sigma^2}X^\top X+S^{-1}\Big)\theta+\frac{1}{\sigma^2}\theta^\top X^\top y+\text{const}.
$$

A Gaussian $$\mathcal{N}(\mu,\Sigma)$$ has log-density $$-\tfrac12\theta^\top\Sigma^{-1}\theta+\theta^\top\Sigma^{-1}\mu+\text{const}$$. Matching the quadratic term gives the precision matrix, and matching the linear term gives the mean:

$$
\boxed{\Sigma^{-1}=\frac{1}{\sigma^2}X^\top X+S^{-1},\qquad \mu=\frac{1}{\sigma^2}\Sigma X^\top y=\big(X^\top X+\sigma^2S^{-1}\big)^{-1}X^\top y\ }
$$

Therefore,

$$
p(\theta\mid D)=\mathcal{N}(\mu,\Sigma).
$$

A Gaussian prior times a Gaussian likelihood is again Gaussian, so everything is in closed form. Interpretation:

- **Precision adds up.** $$\Sigma^{-1}$$ = prior precision $$S^{-1}$$ + data information $$\tfrac1{\sigma^2}X^\top X$$. More data → larger $$X^\top X$$ → smaller $$\Sigma$$ → more concentrated posterior.
- **Ridge regression in disguise.** With $$S=\tau^2I$$ the posterior mean (which is also the posterior mode, the MAP estimate) is $$(X^\top X+\lambda I)^{-1}X^\top y$$ with $$\lambda=\sigma^2/\tau^2$$. The Gaussian prior *is* the $$\ell_2$$ penalty. What the Bayesian view adds is the spread $$\Sigma$$ around that point estimate.

For $$y=\beta_1x+\beta_0$$, the weight space is 2-D, so the distribution over $$(\beta_0,\beta_1)$$ can be drawn as a heatmap (top row of the figure below). Each point in the heatmap is one line in the data space below it, so every red line is a weight vector sampled from the distribution directly above. Reading left to right, we observe 0, 1, 2, and about 20 points:

- **No data (prior):** a broad round blob, so no weights are preferred and the lines point in every direction.
- **One point:** a narrow ellipse. Every plausible line must pass near the point, which pins down the intercept $$\beta_0$$, but the slope $$\beta_1$$ is still uncertain.
- **Two points:** a small tilted ellipse. The second point constrains the slope, and the tilt shows that $$\beta_0$$ and $$\beta_1$$ are correlated (the sign of the tilt depends on where the data sit along $$x$$).
- **About 20 points:** a tiny spot, with the lines bunched tightly along the trend.

Each observation removes the lines that miss it, so the posterior shrinks and the sampled lines concentrate around the true trend. This is the update $$\Sigma^{-1}=S^{-1}+\tfrac{1}{\sigma^2}X^\top X$$ seen visually: every new row of $$X$$ adds information and tightens the blob.

![Four panels of Bayesian line fitting as data points accumulate: the weight-space heatmap shrinks from a broad blob to a tiny spot, while the sampled red lines bunch around the blue data points](/Gatech-CSE6740/assets/semesters/fall-2026/lectures/10/figures/bayesian-line-fitting.jpg)

*Bayesian line fitting: posterior over $$(\beta_0,\beta_1)$$ and sampled lines. Figure from the Lecture 10 slides (slide 9), K. Wang, CSE 6740. Image credit: Gregory Gundersen.*

### 3.4 Prediction: Averaging over All Plausible Weights (slides 7, 10–12)

**Recall: prediction in linear regression (MLE).** Linear regression first computes a single weight vector $$\hat\theta=(X^\top X)^{-1}X^\top y$$ and then plugs it in:

$$
\hat y=\hat\theta^\top x,\qquad p(y\mid x,\hat\theta)=\mathcal{N}(\hat\theta^\top x,\ \sigma^2)
$$

This is equivalent to giving 100% of the "vote" to one $$\theta$$, so the only uncertainty it reports is the noise $$\sigma^2$$.

**Bayesian prediction.** Instead of plugging in one $$\theta$$, we average the likelihood over the posterior. Each $$\theta$$ makes its own prediction $$p(y\mid x,\theta)$$, weighted by its posterior probability $$p(\theta\mid D)$$, that is, by how plausible it is after seeing the data:

$$
p(y\mid x,D)=\int p(y\mid x,\theta)\ p(\theta\mid D)\ d\theta,
\qquad y\mid x,\theta\sim\mathcal{N}(\theta^\top x,\sigma^2),\quad \theta\mid D\sim\mathcal{N}(\mu,\Sigma)
$$

Here $$y=\theta^\top x+\epsilon$$ is a linear combination of independent Gaussians ($$\theta$$ and $$\epsilon$$), so it is also Gaussian. We therefore only need its mean and variance:

$$
\mathbb{E}[y]=\mu^\top x,\qquad \mathrm{Var}[y]=x^\top\Sigma x+\sigma^2
$$

$$
\boxed{y\mid x,D\sim\mathcal{N}\big(\mu_{\text{pred}},\sigma^2_{\text{pred}}\big),\quad
\mu_{\text{pred}}=\mu^\top x=x^\top(X^\top X+\sigma^2S^{-1})^{-1}X^\top y,\quad
\sigma^2_{\text{pred}}=x^\top\Sigma x+\sigma^2}
$$

**The variance has two parts:**

- $$\sigma^2$$ is the **noise floor**: even if $$\theta$$ were known exactly, a new observation still scatters by $$\sigma^2$$.
- $$x^\top\Sigma x$$ is the **parameter uncertainty** seen along direction $$x$$. It is small near the training inputs, grows when extrapolating, and shrinks as $$n$$ grows.

**Comparison:**

| | Linear regression (MLE) | Bayesian linear regression |
|---|---|---|
| What we keep | one point estimate $$\hat\theta$$ | the full posterior $$p(\theta\mid D)=\mathcal{N}(\mu,\Sigma)$$ |
| Prediction mean | $$\hat\theta^\top x$$, with $$\hat\theta=(X^\top X)^{-1}X^\top y$$ | $$\mu^\top x$$, with $$\mu=(X^\top X+\sigma^2S^{-1})^{-1}X^\top y$$ |
| Prediction variance | $$\sigma^2$$ (noise only) | $$x^\top\Sigma x+\sigma^2$$ (parameter uncertainty + noise) |
| Far from the data | same width everywhere | band widens when extrapolating |
| Prior | none | $$\theta\sim\mathcal{N}(0,S)$$; the term $$\sigma^2S^{-1}$$ acts as regularization |

The two means differ only by the prior term $$\sigma^2S^{-1}$$ inside the inverse. With $$S=\tau^2I$$ this is exactly the Ridge prediction with $$\lambda=\sigma^2/\tau^2$$. As $$n\to\infty$$, $$X^\top X$$ dominates, $$\Sigma\to0$$, and the Bayesian prediction converges to the MLE prediction. The Bayesian model therefore only behaves differently when data is scarce or $$x$$ lies far from the training inputs, which is exactly where we most want the model to know that it is unsure.

## 4. Gaussian Process Regression

**Why do we want GPR?** The two models so far each fix only part of the problem:

- **Bayesian linear regression** (Section 3) gives predictions with uncertainty, but the mean is still a **straight line** in $$x$$.
- **Kernel regression** (Lecture 9) gives **non-linear** predictions, but they are deterministic: a curve with **no confidence band**.

**We want both: a flexible non-linear curve and an honest measure of how much to trust it.** The natural route is to apply the Bayesian model of Section 3 to features $$\phi(x)$$, as in Lecture 9. The obstacle is cost: $$\phi(x)$$ can be very high-dimensional or even infinite-dimensional, so computing $$\mu$$ and $$\Sigma$$ directly is impractical. The kernel trick removes this obstacle. We will rewrite the Bayesian prediction so that it depends on the data only through the kernel $$k(x,x')$$, never through $$\phi$$ itself. The result is **Gaussian process regression**.

The derivation has three steps. First we recall the kernel quantities we need. Then we write the Bayesian model on features. Finally we convert the predictive mean and variance into kernel form.

### Recap: kernel methods (Lecture 9)

- **Feature map.** Replace $$x$$ by $$\phi(x)\in\mathbb{R}^D$$ so a model that is linear in $$\phi(x)$$ is non-linear in $$x$$. Write $$\tilde X\in\mathbb{R}^{n\times D}$$ for the matrix whose rows are $$\phi(x^{(i)})^\top$$. The cost: $$D$$ can be huge or infinite.
- **Kernel.** $$k(x,x')=\phi(x)^\top\phi(x')$$ is an inner product of features, i.e. a **similarity score** between two inputs that can be computed without forming $$\phi$$.
- **Gram matrix and kernel vector.** For the training set and a test input $$x$$:

$$
K=\big[k(x^{(i)},x^{(j)})\big]_{ij}\in\mathbb{R}^{n\times n}\ (=\tilde X\tilde X^\top),\qquad
k_x=\big[k(x,x^{(i)})\big]_{i}\in\mathbb{R}^{n}\ (=\tilde X\phi(x)),\qquad k_{xx}=k(x,x).
$$

### Step 1: Bayesian linear regression on features (slide 16)

Take the Bayesian model of Section 3 with $$x\to\phi(x)$$ and $$X\to\tilde X$$. For simplicity the slides use $$S=I$$. Then

$$
\mu=\big(\tilde X^\top\tilde X+\sigma^2I\big)^{-1}\tilde X^\top y,\qquad \Sigma=\Big(\tfrac1{\sigma^2}\tilde X^\top\tilde X+I\Big)^{-1},
$$

$$
\mu_{\text{pred}}=\phi(x)^\top\mu,\qquad \sigma^2_{\text{pred}}=\phi(x)^\top\Sigma\ \phi(x)+\sigma^2=\sigma^2\ \phi(x)^\top\big(\tilde X^\top\tilde X+\sigma^2I\big)^{-1}\phi(x)+\sigma^2 .
$$

These need a $$D\times D$$ inverse and explicit features. We now rewrite them using only $$k_{xx},k_x,K$$.

### Step 2: the mean (slide 17)

$$
\mu_{\text{pred}}=\phi(x)^\top\tilde X^\top\big(\tilde X\tilde X^\top+\sigma^2I\big)^{-1}y=\boxed{k_x^\top\big(K+\sigma^2I\big)^{-1}y\ }
$$

### Step 3: the variance (slides 17–18)

Use the **Woodbury (matrix inversion) identity** $$(I+UV)^{-1}=I-U(I+VU)^{-1}V$$. Note $$\Sigma=\big(I+\tfrac1{\sigma^2}\tilde X^\top\tilde X\big)^{-1}$$; choose $$U=\tfrac1\sigma\tilde X^\top$$ and $$V=\tfrac1\sigma\tilde X$$, so $$UV=\tfrac1{\sigma^2}\tilde X^\top\tilde X$$ and $$VU=\tfrac1{\sigma^2}K$$. Then

$$
\Sigma=I-\tfrac{1}{\sigma^2}\tilde X^\top\Big(I+\tfrac{1}{\sigma^2}K\Big)^{-1}\tilde X=I-\tilde X^\top\big(K+\sigma^2I\big)^{-1}\tilde X .
$$

Sandwich with $$\phi(x)$$ and use $$\tilde X\phi(x)=k_x$$:

$$
\phi(x)^\top\Sigma\ \phi(x)=\phi(x)^\top\phi(x)-k_x^\top\big(K+\sigma^2I\big)^{-1}k_x=k_{xx}-k_x^\top\big(K+\sigma^2I\big)^{-1}k_x .
$$

Adding the noise floor:

$$
\boxed{\sigma^2_{\text{pred}}=k_{xx}-k_x^\top\big(K+\sigma^2I\big)^{-1}k_x+\sigma^2}
$$

### The GPR recipe (slides 19–20)

Given a kernel $$k$$, data $$D$$ and noise level $$\sigma^2$$:

1. Form $$K$$ once from the training inputs.
2. For each test input $$x$$ compute $$k_x$$ and $$k_{xx}$$.
3. The belief about the **noise-free function value** $$f(x)$$ is Gaussian:

$$
f(x)\mid D\sim\mathcal{N}\Big(k_x^\top(K+\sigma^2I)^{-1}y,\ \ k_{xx}-k_x^\top(K+\sigma^2I)^{-1}k_x\Big),
$$

and the distribution of a new **noisy observation** $$y$$ has the same mean and variance $$+\ \sigma^2$$ (the $$\sigma^2_{\text{pred}}$$ of the slides).

Repeating this over a grid of $$x$$ gives a mean curve with a confidence band (e.g. $$\pm2$$ standard deviations for about 95%).

**Computation (slide 20).** The matrix $$(K+\sigma^2I)^{-1}$$ depends only on the training data, so it is computed once and reused. In practice we store a Cholesky factorization and $$\alpha=(K+\sigma^2I)^{-1}y$$ instead of an explicit inverse. This costs $$O(n^3)$$ once, then $$O(n)$$ per test mean and $$O(n^2)$$ per test variance. The $$n^3$$ cost is the main limit of exact GPs on big datasets.

## 5. Interpreting Gaussian Process Regression

GPR returns a Gaussian for every test input, so we can read its mean and variance separately, and then ask what the model looks like as a whole.

### 5.1 The mean is kernel regression

Write $$\alpha=(K+\sigma^2I)^{-1}y\in\mathbb{R}^n$$. The predictive mean is

$$
\mu_{\text{pred}}(x)=\sum_{i=1}^n\alpha_i\ k(x,x^{(i)})
$$

This is a sum of kernel bumps centered at the training points, which is exactly the kernel regression of Lecture 9, with $$\sigma^2$$ playing the role of the ridge parameter $$\lambda$$. GPR does not change the prediction of kernel regression. What it adds is the variance.

### 5.2 The variance measures what the data do not tell us

The variance of the function value is

$$
k_{xx}-k_x^\top(K+\sigma^2I)^{-1}k_x
$$

that is, the prior variance $$k_{xx}$$ minus the part that the training data explain.

- **Near the training data:** $$k_x$$ is large, so the subtracted term is large and the uncertainty drops. A larger noise level $$\sigma^2$$ keeps the band wider, because the model refuses to trust a single noisy observation.
- **Far from the data:** for the RBF (squared exponential) kernel, $$k(x,x')=\exp\big(-\|x-x'\|^2/(2\ell^2)\big)$$ with length-scale $$\ell$$, we have $$k(x,x^{(i)})\to0$$, so $$k_x\to0$$ and the variance returns to the prior value $$k_{xx}$$. With no information, the model is no more confident than before seeing any data.
- **Dependence on $$y$$:** the variance depends only on *where* the inputs are, not on the targets $$y$$. The mean depends on $$y$$, but the width of the band does not.

For a new noisy observation $$y$$, add the noise floor: the variance becomes the expression above $$+\ \sigma^2$$.

### 5.3 A GP is a distribution over functions

Since $$f(x)=\theta^\top\phi(x)$$ is linear in $$\theta$$, a Gaussian prior on $$\theta$$ makes any collection of function values $$(f(x_1),\dots,f(x_m))$$ jointly Gaussian. With the prior $$\theta\sim\mathcal{N}(0,I)$$ used in Step 1,

$$
\mathbb{E}[f(x)]=0,\qquad \mathrm{Cov}\big(f(x),f(x')\big)=k(x,x')
$$

A distribution over functions with this property is a **Gaussian process**, written $$f\sim\mathcal{GP}(0,k)$$. The kernel is its covariance function: inputs that the kernel calls similar receive similar function values. Conditioning on the data gives exactly the predictive formulas of Section 4.

The figure below shows this with the RBF kernel. Each dashed curve is one random function drawn from the distribution, the black line is the mean, and the grey band is $$\pm1$$ standard deviation.

![Two stacked plots of a Gaussian process with an RBF kernel: random functions with a constant-width grey band before any data (top), and functions pinned to red observations with a band that narrows at the data and widens in the gaps (bottom)](/Gatech-CSE6740/assets/semesters/fall-2026/lectures/10/figures/gp-rbf-prior-posterior.jpg)

*Gaussian process regression with an RBF kernel: prior (top) and posterior (bottom) samples. Figure from the Lecture 10 slides (slide 21), K. Wang, CSE 6740. Image credit: scikit-learn.*

- **Top (prior):** before seeing data, the samples wander freely. The mean is zero everywhere, and the band has constant width $$\pm\sqrt{k(x,x)}$$, so the model is equally uncertain at every $$x$$.
- **Bottom (posterior):** after conditioning on the observations, the sampled functions are pulled toward the red points, so the samples agree with each other there and the band pinches. In this figure the observation noise is negligible (scikit-learn's default adds only a tiny $$10^{-10}$$ to the diagonal), so the samples pass essentially through the points and the band nearly vanishes there. With noise $$\sigma^2>0$$, as in the formulas of Section 4, the samples need not pass exactly through the observed targets, and the variance at a training input, $$k_{xx}-k_x^\top(K+\sigma^2I)^{-1}k_x$$, stays above zero. In the gaps between points, such as around $$x=2.5$$ and $$x=4$$, the samples disagree and the band widens.

This is why a GP is useful when we need to measure uncertainty: it gives not only a curve but also where that curve can be trusted.

### 5.4 The big picture (slide 22)

The four models in this lecture are not separate methods but four corners of one grid. Each row asks whether the model is linear or kernelized, and each column asks whether it keeps a single point estimate or a full Bayesian posterior. Moving along one axis changes exactly one ingredient, and Gaussian process regression sits in the corner where both changes are applied.

| | Deterministic (point estimate) | Bayesian (Gaussian prior on parameters) |
|---|---|---|
| **Linear** | Linear / ridge regression | Bayesian linear regression |
| **Non-linear (kernel trick)** | Kernel regression | **Gaussian process regression** |

- **Left → right** (deterministic → Bayesian): put a Gaussian prior on the parameters and keep the whole posterior instead of only its peak. This gives uncertainty.
- **Top → bottom** (linear → non-linear): replace inner products $$x^\top x'$$ with kernels $$k(x,x')$$. This gives flexibility without working in a huge feature space.
- **Cost:** GPR combines both changes, but exact GPR needs an $$O(n^3)$$ factorization of $$K+\sigma^2I$$, which is the main limit on large datasets.

## References

1. Kai Wang. *CSE/ISyE 6740 Lecture 10: Gaussian Process Regression*, September 28, 2026, slides 6–22. Main source for the Bayesian linear model, the kernel derivation, and the GPR figures. [Course slides](https://github.com/GT-KOALA/Gatech-CSE6740/blob/main/assets/semesters/fall-2026/lectures/10/lecture-10-gaussian-processes.pdf).
2. Kai Wang. *CSE/ISyE 6740 Lecture 09: Kernel Methods*, September 23, 2026. Source for the kernel and ridge regression recap in Section 4. [Course slides](https://github.com/GT-KOALA/Gatech-CSE6740/blob/main/assets/semesters/fall-2026/lectures/09/lecture-09-kernel-methods.pdf).
3. Gregory Gundersen. *Bayesian Linear Regression*, February 4, 2020. Image credit for the Bayesian line-fitting figure on the Lecture 10 slides (slide 9). [Blog post](https://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/).
4. scikit-learn developers. *Illustration of prior and posterior Gaussian process for different kernels*, scikit-learn 1.9.1 documentation, 2026. Source of the GP prior and posterior figure. [Example page](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html).
5. Christopher M. Bishop. *Pattern Recognition and Machine Learning*, Springer, 2006, Section 3.3 (Bayesian linear regression) and Section 6.4 (Gaussian processes). Supplementary reference.
6. Carl Edward Rasmussen and Christopher K. I. Williams. *Gaussian Processes for Machine Learning*, MIT Press, 2006, Chapter 2 (weight-space and function-space views). Supplementary reference. [Free online](https://gaussianprocess.org/gpml/).
7. GeeksforGeeks. *Gaussian Process Regression (GPR)*, last updated August 11, 2025. Supplementary reference for a scikit-learn implementation. [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/gaussian-process-regression-gpr/).
8. GeeksforGeeks. *Gaussian Processes in Machine Learning*, last updated June 15, 2026. Supplementary reference for GP concepts (prior, posterior, kernels). [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/gaussian-processes-in-machine-learning/).

---

*Claude was used to do latex formatting and polishing the wording of this note*
