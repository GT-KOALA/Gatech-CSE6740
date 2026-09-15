---
semester: fall-2026
lecture_number: "04"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: liuyunhaozz
issue: 6
---
# 1. Outline
## 1.1 Density estimation
- What is density estimation
- Parametric models and maximum likelihood estimation (MLE)
- Nonparametric models: histogram

(Kernel density estimation and Gaussian mixture models come later in the lecture and are not covered in this part.)

# 2. Notation
- A **dataset** \(D = \{x^1, x^2, \ldots, x^n\}\) has \(n\) data points. The superscript \(i\) in \(x^i\) is the sample index, not a power.
- Each data point is in \(\mathbb{R}^d\), where \(d\) is the dimension.
- \(P(x \mid \theta)\) is a model with parameter \(\theta\), and \(P^*(x)\) is the true (unknown) distribution.
- For histograms: \(m\) is the number of bins per dimension, \(B_j\) is the \(j\)-th bin, \(c_j\) is the number of points in \(B_j\), and \(I(\cdot)\) is the indicator function (1 if the condition is true, 0 otherwise).

# 3. Introduction to Density Estimation
**Goal:** Given a bag of data points, estimate the density function \(p(x)\) that the samples come from, instead of only looking at the scatter plot.

Density estimation helps us:
- Learn more about the "shape" of the data cloud, e.g. where the data concentrate, how they spread, and how many clusters there are.
- Assess the likelihood of seeing a particular data point:
  - **High density value** \(\rightarrow\) typical data point
  - **Low density value** \(\rightarrow\) abnormal data point / outlier (useful for anomaly detection)

**Example (test scores):** For students' statistics scores and linear algebra scores, we can draw a 1D histogram for each score (a histogram is also a density estimation method), or estimate a 2D density using both scores. The 2D contours tilt from bottom-left to top-right, which means students who do well in statistics tend to also do well in linear algebra.

# 4. Parametric Models
**Definition:** Models which can be described by **a fixed number of parameters**.

- **Discrete case:** Bernoulli distribution
\[
P(x \mid \theta) = \theta^x (1-\theta)^{1-x}, \qquad x \in \{0, 1\}
\]
One parameter \(\theta \in [0,1]\), which generates a family of models \(\mathcal{F} = \{P(x \mid \theta) : \theta \in [0,1]\}\).

- **Continuous case:** multivariate Gaussian distribution
\[
P(x \mid \mu, \Sigma) = \frac{1}{|\Sigma|^{\frac{1}{2}} (2\pi)^{\frac{d}{2}}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
\]
Two sets of parameters: mean \(\mu \in \mathbb{R}^d\) and covariance \(\Sigma \in \mathbb{R}^{d \times d}\) (symmetric PSD, and it needs to be invertible for this formula). Even though \(\mu\) is a vector and \(\Sigma\) is a matrix, the number of entries is fixed once \(d\) is fixed, so it is still parametric.

Note: the slide writes the dimension as \(n\). Here I use \(d\) so it does not get mixed up with the number of samples.

## 4.1 Maximum Likelihood Estimator (MLE)
**Input:** \(n\) data points \(D = \{x^1, x^2, \ldots, x^n\}\) drawn independently and identically distributed (i.i.d.) from some distribution \(P^*(x)\).

**Goal:** Fit the model \(P(x \mid \theta)\) to the dataset \(D\), i.e. find the \(\theta\) that maximizes the probability of seeing the data.

Since the samples are i.i.d., the joint likelihood is the product of the individual likelihoods. We take the log because log is increasing (so the maximizer does not change) and it turns the product into a sum, which is easier to differentiate:
\[
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \log \prod_{i=1}^{n} P(x^i \mid \theta) = \arg\max_\theta \sum_{i=1}^{n} \log P(x^i \mid \theta)
\]

A trick we will use throughout the semester: compute the derivative of the log-likelihood, set it to zero, and solve for the parameters. MLE is used in both unsupervised and supervised learning.

## 4.2 Example: Biased Coin
**Problem:** Estimate the probability \(\theta\) of landing heads using a biased coin.

**Input:** A sequence of \(n\) i.i.d. flips \(D = \{x^1, \ldots, x^n\}\), where \(x^i \in \{0, 1\}\) (1 = head, 0 = tail).

**Model:**
\[
P(x \mid \theta) = \theta^x (1-\theta)^{1-x} =
\begin{cases}
1-\theta & \text{for } x = 0 \\
\theta & \text{for } x = 1
\end{cases}
\]
Note: slide 10 writes the exponent of \((1-\theta)\) as \(x\), but it should be \(1-x\).

**Likelihood:** \(P(D \mid \theta) = \theta^{n_{\text{head}}} (1-\theta)^{n_{\text{tail}}}\), where \(n_{\text{head}}\) is the number of heads and \(n_{\text{tail}} = n - n_{\text{head}}\) is the number of tails.

**Solve:**
\[
\begin{aligned}
l(\theta; D) &= n_{\text{head}} \log\theta + (n - n_{\text{head}}) \log(1-\theta) && \text{Take log} \\
\frac{\partial l}{\partial \theta} &= \frac{n_{\text{head}}}{\theta} - \frac{n - n_{\text{head}}}{1-\theta} = 0 && \text{Set derivative to zero} \\
n_{\text{head}}(1-\theta) &= (n - n_{\text{head}})\,\theta && \text{Multiply both sides by } \theta(1-\theta) \\
\hat{\theta}_{\text{MLE}} &= \frac{n_{\text{head}}}{n} = \frac{1}{n}\sum_{i=1}^{n} x^i && \text{Rearrange}
\end{aligned}
\]

The second derivative \(-\frac{n_{\text{head}}}{\theta^2} - \frac{n - n_{\text{head}}}{(1-\theta)^2}\) is negative, so this is a maximum.

**Intuition:** The estimate is just the fraction of heads. For example, 7 heads out of 10 flips gives \(\hat{\theta}_{\text{MLE}} = 0.7\), which matches our intuition.

## 4.3 Example: Univariate Gaussian
**Model:** Gaussian distribution in \(\mathbb{R}\)
\[
P(x \mid \mu, \sigma) = \frac{1}{(2\pi)^{\frac{1}{2}}\sigma} \exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)
\]

**Input:** \(n\) i.i.d. samples \(D = \{x^1, \ldots, x^n\}\), \(x^i \in \mathbb{R}\).

**Goal:** Estimate the two parameters \(\mu\) and \(\sigma\), i.e. find which bell curve fits the samples best.

Note: the slide writes the likelihood of one data point as \(P(x^i \mid \mu, \sigma) \propto \exp\left(-\frac{1}{2\sigma^2}(x^i-\mu)^2\right)\), dropping the constant. But the constant \(\frac{1}{(2\pi)^{1/2}\sigma}\) depends on \(\sigma\), so we have to keep it when estimating \(\sigma\).

**Log-likelihood:**
\[
\begin{aligned}
l(\mu, \sigma; D) &= \log \prod_{i=1}^{n} \frac{1}{(2\pi)^{\frac{1}{2}}\sigma} \exp\left(-\frac{1}{2\sigma^2}(x^i-\mu)^2\right) \\
&= -\frac{n}{2}\log 2\pi - \frac{n}{2}\log\sigma^2 - \sum_{i=1}^{n} \frac{(x^i-\mu)^2}{2\sigma^2} && \text{Log turns product into sum}
\end{aligned}
\]

**Solve for \(\mu\)** (only the last term depends on \(\mu\)):
\[
\begin{aligned}
\frac{\partial l}{\partial \mu} &= \sum_{i=1}^{n} \frac{x^i - \mu}{\sigma^2} = 0 && \text{Set derivative to zero} \\
\sum_{i=1}^{n} x^i &= n\mu && \text{Cancel } \sigma^2 \\
\hat{\mu}_{\text{MLE}} &= \frac{1}{n}\sum_{i=1}^{n} x^i
\end{aligned}
\]

**Solve for \(\sigma^2\)** (treat \(\sigma^2\) as the variable; only the 2nd and 3rd terms depend on it):
\[
\begin{aligned}
\frac{\partial l}{\partial \sigma^2} &= -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n} (x^i-\mu)^2 = 0 && \text{Set derivative to zero} \\
\sum_{i=1}^{n} (x^i-\mu)^2 &= n\sigma^2 && \text{Multiply both sides by } 2\sigma^4 \\
\hat{\sigma}^2_{\text{MLE}} &= \frac{1}{n}\sum_{i=1}^{n} (x^i-\hat{\mu}_{\text{MLE}})^2 && \text{Plug in } \hat{\mu}_{\text{MLE}}
\end{aligned}
\]

**Intuition:** The mean is the sample average, and the variance is the average squared distance from the mean. Note that MLE divides by \(n\), not \(n-1\) like the unbiased sample variance in statistics, so it slightly underestimates the variance when \(n\) is small.

## 4.4 Density Example (2D Gaussian)
We can fit a 2D Gaussian to the (statistics score, linear algebra score) data. The fitted density has one peak, and its contours are ellipses tilted to the top-right (positive correlation). With the fitted model we can ask the likelihood of any score pair. For example, a student with statistics score 7 but linear algebra score 1 falls outside the contours, so this is very unlikely.

Drawback: a single Gaussian has only one peak, so it cannot capture the two bumps we see in the score data.

# 5. Nonparametric Models
**Definition:** Models which can **NOT** be described by a fixed number of parameters.
- "Nonparametric" does **NOT** mean there are no parameters.
- One can think of them as having many many (infinite) parameters.
- Examples: histogram, kernel density estimator (covered later).

**Why is a histogram nonparametric?** The number of bins is our choice: 2 bins, 10 bins, or as many as we want, and each bin has its own height. So the number of parameters is not fixed. If we fix the bins once and for all, that histogram becomes parametric.

## 5.1 1-D Histogram
One of the simplest nonparametric density estimators.

**Input:** \(n\) i.i.d. samples \(D = \{x^1, \ldots, x^n\}\), \(x^i \in [0, 1)\).

- **Step 1:** Split \([0,1)\) into \(m\) bins:
\[
B_1 = \left[0, \frac{1}{m}\right),\ B_2 = \left[\frac{1}{m}, \frac{2}{m}\right),\ \ldots,\ B_m = \left[\frac{m-1}{m}, 1\right)
\]
- **Step 2:** Count the number of points: \(c_1\) points in \(B_1\), \(c_2\) points in \(B_2\), ...
- **Step 3:** For a new test point \(x\), the probability density function is:
\[
p(x) = \sum_{j=1}^{m} \frac{m c_j}{n} I(x \in B_j)
\]

**Intuition:** The height of each bin is \(\frac{c_j/n}{1/m}\), the fraction of points in the bin divided by the bin width. Dividing by the width turns a probability into a density.

## 5.2 Why is Histogram Valid?
**Requirement for density:** \(p(x) \geq 0\) and \(\int_\Omega p(x)\,dx = 1\).

- **Non-negative:** counts \(c_j \geq 0\), so \(p(x) \geq 0\).
- **Integrates to 1:**
\[
\begin{aligned}
\int_\Omega p(x)\,dx &= \int_{[0,1)} \sum_{j=1}^{m} \frac{m c_j}{n} I(x \in B_j)\,dx \\
&= \sum_{j=1}^{m} \int_{\left[\frac{j-1}{m}, \frac{j}{m}\right)} \frac{m c_j}{n}\,dx && \text{Swap sum and integral; indicator is 1 only inside } B_j \\
&= \sum_{j=1}^{m} \frac{m c_j}{n} \cdot \frac{1}{m} && \text{Integral of a constant over width } \tfrac{1}{m} \\
&= \sum_{j=1}^{m} \frac{c_j}{n} = 1 && \text{Counts add up to } n
\end{aligned}
\]

This is why we need the factor \(\frac{m}{n}\): \(m\) cancels the bin width, and \(n\) cancels the total count.

## 5.3 Output Depends on Where You Put the Bins
Same data and same bin width (0.5); only the bin edges are shifted:
- Breaks at n.0 and n.5: counts are 5, 3, 2, 2, so it looks like one peak that decays.
- Breaks at n.25 and n.75: counts are 2, 6, 0, 3, 1, so it looks like two separate groups.

The bin width also matters. Bins that are too wide over-smooth the data and hide structure. Bins that are too narrow make the histogram spiky, with many empty bins.

## 5.4 Higher Dimensional Histogram
**Input:** \(n\) i.i.d. samples \(D = \{x^1, \ldots, x^n\}\), \(x^i \in [0,1)^d\).

Split each dimension into \(m\) intervals, so \([0,1)^d\) is split evenly into \(m^d\) bins:
\[
\begin{aligned}
B_1 &= \left[0, \frac{1}{m}\right) \times \left[0, \frac{1}{m}\right) \times \cdots \times \left[0, \frac{1}{m}\right) \\
B_2 &= \left[\frac{1}{m}, \frac{2}{m}\right) \times \left[0, \frac{1}{m}\right) \times \cdots \times \left[0, \frac{1}{m}\right) \\
&\ \ \vdots \\
B_{m^d} &= \left[\frac{m-1}{m}, 1\right) \times \left[\frac{m-1}{m}, 1\right) \times \cdots \times \left[\frac{m-1}{m}, 1\right)
\end{aligned}
\]

Bin size is \(h = \frac{1}{m}\). The density is the same idea as 1D, but we divide by the bin volume \(h^d = \frac{1}{m^d}\):
\[
p(x) = \sum_{j=1}^{m^d} \frac{m^d c_j}{n} I(x \in B_j)
\]

**Example:** the 2D histogram of statistics vs. linear algebra scores (10 × 10 bins) has its tall bars along the diagonal, which shows the same positive correlation.

## 5.5 Computation Consideration
**Problem:** too many bins. Not good for high dimensional data.

- The number of bins \(m^d\) grows exponentially with \(d\). For example, \(m = 10\) and \(d = 6\) already gives \(10^6\) bins, which is more than the number of samples we usually have.
- If \(m^d\) is larger than the number of samples \(n\), most bins are empty (at most \(n\) bins can contain points).
- Consequences:
  - Most new points fall into empty bins and get \(p(x) = 0\), so we cannot tell typical points from outliers.
  - Non-empty bins only have a few points each, so the estimate is noisy.
  - Storing \(m^d\) counts is expensive in memory and computation.

# 6. Classroom Dialog
- **Question:** For what kind of data do we use nonparametric models like the histogram? Is it use case dependent or data dependent?
- **Answer:** Both, since the use case also affects the data. Histograms mostly work for low dimensions like 1D or 2D. For high dimensional data (e.g. 10 dimensions), a multi-dimensional histogram just doesn't work. The histogram is usually the initial step for density estimation and data analysis.
