---
semester: fall-2026
lecture_number: "03"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: PabloRaigoza
issue: 5
---
# 1. Outline
## 1.1 Linear dimensionality reduction
- Principle component analysis (PCA)
- Eigenvalue decomposition
- Reconstruction

# 2. Matrix and Vector Conventions
- A **data point** or **feature** is always a column vector in $$\mathbb{R}^d$$ with dimension $$d$$.
- A **feature matrix** is an $$n\times d$$ matrix concatenating $$n$$ data points as rows.
$$
\underbrace{x^i = \begin{bmatrix}
    x^i_1 \\
    x^i_2 \\
    \vdots \\
    x^i_d
\end{bmatrix} \in \mathbb{R}^d}_{\text{data point/feature}} \qquad \underbrace{X = [x^1, x^2, \ldots, x^n]^\top = \begin{bmatrix}
                (x^1)^\top  \\
                (x^2)^\top  \\
                \vdots \\
                (x^n)^\top 
            \end{bmatrix} \in \mathbb{R}^{n\times d}}_{\text{feature matrix}}
$$

# 3 Introduction to Dimensionality Reduction
**Definition:** Dimensionallity reduction is the process of reducing the number of random variables under considerations.

There are two ways to reduce number of random variables:
- Combine, transform, or select variables
- Use linear or non-linear operations

Dimension reduced data can be used for:
- Visualization
- Aggregating weak signals
- Cleaning data
- Speeding up subsequent learning task
- Simplify model

# 4 Principle Component Analysis (PCA)
**Algorithm:**

**Input:** Given $$n$$ data points $$\{x^1,x^2,\ldots,x^n\}$$ in $$\mathbb{R}^d$$.

- **Step 1:** Estimate the mean and covariance of the dataset.
$$
    \underbrace{\mu = \frac{1}{n}\sum_{i=1}^{n}x^i}_{\text{mean vector}} \in \mathbb{R}^d \qquad \underbrace{C = \frac{1}{n}\sum_{i=1}^{n}(x^i - \mu)(x^i - \mu)^\top }_{\text{covariance matrix}} \in \mathbb{R}^{d\times d}
$$

- **Step 2:** Compute eigenvectors $$w^1,w^2,\ldots,w^d$$ of $$C$$ and their corresponding eigenvalues $$\lambda_1,\lambda_2,\ldots,\lambda_d$$ such that $$\lambda_1$$ is the largest eigenvalue, $$\lambda_2$$ is the second largest eigenvalue, and so on.

- **Step 3:** Compute reduced representation with $$\Lambda_k = \text{diag}(\lambda_1,\lambda_2,\ldots,\lambda_k)$$ and $$W_k = [w^1,w^2,\ldots,w^k]^\top$$:
$$
    z^i = \begin{bmatrix}
        (w^1)^\top (x^i - \mu) / \sqrt{\lambda_1} \\
        (w^2)^\top (x^i - \mu) / \sqrt{\lambda_2} \\
        \vdots \\
        (w^k)^\top (x^i - \mu) / \sqrt{\lambda_k}
    \end{bmatrix} = \Lambda_k^{-\frac{1}{2}} W_k (x^i - \mu)
$$

One criterion we can use for dimensionality reduction is capture the variation of the data. While we do this processing we can discover variables or dimensions that highly correlated with each other to combine them into a single variable, represent highly related phenomena, or lead to simpler representation.
    

## 4.1 Reduction Problem Formulation
**Input:** Given $$n$$ data points $$\{x^1,x^2,\ldots,x^n\}$$ in $$\mathbb{R}^d$$ with mean $$\mu$$.

**Find:** Direction unit vector $$w \in \mathbb{R}^d$$ such that the variance of the projected data points is maximized.
$$
    \max_{w :\|w\|_2 = 1} \frac{1}{n}\sum_{i=1}^{n}(w^\top (x^i - \mu))^2
$$

**Notice that the maximation problem is the same as:**
$$
\begin{aligned}
    \max_{w:\|w\|_2=1} \frac{1}{n}\sum_{i=1}^{n}(w^\top(x^i-\mu))^2
    &= \max_{w:\|w\|_2=1} \frac{1}{n}\sum_{i=1}^{n}
    w^\top(x^i-\mu)(x^i-\mu)^\top w
    && \text{Transpose} \\[6pt]
    &= \max_{w:\|w\|_2=1} w^\top
    \underbrace{\left(\frac{1}{n}\sum_{i=1}^{n}(x^i-\mu)(x^i-\mu)^\top\right)}
    _{\text{covariance matrix }C} w \\[6pt]
    &= \max_{w:\|w\|_2=1} w^\top Cw
    && \text{Substitute }C
\end{aligned}
$$

## 4.2 Case example 2D
Suppose we have a 2D dataset where the covariance matrix is:
$$
    C = \begin{bmatrix}
        1 & 0 \\
        0 & 2
    \end{bmatrix}
$$

Use the formulation above, the optimization problem becomes:
$$
    \max_{w :\|w\|_2 = 1} w^\top Cw = \max_{w :\|w\|_2 = 1} \begin{bmatrix}
        w_1 & w_2
    \end{bmatrix}\begin{bmatrix}
        1 & 0 \\
        0 & 2
    \end{bmatrix}\begin{bmatrix}
        w_1 \\
        w_2
    \end{bmatrix} = \max_{w :\|w\|_2 = 1} w_1^2 + 2w_2^2
$$

## 4.3 Eigenvalue Problem
**Definition:** Given a symmetric matrix $$C \in \mathbb{R}^{d\times d}$$, find a vector $$w\in \mathbb{R}^d$$ such that $$\|w\|_2 = 1$$ and:
$$
    Cw = \lambda w
$$
Note: There will be multiple solutions to this problem with different eigenvectors $$w^1,w^2,\ldots,w^d$$ and their corresponding eigenvalues $$\lambda_1,\lambda_2,\ldots,\lambda_d$$. Here we will use the notion where all of the eigenvectors are orthogonal (i.e. $$(w^i)^\top w^j = 0$$ for $$i\neq j$$).

## 4.4 Equivalent to Eigenvalue Problem

**Claim:** The following maximization problem is equivalent to the eigenvalue problem defined above:
$$
    w \in \operatorname*{arg\,max}_{\|u\|_2 = 1} u^\top Cu \iff Cw = \lambda_{\max}w,\quad \|w\|_2=1
$$
**Proof:** We can solve the maximiation problem by setting the gradient of the Lagrangian to zero:
$$
\begin{aligned}
    L(w,\lambda) &= w^\top Cw - \lambda(w^\top w - 1) && \text{Define the Lagrangian} \\
    \frac{\partial L}{\partial w} &= 2Cw - 2\lambda w = 0 && \text{Set the gradient to zero*} \\
    Cw &= \lambda w && \text{Rearrange. Note: w is eigenvector of C} \\
\end{aligned}
$$
Note (*): If $$w$$ is maximum of original problem, then there must exist a $$\lambda$$ such that $$w$$ is a stationary point of the Lagrangian.

We can compute the variance in the principal direction:
$$
    w^\top Cw = w^\top \lambda w = \lambda w^\top w = \lambda
$$

Therefore, the principal direction $$w$$ that maximizes variance is the eigenvector of $$C$$ with the largest eigenvalue $$\lambda$$. In fact, the ordering of the eigenvalues $$\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d$$ corresponds to the ordering of the variance captured by each principal direction $$w^1,w^2,\ldots,w^d$$.

# Classroom Dialog
- **Question:** Is or when is PCA best used for classification task?
- **Answer:** Highest variance is not always the best for classification, sometimes second highest variance is better for classification, depends on the data. PCA is unsupervised, so it does not take into account the labels of the data. PCA preserves variance, but it does not necessarily preserve class separability. Be cautious because sometimes finding the direction that maximizes variance may not be the best for classification.

## 4.5 When to use PCA? Drawbacks of PCA?
**When to use PCA?**
- **Visualization:** Use PCA to reduce to 2D or 3D for visualization.
- **Feature distribution:** Use PCA to analyze variance, mean, distribution, etc.
- **Feature engineering:** Identify independent principal direction, reduce number of features (when you have more features than data points).
- **Data compression**

**Drawbacks of PCA**
- Lose interpretability
- Large variance does not always mean more information/predictability
- Label agnostic, PCA does not take into account the labels of the data so it may not fit labels well
- PCA is a linear method and thus can only capture linear relationships in the data.
- If there are non-linear relationships in the data such as spirals, PCA will not be able to capture the underlying structure of the data.
    


## 4.6 PCA Reconstruction

**Input:** Given $$n$$ data points $$\{x^1,x^2,\ldots,x^n\}$$ in $$\mathbb{R}^d$$ with mean $$\mu$$ and the first $$k$$ principal directions $$w^1,w^2,\ldots,w^k$$ and their corresponding eigenvalues $$\lambda_1,\lambda_2,\ldots,\lambda_k$$. Define their matrices like so: $$W_k = [w^1,w^2,\ldots,w^k]^\top \in \mathbb{R}^{k\times d}$$ and $$\Lambda_k = \text{diag}(\lambda_1,\lambda_2,\ldots,\lambda_k) \in \mathbb{R}^{k\times k}$$.

**Goal:** Recover $$x^i$$ from its reduced representation $$z^i$$.
$$
\begin{align*}
    z^i &= \begin{bmatrix} 
        z^i_1 \\
        z^i_2 \\
        \vdots \\
        z^i_k
    \end{bmatrix} = \begin{bmatrix}
        (w^1)^\top (x^i - \mu) / \sqrt{\lambda_1} \\
        (w^2)^\top (x^i - \mu) / \sqrt{\lambda_2} \\
        \vdots \\
        (w^k)^\top (x^i - \mu) / \sqrt{\lambda_k}
    \end{bmatrix} = \Lambda_k^{-\frac{1}{2}} W_k (x^i - \mu) \in \mathbb{R}^k && \text{Definition of reduced representation} \\
    x^i &\simeq \mu + \sum_{j=1}^{k} z^i_j w^j \sqrt{\lambda_j} && \text{Approximation when } k < d \\
    &= \mu + W_k^\top \Lambda_k^{\frac{1}{2}} z^i && \text{Matrix form}
\end{align*}
$$
