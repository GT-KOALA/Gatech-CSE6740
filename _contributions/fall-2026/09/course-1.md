---
semester: fall-2026
lecture_number: "09"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: ish-unde
issue: 27
---
## Lecture 09: Kernel Methods 

## Kernel Methods: Nonlinear Representations and the Kernel Function

*CSE/ISyE 6740, Lecture 9. This section covers the material from the ridge regression recap up to kernel regression. AI was used to help in formatting of the notes.*

### Notation: (was useful for me to keep track of everything!

- $n$: number of training points; $d$: dimension of the raw input.

- $D_{train} = \lbrace (x^{(i)}, y^{(i)})\rbrace _{i \in \lbrace 1, \dots, n\rbrace }$: the training set, with $x^{(i)} \in \mathbb{R}^d$ and $y^{(i)} \in \mathbb{R}$.

- $X = [x^{(1)}, \dots, x^{(n)}]^\top \in \mathbb{R}^{n \times d}$: the data matrix (one row per data point); $y = [y^{(1)}, \dots, y^{(n)}]^\top \in \mathbb{R}^n$.

- $\phi : \mathbb{R}^d \to \mathbb{R}^{d'}$: a (usually nonlinear) feature map or embedding; $d'$ is the dimension of the new feature space and is typically much larger than $d$ (possibly infinite).

- $\tilde{x} = \phi(x)$: the expanded feature vector, and $\tilde{X} = [\phi(x^{(1)}), \dots, \phi(x^{(n)})]^\top \in \mathbb{R}^{n \times d'}$.

- $k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$: a kernel function, with $k(x, x') = \phi(x)^\top \phi(x')$.

- $K \in \mathbb{R}^{n \times n}$ with $K_{ij} = k(x^{(i)}, x^{(j)})$: the kernel matrix or Gram matrix.

- $\lambda$: the ridge regularization strength.

### 1. Starting point: linear and ridge regression

We assume the label is a linear function of the features plus noise:

$$
y = \theta^\top x + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2).
$$

Ridge regression picks $\theta$ by minimizing the mean squared error plus a penalty on large weights:

$$
\hat{\theta} = \arg\min_\theta \; L(\theta) + \frac{\lambda}{2}\Vert \theta\Vert ^2,
\qquad
L(\theta) := \frac{1}{n}\sum_{i=1}^n \left(y^{(i)} - \theta^\top x^{(i)}\right)^2 .
$$

**Derivation.** To derive this, write the loss in matrix form, $L(\theta) = \frac{1}{n}\Vert y - X\theta\Vert ^2 = \frac{1}{n}(y^\top y - 2\theta^\top X^\top y + \theta^\top X^\top X \theta)$. then, differentiate term by term:

- $\frac{\partial}{\partial \theta}\left(-\frac{2}{n}\theta^\top X^\top y\right) = -\frac{2}{n}X^\top y$ (this term pulls $\theta$ toward directions correlated with the labels),

- $\frac{\partial}{\partial \theta}\left(\frac{1}{n}\theta^\top X^\top X\theta\right) = \frac{2}{n}X^\top X\theta$ (this term measures how much the current predictions already explain),

- $\frac{\partial}{\partial \theta}\left(\frac{\lambda}{2}\Vert \theta\Vert ^2\right) = \lambda\theta$ (this is the penalty term that shrinks $\theta$ toward zero).

then, set the gradient to zero:

$$
\frac{\partial}{\partial\theta}\left(L(\theta) + \frac{\lambda}{2}\Vert \theta\Vert ^2\right)
= -\frac{2}{n}X^\top y + \frac{2}{n}X^\top X\theta + \lambda\theta = 0
$$

$$
\Longrightarrow\quad
\theta^* = \left(X^\top X + \lambda I\right)^{-1} X^\top y .
$$

(only solving the middle equation will give you $(X^\top X + \tfrac{n\lambda}{2} I)^{-1}X^\top y$; the constant $\tfrac{n}{2}$ is absorbed into $\lambda$, since $\lambda$ is a tuning knob that we choose anyway.)

Two things I think are important to remember. First, adding $\lambda I$ makes $X^\top X + \lambda I$ invertible even when $X^\top X$ is not, which is why ridge regression is numerically stable. Second, the prediction $\theta^\top x$ is just a weighted sum of the raw inputs: the model is degree 1 in $x$. If the true relationship between $x$ and $y$ is not linear, no choice of $\theta$ can capture it. That limitation is what motivates this lecture.

### 2. Polynomial regression: nonlinear in $x$, linear in $\theta$

Suppose $x \in \mathbb{R}$ is a scalar and we want to fit a degree-$d$ polynomial:

$$
y = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d + \epsilon .
$$

This looks nonlinear, but notice that $y$ is still linear in the parameters $\theta$. Define

$$
\tilde{x} = \phi(x) = [1, x, x^2, \dots, x^d]^\top \in \mathbb{R}^{d+1},
\qquad
\theta = [\theta_0, \theta_1, \dots, \theta_d]^\top .
$$

Then the model simply becomes $y = \theta^\top \tilde{x} + \epsilon$. We have one input, but we have turned it into many features, so the curve in the original $x$-space becomes a plane in $\tilde{x}$-space. This is now a high-dimensional “linear” regression problem on $\tilde{x}$.

**Least squares for polynomials.** Minimize $L(\theta) = \frac{1}{n}\sum_{i=1}^n (y^{(i)} - \theta^\top \tilde{x}^{(i)})^2$. Stack the expanded features as rows of $\tilde{X} = [\tilde{x}^{(1)}, \dots, \tilde{x}^{(n)}]^\top \in \mathbb{R}^{n \times (d+1)}$. The derivation is the same as the earlier one, but now only with $\tilde{X}$ in place of $X$ (and no penalty):

$$
\frac{\partial L(\hat\theta)}{\partial\theta} = -\frac{2}{n}\tilde{X}^\top y + \frac{2}{n}\tilde{X}^\top\tilde{X}\theta = 0
\;\;\Longrightarrow\;\;
\hat{\theta} = \left(\tilde{X}^\top\tilde{X}\right)^{-1}\tilde{X}^\top y .
$$

Once the features are fixed, nothing about the algorithm changes. All the nonlinearity now lives in $\phi$.

**Choosing the degree.** Each maximal degree $d$ gives a different solution. In the lecture’s plots (degrees 0 through 5 on the same data), degree 0 is a flat line and degree 1 is a tilted line, both of which miss the U-shape of the data (underfitting). From degree 2 upward the fit follows the curve. Pushing the degree much higher lets the polynomial bend to chase individual noisy points (overfitting). This is exactly why we pair feature expansion with the ridge penalty $\lambda\Vert \theta\Vert ^2$ and pick hyperparameters by validation.

### 3. Multivariate polynomials: the feature explosion

Now let $x = [x_1, \dots, x_d]^\top \in \mathbb{R}^d$. A degree-$k$ polynomial model must include not only each $x_j$ but every interaction between variables:

$$
\begin{aligned}
y = \theta_0 &+ \theta_1 x_1 + \dots + \theta_d x_d \\
  &+ \theta_{1,1}x_1x_1 + \theta_{1,2}x_1x_2 + \dots
\\
  &+ \theta_{i_1 i_2 \dots i_m}x_{i_1}x_{i_2}\cdots x_{i_m} + \dots + \epsilon
  \;=\; \theta^\top\tilde{x},
\end{aligned}
$$

with $\tilde{x} = \phi(x) = [1, x_1, \dots, x_{i_1}x_{i_2}\cdots x_{i_k}]^\top \in \mathbb{R}^{d'}$. This is also a “linear” regression in $\tilde{x}$.

The problem is the size of $d'$. The number of distinct monomials of degree exactly $k$ in $d$ variables is $\binom{k+d-1}{k}$ (this is the count shown on the slide), and counting all degrees from 0 up to $k$ gives $\binom{d+k}{k}$. Either way it grows roughly like $d^k$. For example, with $d = 100$ raw features:

| degree $k$ | features up to degree $k$, $\binom{100+k}{k}$ |
|:----------:|:---------------------------------------------:|
|     1      |                      101                      |
|     2      |                     5,151                     |
|     3      |                    176,851                    |
|     5      |           $\approx 9.7 \times 10^7$           |

Ridge regression on $\tilde{X}$ requires inverting the $d' \times d'$ matrix $\tilde{X}^\top\tilde{X} + \lambda I$, which costs $O(d'^3)$. Many inputs produce very many features, and explicit expansion quickly becomes infeasible.

### 4. Generalizing: arbitrary feature maps

Polynomials are just one choice, you can also chose other feature maps. In general, the definition for any nonlinear mapping

$$
\phi : \mathbb{R}^d \to \mathbb{R}^{d'}, \qquad x \mapsto \tilde{x} = \phi(x),
$$

that sends $d$-dimensional points to $d'$-dimensional points. The recipe for nonlinear regression with a given embedding is:

1. Start with $D_{train} = \lbrace (x^{(i)}, y^{(i)})\rbrace _{i \in \lbrace 1,\dots,n\rbrace }$.

2. Build a new dataset $D'_{train} = \lbrace (\phi(x^{(i)}), y^{(i)})\rbrace $ by transforming every input.

3. Run ordinary ridge regression on $D'_{train}$: with $\tilde{X} = [\phi(x^{(1)}), \dots, \phi(x^{(n)})]^\top$ and $y = [y^{(1)}, \dots, y^{(n)}]^\top$,

$$
\theta^* = \left(\tilde{X}^\top\tilde{X} + \lambda I\right)^{-1}\tilde{X}^\top y .
$$

The penalty $\lambda I$ matters even more here than before: with $d' \gg n$, the matrix $\tilde{X}^\top\tilde{X}$ has rank at most $n$ and is singular, and the model has enough freedom to overfit badly.

**Why this helps (in a geometric sense).** A relationship that is curved in the input space can become straight in a well-chosen feature space. The simplest example is 1-D classification where one class sits in the middle and the other on both sides. No single threshold on $x$ separates them, but mapping $x \mapsto \phi(x) = (x, x^2)$ lifts the points onto a parabola, and a horizontal line separates them perfectly.

We also should consider cost. A good $\phi$ can be enormous (or infinite-dimensional), so we want the benefits of $\phi$ without ever computing it.

### 5. The kernel trick

**Definition.** A kernel $k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ measures the nonlinear similarity between two points $x^{(i)}, x^{(j)}$. The main idea is to choose $k$ so that

$$
k(x^{(i)}, x^{(j)}) = \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle = \phi(x^{(i)})^\top\phi(x^{(j)}),
$$

and then **replace every inner product** in an algorithm with a call to $k$. If the algorithm only ever touches the data through inner products, we never need to build $\phi$ at all.

**What helps me think about it:** An inner product is a similarity score: it is large when two vectors point the same way. So $k(x, x')$ answers the question “how similar are $x$ and $x'$ after the nonlinear expansion?” Because the similarity is computed in the expanded space, it can capture curved relationships, yet it is evaluated with a cheap formula on the original $d$-dimensional vectors.

**example** Let $x, z \in \mathbb{R}$ be scalars, and use the polynomial kernel of degree up to 2, $k(x,z) = (xz + 1)^2$. Expanding,

$$
(xz + 1)^2 = x^2z^2 + 2xz + 1 = \phi(x)^\top\phi(z),
$$

where

$$
\phi(x) = \left(1,\; \sqrt{2}\,x,\; x^2\right).
$$

These are the same features $1, x, x^2$ from polynomial regression in Section 2, just with a $\sqrt{2}$ scaling on the middle term. Plug in numbers: $x = 2$, $z = 3$.

- **Kernel route:** $xz + 1 = 6 + 1 = 7$, so $k(x,z) = 49$. (One multiplication, one addition, one square.)

- **Feature route:** $\phi(2) = (1, 2\sqrt{2}, 4)$, $\phi(3) = (1, 3\sqrt{2}, 9)$, so $\phi(2)^\top\phi(3) = 1 + 12 + 36 = 49$.

Both give 49, but the kernel route never formed $\phi$. In general, $(x^\top z)^k$ costs $O(d)$ to compute, while the explicit features has about $d^k$ entries --> this is the biggest advantage!

### 6. Which functions are valid kernels?

We want $k(x,x') = \phi(x)^\top\phi(x')$, but usually we do not know $\phi$. How can we check that a $\phi$ exists?

Given a dataset $D = \lbrace (x^{(i)}, y^{(i)})\rbrace _{i \in \lbrace 1,\dots,n\rbrace }$ (the labels are optional here), compute every pairwise kernel value and form the $n \times n$ kernel (Gram) matrix:

$$
K = \begin{bmatrix}
k(x^{(1)},x^{(1)}) & k(x^{(1)},x^{(2)}) & \cdots & k(x^{(1)},x^{(n)})\\
k(x^{(2)},x^{(1)}) & k(x^{(2)},x^{(2)}) & \cdots & k(x^{(2)},x^{(n)})\\
\vdots & & \ddots & \vdots\\
k(x^{(n)},x^{(1)}) & k(x^{(n)},x^{(2)}) & \cdots & k(x^{(n)},x^{(n)})
\end{bmatrix}.
$$

Note that if $\phi$ were known, $K = \tilde{X}\tilde{X}^\top$: it is the matrix of all pairwise inner products in feature space.

**Criterion.** $k(x,x')$ is a kernel function if and only if the matrix $K$ is **positive semidefinite** (for every choice of dataset):

$$
\forall v \in \mathbb{R}^n, \quad v^\top K v \ge 0 .
$$

($K$ must also be symmetric, which holds automatically when $k(x,x') = k(x',x)$.)

**Why we need K to follow those criterion:** If $k(x,x') = \phi(x)^\top\phi(x')$, then for any $v$,

$$
v^\top K v = \sum_{i,j} v_i v_j\, \phi(x^{(i)})^\top\phi(x^{(j)})
$$

$$
= \Big(\sum_i v_i\phi(x^{(i)})\Big)^\top\Big(\sum_j v_j\phi(x^{(j)})\Big)
= \Big\lVert \sum_i v_i\phi(x^{(i)})\Big\rVert ^2 \ge 0 .
$$

A squared length can never be negative. The converse (PSD implies some $\phi$ exists) is the deeper half of the theorem: it guarantees that a function passing this test really is an inner product in some feature space, even if we never write that space down.

The SVM dual and ridge regression rely on the quadratic form being nonnegative (that is what makes the problems convex with a unique, well-behaved solution). A “kernel” that breaks PSD-ness can make the optimization unbounded or meaningless.

**For non-example:** Try $k(x,x') = -\Vert x - x'\Vert ^2$ on two distinct points. The diagonal entries are 0 and the off-diagonal entries are $-\Vert x^{(1)} - x^{(2)}\Vert ^2 < 0$. With $v = (1,1)$, $v^\top K v = -2\Vert x^{(1)} - x^{(2)}\Vert ^2 < 0$, so this is not a valid kernel. Compare with RBF, which passes: its diagonal entries are 1 and its off-diagonal entries lie in $(0,1)$.

### 7. Example: kernel SVM

The SVM dual from the previous lecture is a natural fit, because the data appear only through inner products $x^{(i)\top}x^{(j)}$.

**Dual problem:**

$$
\max_\alpha \; g(\alpha) := \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i\alpha_j y^{(i)}y^{(j)}\, x^{(i)\top}x^{(j)}
$$

$$
\text{s.t.}\quad C \ge \alpha_i \ge 0\;\forall i,\quad \sum_{i=1}^n \alpha_i y^{(i)} = 0 .
$$

**Dual problem + embedding $\phi$:** apply the SVM to $\phi(x^{(i)})$ instead of $x^{(i)}$. The only change is in the inner product:

$$
\max_\alpha \; \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i\alpha_j y^{(i)}y^{(j)}\, \phi(x^{(i)})^\top\phi(x^{(j)})
\quad \text{(same constraints)}.
$$

**Dual problem + kernel $k$:** replace the inner product with the kernel:

$$
\max_\alpha \; \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i\alpha_j y^{(i)}y^{(j)}\, k(x^{(i)}, x^{(j)})
\quad \text{(same constraints)}.
$$

The constraints do not involve $x$ at all, so they are untouched. The optimization problem has $n$ variables no matter how large $d'$ is.

**Making predictions without $\phi$.** From the SVM lecture, the primal weights are $w = \sum_i \alpha_i y^{(i)} x^{(i)}$. With an embedding this becomes $w = \sum_i \alpha_i y^{(i)} \phi(x^{(i)})$, which may be infinite-dimensional, but we never need $w$ itself:

$$
f(x) = w^\top\phi(x) + b = \sum_{i=1}^n \alpha_i y^{(i)}\, k(x^{(i)}, x) + b .
$$

Only support vectors ($\alpha_i > 0$) contribute. The offset $b$ can also be computed with kernels, e.g. from any support vector with $0 < \alpha_j < C$: $b = y^{(j)} - \sum_i \alpha_i y^{(i)} k(x^{(i)}, x^{(j)})$.

**The advantage:** we never need to know or specify the embedding $\phi$; choosing $k$ implicitly chooses it. Geometrically, kernel SVM implicitly maps the data into a nonlinear feature space and finds a *linear* max-margin boundary there. Mapped back to the input space, that boundary is curved, as in the lecture’s S-shaped boundary example that becomes a straight line under $\phi$.

### 8. Going back to linear methods

|                | Linear method             | Kernelized method                           |
|:---------------|:--------------------------|:--------------------------------------------|
| Representation | raw $x \in \mathbb{R}^d$  | $\phi(x) \in \mathbb{R}^{d'}$, never formed |
| Similarity     | $x^\top x'$               | $k(x,x') = \phi(x)^\top\phi(x')$            |
| Boundary / fit | linear in $x$             | linear in $\phi(x)$, curved in $x$          |
| Data summary   | $X^\top X$ ($d \times d$) | $K$ ($n \times n$)                          |

The linear kernel $k(x,x') = x^\top x'$ (i.e., $\phi(x) = x$) recovers the original linear method exactly, so kernel methods generalize what we had. The most basic thing to remember is the same: (1) write the algorithm so the data enter only through inner products, (2) swap those for $k$, and (3) pick $k$ to encode the similarity we care about.

