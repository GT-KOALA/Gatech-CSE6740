---
semester: fall-2026
lecture_number: "05"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: yjz0206168-alt
issue: 8
---
## Gaussian Mixture Models

A single Gaussian distribution is unimodal, so it may not be flexible enough when the data has several clusters or several peaks. A Gaussian mixture model (GMM) handles this by modeling the density as a weighted combination of multiple Gaussian components.

Let \(x\in\mathbb{R}^d\), where \(d\) is the dimension of the data. For Gaussian component \(k\), let \(\mu_k\in\mathbb{R}^d\) be its mean and let \(\Sigma_k\) be its covariance matrix. Then

\[
\mathcal{N}(x \mid \mu_k,\Sigma_k)
=
\frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
\exp\left(
-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)
\right).
\]

Here, \(|\Sigma_k|\) denotes the determinant of the covariance matrix.

A mixture of \(K\) Gaussian components is

\[
p(x)
=
\sum_{k=1}^K
\pi_k \mathcal{N}(x\mid \mu_k,\Sigma_k),
\]

where the mixing proportions satisfy

\[
0 < \pi_k < 1,
\qquad
\sum_{k=1}^K \pi_k = 1.
\]

For each component \(k\), the parameters are

- \(\pi_k\): mixing proportion,
- \(\mu_k\): component mean,
- \(\Sigma_k\): component covariance matrix.

The mixing proportion tells us how much probability mass is assigned to each component, while \(\mu_k\) and \(\Sigma_k\) determine the center and shape of that Gaussian.

### Latent component label

For each observation \(x^i\), introduce a latent variable

\[
z^i \in \{1,\ldots,K\}.
\]

If \(z^i=k\), then \(x^i\) is associated with component \(k\). The joint probability is

\[
p(x^i,z^i=k\mid\theta)
=
\pi_k
\mathcal{N}(x^i\mid\mu_k,\Sigma_k),
\]

where

\[
\theta
=
\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K.
\]

Since the component label is not observed, the marginal density for \(x^i\) is obtained by summing over all possible component labels:

\[
p(x^i\mid\theta)
=
\sum_{k=1}^K
p(x^i,z^i=k\mid\theta).
\]

The key point is that \(z^i\) is hidden, so we only observe the marginal probability of \(x^i\). If the labels \(z^i\) were known, fitting each Gaussian component would be much easier.

## Learning the Parameters with Maximum Likelihood

Given data

\[
D=\{x^1,\ldots,x^n\},
\]

we want to choose the GMM parameters that maximize the likelihood of the observed data:

\[
\theta^*
=
\arg\max_\theta \ell(\theta;D),
\]

with

\[
\ell(\theta;D)
=
\log p(D\mid\theta).
\]

Assuming the observations are independent,

\[
\ell(\theta;D)
=
\log
\prod_{i=1}^n
p(x^i\mid\theta).
\]

Using the latent component label,

\[
p(x^i\mid\theta)
=
\sum_{k=1}^K
p(x^i,z^i=k\mid\theta),
\]

so

\[
\ell(\theta;D)
=
\sum_{i=1}^n
\log
\left(
\sum_{k=1}^K
p(x^i,z^i=k\mid\theta)
\right).
\]

For a Gaussian mixture model,

\[
\ell(\theta;D)
=
\sum_{i=1}^n
\log
\left(
\sum_{k=1}^K
\pi_k
\mathcal{N}(x^i\mid\mu_k,\Sigma_k)
\right).
\]

The annoying part is the sum inside the logarithm, since the component parameters cannot be separated easily. This is why the lecture next reviews convexity and Jensen's inequality.

---

## Convex Sets

A set \(A\) is convex if, for any \(x,y\in A\) and any \(0\le\alpha\le1\),

\[
\alpha x+(1-\alpha)y \in A.
\]

Geometrically, the line segment joining any two points in the set stays completely inside the set.

### Convex cones

A set \(C\) is a convex cone if, for any \(x_1,x_2\in C\) and any \(\theta_1,\theta_2\ge0\),

\[
\theta_1x_1+\theta_2x_2\in C.
\]

Compared with the usual convex-set definition, the coefficients do not have to sum to one.

### Hyperplanes and halfspaces

A hyperplane has the form

\[
\{x \mid a^T(x-x_0)=0,\ a\neq0\}.
\]

A halfspace has the form

\[
\{x \mid a^T(x-x_0)\le0,\ a\neq0\}.
\]

Both are convex.

### Euclidean balls

A Euclidean ball centered at \(x_c\) with radius \(r\) is

\[
B(x_c,r)
=
\{x \mid \|x-x_c\|_2\le r\}.
\]

### Ellipsoids

An ellipsoid can be written as

\[
E
=
\left\{
x
\mid
(x-x_c)^T P^{-1}(x-x_c)\le1
\right\}.
\]

The eigenvectors and eigenvalues determine the directions and lengths of the semi-axes.

### Polyhedra

A polyhedron is the solution set of finitely many linear equalities and inequalities. One general form is

\[
P
=
\left\{
x
\mid
a_j^Tx\le b_j,\ j=1,\ldots,m,
\quad
c_j^Tx=d_j,\ j=1,\ldots,p
\right\}.
\]

Since halfspaces and hyperplanes are convex, their intersection is also convex.

---

## Convex and Concave Functions

A function \(f:\mathbb{R}^n\to\mathbb{R}\) is convex if its domain is convex and

\[
f(\theta x+(1-\theta)y)
\le
\theta f(x)+(1-\theta)f(y)
\]

for all \(x,y\) in the domain and all \(0\le\theta\le1\).

Geometrically, the line segment joining \((x,f(x))\) and \((y,f(y))\) lies above the graph of a convex function.

A function is concave if the inequality is reversed:

\[
f(\theta x+(1-\theta)y)
\ge
\theta f(x)+(1-\theta)f(y).
\]

For a concave function, the same line segment lies below the graph.

### First-order characterization

If \(f\) is differentiable, convexity can be characterized by

\[
f(y)
\ge
f(x)+\nabla f(x)^T(y-x)
\]

for all \(x,y\) in the domain.

So for a convex function, the tangent line or tangent plane at \(x\) gives a global lower bound on the function.

### Second-order characterization

If \(f\) is twice differentiable on a convex domain, then \(f\) is convex if and only if

\[
\nabla^2 f(x)\succeq0
\]

for every \(x\) in its domain.

The notation \(A\succeq0\) means that \(A\) is positive semidefinite, i.e.,

\[
y^TAy\ge0
\]

for every vector \(y\).

For example,

\[
f(x)=\frac{1}{2}x^TAx
\]

is convex when \(A\succeq0\).

### Examples from lecture

Some representative convex and concave functions are:

- \(e^{ax}\) is convex for any \(a\in\mathbb{R}\).
- \(x^a\) is convex on \(\mathbb{R}_{++}\) when \(a\ge1\) or \(a\le0\).
- \(x^a\) is concave on \(\mathbb{R}_{++}\) when \(0\le a\le1\).
- \(|x|^p\) is convex for \(p\ge1\).
- \(\log x\) is concave on \(\mathbb{R}_{++}\).
- \(x\log x\) is convex.
- Every norm is convex.
- \(\max(x_1,\ldots,x_n)\) is convex.

The important example for the rest of this lecture is \(\log x\), because its concavity gives the direction of Jensen's inequality that we need.

---

## Jensen's Inequality

Jensen's inequality describes how a convex or concave function behaves when applied to an average.

For a concave function \(f\),

\[
f\left(\sum_i a_i x_i\right)
\ge
\sum_i a_i f(x_i),
\]

where

\[
a_i\ge0,
\qquad
\sum_i a_i=1.
\]

For a convex function, the inequality goes in the opposite direction.

A useful probabilistic form is

\[
f(\mathbb{E}[X])
\ge
\mathbb{E}[f(X)]
\]

when \(f\) is concave.

Since \(\log x\) is concave,

\[
\log(\mathbb{E}[X])
\ge
\mathbb{E}[\log X].
\]

### Small numerical example

Take the concave function

\[
f(x)=\log x
\]

and two values \(x_1=1\) and \(x_2=4\), each with weight \(1/2\).

Jensen's inequality gives

\[
\log\left(
\frac{1}{2}\cdot1+\frac{1}{2}\cdot4
\right)
\ge
\frac{1}{2}\log1+\frac{1}{2}\log4.
\]

The left-hand side is

\[
\log(2.5),
\]

while the right-hand side is

\[
\frac{1}{2}\log4=\log2.
\]

Since \(2.5>2\),

\[
\log(2.5)>\log2,
\]

so the inequality has the expected direction.

This form of Jensen's inequality will be useful later when working with the GMM log-likelihood.
