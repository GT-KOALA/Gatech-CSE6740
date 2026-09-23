---
semester: fall-2026
lecture_number: "06"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: lyh7507
issue: 13
---

This note covers Lecture 06, slides 1–18, with the main material presented in slides 1–17. This part of the lecture centers on a fundamental question: how can we determine whether a feature is useful for predicting a class label? The lecture begins with the prediction task and considers what it means for a feature to provide useful information. It then introduces uncertainty as a way to formalize this idea. Within this framework, entropy measures the uncertainty in the class label, while mutual information measures how much that uncertainty is reduced after observing a feature. These quantities can therefore be used to compare and rank features before fitting a classifier.

In organizing these notes, I also included several derivations and simple Python examples to illustrate the relationships among entropy, mutual information, and feature scoring, and to make the underlying calculations more concrete.

## 1. Supervised Classification and Feature Selection (Slides 3–8)

### 1.1 Data and Notation for Supervised Classification

In supervised classification, we learn from training samples that pair an input vector with an observed class label. A dataset of these pairs has the form:

$$
D=\left\lbrace \left(\mathbf{x}^{(r)},y^{(r)}\right)\right\rbrace_{r=1}^{n},
\qquad
\mathbf{x}^{(r)}\in\mathbb{R}^{d},
\qquad
y^{(r)}\in\lbrace 1,\ldots,K\rbrace
$$

Here, $n$ is the number of samples, $d$ is the number of features, and $K$ is the number of classes. The superscript $r$ identifies a sample, while the subscript $i$ in $x_i^{(r)}$ identifies a feature of that sample.

When discussing the distributions underlying these observations, we use uppercase $X_i$ and $Y$ for the feature and label random variables, and lowercase symbols for their values. The binary examples use class labels $0$ and $1$.

From these labeled samples, we learn a mapping $f:\mathcal{X}\to\mathcal{Y}$ that predicts the label associated with a new input.

### 1.2 The Purpose of Feature Selection

In the email example on slide 6, we represent each email using indicators of whether particular words or phrases occur. Some of these indicators help distinguish spam from other emails, whereas others are equally common across classes and contribute little to that distinction.

Feature selection retains a subset of these original variables rather than using every available input. This differs from PCA, which combines inputs to construct new coordinates. By keeping a smaller set of clearly defined variables, we can more readily identify the information used by the model.

To decide which variables to retain, we need to assess how much each one tells us about the label. Before we observe a feature, we use the overall label distribution to describe the uncertainty in our prediction. Once we know the feature's value, we instead use the label distribution conditional on that value. The criterion on slide 8 compares the uncertainty in these two cases and evaluates the feature by the average reduction.

## 2. Measuring Uncertainty with Entropy (Slides 9–11)

### 2.1 From the Label Distribution to a Measure of Uncertainty

We begin with the uncertainty that remains before we use any features. At this stage, our prediction depends only on the probabilities of the possible classes, which we denote by:

$$
p_k=P(Y=k),\qquad \sum_{k=1}^{K}p_k=1
$$

If most of the probability is assigned to one class, predicting that class will often be correct, even without observing an input. By contrast, when all classes are equally likely, the label distribution gives us no reason to favor one class over another, so the label is more uncertain.

To quantify this uncertainty, we assign an outcome with probability $p_k$ the information content $-\log_2 p_k$, which is larger for less likely outcomes. Averaging this quantity over the possible labels gives Shannon entropy:

$$
H(Y)=\sum_{k=1}^{K}p_k\bigl(-\log_2 p_k\bigr)
=-\sum_{k=1}^{K}p_k\log_2 p_k
$$

All logarithms use base 2, so discrete entropy and mutual information are measured in bits. For zero-probability outcomes, we use the convention $0\log_2 0=0$.

On slide 9, the coding interpretation gives the same expression another meaning. Here, $-\log_2 p_k$ is the ideal length associated with an outcome, and entropy averages these lengths to describe the theoretical limit on the number of bits per symbol in efficient lossless encoding.

### 2.2 High and Low Entropy

The contrast between concentrated and uniform distributions corresponds to the low- and high-entropy cases on slide 10. For a fixed set of $K$ possible classes, entropy satisfies:

$$
0\leq H(Y)\leq\log_2 K
$$

When one class has probability 1, the label is certain and entropy reaches the lower bound. At the other extreme, all classes have probability $\frac{1}{K}$, giving the upper bound $H(Y)=\log_2 K$. In particular, two equally likely classes give a label entropy of 1 bit.

We can see the same pattern in the six-toss coin example from slide 11. The table lists several empirical distributions and their entropies, including the balanced case:

| Heads | Tails | Empirical Probability of Heads | Entropy (bits) |
|---|---|---|---|
| 0 | 6 | $0$ | $0$ |
| 1 | 5 | $1/6$ | $0.6500$ |
| 2 | 4 | $2/6$ | $0.9183$ |
| 3 | 3 | $1/2$ | $1.0000$ |

For the second row, substituting the observed proportions of heads and tails into the entropy formula, we have:

$$
H=-\frac16\log_2\frac16-\frac56\log_2\frac56
\approx0.6500
$$

These calculations describe the empirical distributions obtained from the observations. Thus, six tails give zero empirical entropy, but they do not establish that the coin's true probability of heads is zero.

## 3. Continuous Variables and Differential Entropy (Slide 9)

The entropy formula above sums over discrete outcomes, but some variables, such as attendance, can be modeled as continuous. Slide 9 introduces differential entropy for this case. If $f_Z$ is the probability density of a continuous variable $Z$, its differential entropy is:

$$
h(Z)=-\int f_Z(z)\log_2 f_Z(z)\,dz
$$

Here, lowercase $h$ distinguishes differential entropy from discrete entropy. Because $f_Z(z)$ is a density rather than the probability of the point $z$, probabilities must be obtained by integrating over a range of values. For example:

$$
P(a\leq Z\leq b)=\int_a^b f_Z(z)\,dz
$$

For classification, however, a continuous feature $X_i$ does not make the label $Y$ continuous. We still measure label uncertainty with $H(Y)$; it is the average over feature values that must be computed using a density.

## 4. Conditional Entropy (Slide 12)

### 4.1 Fixing the Observed Feature Value

To measure uncertainty after observing a feature, we replace the overall label probabilities with conditional probabilities. When we observe $X=x$, the relevant label distribution is $P(Y=y\mid X=x)$. Applying the entropy definition to this distribution gives

$$
H(Y\mid X=x)
=-\sum_y p(y\mid x)\log_2 p(y\mid x)
$$

This calculation describes the uncertainty after one particular observation. Other values of $X$ may leave different amounts of uncertainty about $Y$, so a single observation does not yet describe the feature as a whole.

### 4.2 Averaging over Possible Feature Values

To obtain one measure for the feature, we average the label entropies across its possible values, weighting each by the probability of observing that value. For discrete $X$, this gives

$$
H(Y\mid X)
=\sum_x p_X(x)H(Y\mid X=x)
=-\sum_x p_X(x)\sum_y p(y\mid x)\log_2 p(y\mid x)
$$

When $X$ is continuous, we perform the same averaging with its density, replacing the outer sum with an integral:

$$
H(Y\mid X)
=-\int f_X(x)
\left[\sum_{k=1}^{K}P(Y=k\mid X=x)
\log_2 P(Y=k\mid X=x)\right]dx
$$

The inner sum remains unchanged because the label is still discrete; only the average over feature values becomes an integral. When the target is also continuous, the inner calculation changes as well. That case, also shown on slide 12, is covered in Supplement A.

## 5. A Joint Distribution Worked Step by Step (Slides 13–14)

We can obtain both the conditional probabilities and their averaging weights from a joint probability table. Slides 13–14 use the following joint distribution, shown here with its marginal probabilities:

| $X$ | $Y=0$ | $Y=1$ | $p_X(x)$ |
|---|---|---|---|
| $0$ | $1/4$ | $1/4$ | $1/2$ |
| $1$ | $1/2$ | $0$ | $1/2$ |
| $p_Y(y)$ | $3/4$ | $1/4$ | $1$ |

Each interior cell contains a joint probability, so the upper-left entry $1/4$ represents $P(X=0,Y=0)$. To obtain the conditional probability $P(Y=0\mid X=0)$, we must divide this entry by its row total. The row and column totals give the marginal probabilities of $X$ and $Y$, respectively.

### 5.1 Joint Entropy of a Variable Pair

Before conditioning on either variable, we can apply the entropy definition to the pair $(X,Y)$ as a single outcome. Its possible values are $(0,0)$, $(0,1)$, $(1,0)$, and $(1,1)$, so the joint entropy on slide 13 is:

$$
\begin{aligned}
H(X,Y)
&=-\sum_{x,y}p(x,y)\log_2 p(x,y)\\
&=-\left(\frac14\log_2\frac14
+\frac14\log_2\frac14
+\frac12\log_2\frac12\right)\\
&=1.5\text{ bits}
\end{aligned}
$$

The zero-probability outcome contributes zero and is omitted from the expanded sum.

### 5.2 Computing $H(Y\mid X)$ by Row Normalization and Averaging

To condition on $X=x$, we restrict attention to the corresponding row and divide each entry by that row's total probability:

$$
P(Y=y\mid X=x)=\frac{P(X=x,Y=y)}{P(X=x)}
$$

For the first row, where $X=0$, we have:

$$
P(Y=0\mid X=0)=\frac{1/4}{1/2}=\frac12,
\qquad
P(Y=1\mid X=0)=\frac{1/4}{1/2}=\frac12
$$

The two labels are equally likely within this row, so $H(Y\mid X=0)=1$. In the second row, where $X=1$, the conditional probabilities are $1$ and $0$, giving $H(Y\mid X=1)=0$.

Since each row occurs with probability $1/2$, we weight the two entropies equally:

$$
H(Y\mid X)
=\frac12H(Y\mid X=0)+\frac12H(Y\mid X=1)
=\frac12(1)+\frac12(0)=0.5\text{ bits}
$$

The same row probability appears in both stages of the calculation. It first normalizes the row into a conditional distribution, then weights that distribution's entropy in the overall average.

### 5.3 Computing $H(X\mid Y)$ by Column Normalization and Averaging

For the reverse calculation, we condition on $Y$ and work with columns instead of rows. When $Y=0$, normalizing the first column, we have:

$$
P(X=0\mid Y=0)=\frac{1/4}{3/4}=\frac13,
\qquad
P(X=1\mid Y=0)=\frac{1/2}{3/4}=\frac23
$$

In the second column, where $Y=1$, $X$ must be 0, so the conditional entropy is zero. We then average the two column entropies using their respective column probabilities:

$$
\begin{aligned}
H(X\mid Y)
&=\frac34\left(-\frac13\log_2\frac13
-\frac23\log_2\frac23\right)+\frac14(0)\\
&\approx0.6887\text{ bits}
\end{aligned}
$$

These results show that $H(Y\mid X)$ and $H(X\mid Y)$ need not be equal, since they measure uncertainty about different variables after different observations.

## 6. Mutual Information (Slide 15)

### 6.1 Mutual Information and Information Gain

We measure the information supplied by a feature by comparing the label entropy $H(Y)$ with the average entropy $H(Y\mid X)$ that remains after observing it. Their difference is the mutual information:

$$
I(X;Y)=H(Y)-H(Y\mid X)
$$

In the feature-selection setting, the lecture also refers to this reduction as **information gain**. For a discrete label, it satisfies:

$$
0\leq I(X;Y)\leq H(Y)
$$

The lower bound occurs when $X$ and $Y$ are independent, because observing $X$ leaves the label distribution unchanged. At the other extreme, a feature that completely determines the label leaves $H(Y\mid X)=0$, so its information gain equals $H(Y)$.

Returning to the joint distribution in Section 5, we first compute the label entropy from its marginal probabilities:

$$
H(Y)=-\frac34\log_2\frac34-\frac14\log_2\frac14
\approx0.8113\text{ bits}
$$

Subtracting the conditional entropy calculated earlier, we have:

$$
I(X;Y)\approx0.8113-0.5=0.3113\text{ bits}
$$

The reduction applies to the average over observations, not necessarily to each observation separately. In this example, observing $X=0$ raises label entropy from about $0.8113$ bits to $1$ bit, whereas observing $X=1$ removes the uncertainty entirely. Averaging these two cases leaves $0.5$ bits of uncertainty. The inequality $H(Y\mid X)\leq H(Y)$ holds even though the first observation makes the label less certain.

### 6.2 Symmetry of Mutual Information

Although conditional entropy need not be symmetric, mutual information is. To see why, we use the factorization $p(x,y)=p_X(x)p(y\mid x)$ to expand the joint entropy:

$$
\begin{aligned}
H(X,Y)
&=-\sum_{x,y}p(x,y)\log_2\bigl[p_X(x)p(y\mid x)\bigr]\\
&=-\sum_x p_X(x)\log_2 p_X(x)
-\sum_x p_X(x)\sum_y p(y\mid x)\log_2 p(y\mid x)\\
&=H(X)+H(Y\mid X)
\end{aligned}
$$

Factoring the same joint distribution in the reverse direction gives $H(X,Y)=H(Y)+H(X\mid Y)$. These two forms of the entropy chain rule let us express mutual information in terms of the marginal and joint entropies:

$$
\begin{aligned}
I(X;Y)
&=H(X)+H(Y)-H(X,Y)\\
&=H(X)-H(X\mid Y)\\
&=I(Y;X)
\end{aligned}
$$

The two conditional entropies may differ, but each is subtracted from the entropy of its own target variable. In our example, $H(X)=1$, so the reverse calculation gives $1-0.6887\approx0.3113$ bits, matching the result obtained from the label entropy.

## 7. Deriving the Feature Score (Slides 16–17)

### 7.1 Connecting $p(x_i\mid y)$ and $p(y\mid x_i)$

The definition above uses the label distribution after observing a feature. The algorithm on slide 16 instead estimates how the feature is distributed within each class. We can recover the required label probabilities from these class-conditional distributions using Bayes' rule.

For a continuous feature $X_i$ and a discrete label $Y$, we use the notation:

$$
\pi_k=P(Y=k),\qquad
f_{i\mid k}(x)=f_{X_i\mid Y}(x\mid k),\qquad
f_i(x)=f_{X_i}(x)
$$

Here, $\pi_k$ is a class probability, while $f_{i\mid k}$ and $f_i$ are the class-conditional and marginal feature densities. The product $\pi_k f_{i\mid k}(x)$ represents the joint distribution of feature value $x$ and class $k$. Summing these products over the classes gives the marginal density:

$$
f_i(x)=\sum_{k=1}^{K}\pi_k f_{i\mid k}(x).
$$

Dividing the class-specific joint term by this marginal density gives the conditional label probability through Bayes' rule:

$$
P(Y=k\mid X_i=x)=\frac{\pi_k f_{i\mid k}(x)}{f_i(x)}
$$

### 7.2 Expanding the Difference between the Two Entropies

We first rewrite the label entropy so that both entropies use the same weighting term. Each class-conditional density integrates to 1, so inserting that integral leaves the label entropy unchanged:

$$
\begin{aligned}
H(Y)
&=-\sum_{k=1}^{K}\pi_k\log_2\pi_k\\
&=-\sum_{k=1}^{K}\int
\pi_k f_{i\mid k}(x)\log_2\pi_k\,dx
\end{aligned}
$$

For the conditional entropy, we substitute the label probabilities from Bayes' rule. Using
$f_i(x)P(Y=k\mid X_i=x)=\pi_k f_{i\mid k}(x)$, we have:

$$
\begin{aligned}
H(Y\mid X_i)
&=-\sum_{k=1}^{K}\int f_i(x)P(Y=k\mid X_i=x)
\log_2P(Y=k\mid X_i=x)\,dx\\
&=-\sum_{k=1}^{K}\int\pi_k f_{i\mid k}(x)
\log_2\frac{\pi_k f_{i\mid k}(x)}{f_i(x)}\,dx
\end{aligned}
$$

Both expressions now use the same weights, so subtracting them leaves a difference of logarithms. Combining those logarithms, we have:

$$
\begin{aligned}
I(X_i;Y)
&=H(Y)-H(Y\mid X_i)\\
&=\sum_{k=1}^{K}\int\pi_k f_{i\mid k}(x)
\left[\log_2\frac{\pi_k f_{i\mid k}(x)}{f_i(x)}
-\log_2\pi_k\right]dx\\
&=\sum_{k=1}^{K}\int\pi_k f_{i\mid k}(x)
\log_2\frac{f_{i\mid k}(x)}{f_i(x)}\,dx
\end{aligned}
$$

The final line is the feature score on slides 16–17. The derivation applies to classes with $\pi_k>0$, and the density ratio is evaluated where the relevant densities are positive. Terms with zero weight contribute zero.

The ratio compares how common a feature value is within a class with how common it is overall. If every class satisfies $f_{i\mid k}(x)=f_i(x)$, the numerator and denominator agree, so the logarithmic terms vanish and the score is zero. Individual logarithmic terms can otherwise be negative, even though the complete weighted score is nonnegative.

### 7.3 The Attendance Example

Slide 17 illustrates these distributions using **Attendance** as the feature and **Statistics Score** as the target. To estimate the feature score, we distinguish the three quantities by the observations used to calculate them:

| Quantity | Observations Used | Description |
|---|---|---|
| $\pi_6$ | Labels from the entire dataset | Proportion of samples in the score-6 class |
| $f_i(x)$ | Attendance values from all samples | Overall attendance distribution |
| $f_{i\mid6}(x)$ | Attendance values from the score-6 class only | Attendance distribution within this class |

In particular, we estimate $f_{i\mid6}(x)$ by first selecting students with a score of 6 and then estimating the attendance distribution within that group.

The classification setup treats a score of 6 as one discrete class. Although the illustrative table includes decimal scores, the slides do not specify how those scores are discretized or grouped. The figures illustrate the estimation process but do not provide enough information to compute a numerical mutual information value.

## 8. Applying Feature Selection (Slides 16–17)

The feature score depends on probabilities and densities that are usually unknown. We estimate them from labeled training data and then use the estimates to score each feature separately.

**Step 1. Estimate the class probabilities.** We denote the number of training samples in class $k$ by $n_k$ and estimate the class probability from its relative frequency:

$$
n_k=\sum_{r=1}^{n}\mathbf{1}\lbrace y^{(r)}=k\rbrace,
\qquad
\widehat\pi_k=\frac{n_k}{n}
$$

Because these probabilities depend only on the labels, the same estimates are used for every feature and need to be computed only once.

**Step 2. Estimate the feature distributions.** For feature $i$, we use all observations to estimate $f_i$, then restrict the data to observations satisfying $y^{(r)}=k$ to estimate $f_{i\mid k}$. The lecture allows either parametric or nonparametric density estimation [2]. For discrete features, we can use relative frequencies; for continuous features, we can use histograms or kernel density estimation.

For example, suppose a histogram bin $B_j$ has width $\Delta_j$ and contains $n_j$ samples, of which class $k$ contributes $n_{jk}$. If all classes use the same bins, the density estimates for $x\in B_j$ are

$$
\widehat f_i(x)=\frac{n_j}{n\Delta_j},
\qquad
\widehat f_{i\mid k}(x)=\frac{n_{jk}}{n_k\Delta_j}
$$

In each estimate, dividing by the sample count converts the bin count to a proportion, and dividing by the bin width converts that proportion to a density. The class-conditional estimate uses $n_k$ rather than $n$ because it describes only the samples in class $k$. This normalization is also needed for the attendance histogram on slide 17, whose vertical axis shows student counts rather than densities.

**Step 3. Compute the information score.** With the distributions estimated, we substitute them into $I(X_i;Y)$. For discrete features, we sum over the possible values; for continuous features, we evaluate the integral or use a numerical approximation.

**Step 4. Rank and select.** Repeating the calculation for $i=1,\ldots,d$ gives one score per feature. We then rank the features and retain those with high scores, such as the top $m$. The slides do not specify a universal value of $m$.

The result is a selected set of inputs. These inputs must still be used to train a classifier before we can predict labels for new samples.

## 9. A Small Reproducible Example

The following example uses eight observations to reproduce the joint probability table from the lecture. The counts for $(X,Y)=(0,0),(0,1),(1,0),(1,1)$ are $2,2,4,0$, respectively, so dividing each count by eight recovers the probabilities used in Section 5.

For comparison, feature $Z$ divides the same observations into two groups, each with three labels of 0 and one label of 1. Since both groups have the same label proportions, observing either value of $Z$ leaves the empirical label distribution unchanged.

The implementation uses only the Python standard library and follows the conditional-entropy calculation above. It first groups observations by the conditioning variable, computes the label entropy within each group, and then averages those entropies with weights proportional to group size.

```python
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from math import log2


def entropy(values: Sequence[Hashable]) -> float:
    """Compute entropy from empirical frequencies."""
    n = len(values)
    if n == 0:
        raise ValueError("At least one observation is required.")
    return -sum(
        (count / n) * log2(count / n)
        for count in Counter(values).values()
    )


def conditional_entropy(
    target: Sequence[Hashable], given: Sequence[Hashable]
) -> float:
    """Compute H(target | given) by grouping and averaging."""
    n = len(target)
    if n == 0 or len(given) != n:
        raise ValueError("Inputs must have the same nonzero length.")
    groups = defaultdict(list)
    for target_value, given_value in zip(target, given):
        groups[given_value].append(target_value)
    return sum(
        (len(group) / n) * entropy(group)
        for group in groups.values()
    )


def mutual_information(
    feature: Sequence[Hashable], labels: Sequence[Hashable]
) -> float:
    """Compute I(feature; labels) = H(labels) - H(labels | feature)."""
    return entropy(labels) - conditional_entropy(labels, feature)


x = [0, 0, 1, 1, 0, 0, 1, 1]
y = [0, 1, 0, 0, 0, 1, 0, 0]
z = [0, 0, 0, 0, 1, 1, 1, 1]

print(f"H(X) = {entropy(x):.4f}")
print(f"H(Y) = {entropy(y):.4f}")
print(f"H(X, Y) = {entropy(list(zip(x, y))):.4f}")
print(f"H(Y | X) = {conditional_entropy(y, x):.4f}")
print(f"H(X | Y) = {conditional_entropy(x, y):.4f}")
print(f"I(X; Y) = {mutual_information(x, y):.4f}")
print(f"I(Z; Y) = {mutual_information(z, y):.4f}")
```

For these observations, the code produces the following entropy and mutual information values, in bits:

```text
H(X) = 1.0000
H(Y) = 0.8113
H(X, Y) = 1.5000
H(Y | X) = 0.5000
H(X | Y) = 0.6887
I(X; Y) = 0.3113
I(Z; Y) = 0.0000
```

Based on these scores, selecting one feature would retain $X$ rather than $Z$. Although $X$ reduces label uncertainty, it does not remove it completely. By contrast, $Z$ leaves the label distribution unchanged and is independent of $Y$ in this empirical dataset.

Because this implementation counts exact observed values, it estimates discrete distributions. Treating every distinct floating-point measurement as a separate category would not replace the density-estimation step needed for continuous features.


## Supplement A: The Scope of Differential Entropy

The main discussion assumes a discrete label $Y$, but slide 12 also considers a continuous target. If we denote that target by $Z$ and assume that $X$ is also continuous, the conditional differential entropy is

$$
h(Z\mid X)
=-\int f_X(x)\left[\int f_{Z\mid X}(z\mid x)
\log_2f_{Z\mid X}(z\mid x)\,dz\right]dx.
$$

The inner integral computes the target's differential entropy at a fixed $x$, and the outer integral averages that quantity over $x$ using its density.

Unlike discrete entropy, differential entropy depends on the measurement scale and can be negative. For example, if $Z$ is uniform on $[0,a]$, then

$$
h(Z)=-\int_0^a\frac1a\log_2\frac1a\,dz=\log_2 a,
$$

which is negative when $0<a<1$. Differential entropy therefore cannot be interpreted directly as the number of bits needed to encode an exact real-valued observation.

## Supplement B: Further Interpretation and Practical Considerations

We can also interpret mutual information by expressing the entropy difference in terms of the joint and marginal distributions. For discrete variables, expanding the logarithms gives [3]

$$
\begin{aligned}
I(X;Y)
&=\sum_{x,y}p(x,y)\left[\log_2p(y\mid x)-\log_2p_Y(y)\right]\\
&=\sum_{x,y}p(x,y)\log_2\frac{p(x,y)}{p_X(x)p_Y(y)}.
\end{aligned}
$$

This expression is the Kullback–Leibler divergence between the joint distribution and the product of its marginals. It is nonnegative and equals zero if and only if the variables are independent. The dependence measured by mutual information can be nonlinear, but it does not establish a causal relationship.

Although this score measures how informative each feature is on its own, ranking individual features does not guarantee the best subset. Several high-scoring features may contain overlapping information, while other information may appear only when features are combined. When using the selected features to evaluate a predictive model, we must fit the selection procedure on training data so that test labels do not influence which features are retained.

For continuous features, we can construct the estimated marginal density directly from the class-conditional estimates, weighting each by its estimated class probability:

$$
\widehat f_i(x)=\sum_k\widehat\pi_k\widehat f_{i\mid k}(x).
$$

This is the relationship obtained by the count estimates in Section 8 when all classes use the same histogram bins. Adding the class-specific counts within a bin recovers its total count and gives the mixture identity above.

## References
1. Kai Wang. *CSE/ISyE 6740 Lecture 06: Feature Selection and Decision Tree*, September 14, 2026, slides 1–18. The coin and joint-distribution examples above are adapted from slides 11 and 13–14. [Course slides](https://github.com/GT-KOALA/Gatech-CSE6740/blob/main/assets/semesters/fall-2026/lectures/06/lecture-06-feature-selection-trees.pdf).
2. Tsachy Weissman, with lecture notes by Hanchel Cheng, Kyle Chiang, and Ashwin Siripurapu. *EE376A Information Theory, Lecture 9*, February 5, 2015, Section 3.1. Supplementary reference for the distinction between discrete and differential entropy. [Stanford lecture notes](https://web.stanford.edu/class/ee376a/files/scribes/lecture9.pdf).
