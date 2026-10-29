---
semester: fall-2026
lecture_number: "06"
slot: optional-2
slot_order: 4
role: optional_topic
role_label: Optional topic
tab_title: Extension II
assignee: joecalbarboza
issue: 16
---
## Random forests: intuition, advantages, and usage

### 1. From one decision tree to a forest

Lecture 06 builds a classification tree by repeatedly choosing a feature and threshold that reduce label uncertainty. Its final slide identifies ensembles as one way to address overfitting. A **random forest** develops that idea: train many randomized decision trees, then combine their predictions. This note explains the additional machinery rather than repeating the lecture's tree-building algorithm.

A deep tree can fit complex interactions, but its splits can change substantially when the training data change slightly. This is **high variance**. Averaging helps when different trees make different errors. If every tree learned the same splits, adding trees would contribute little.

A forest therefore introduces diversity in two places:

1. **Bootstrap samples:** each tree receives a different resampled training set.
2. **Random feature subsets:** each node searches for its split among a randomly selected subset of features.

The objective is to retain useful individual trees while reducing how similarly they behave. Breiman's original paper formalizes random forests using randomized tree predictors [1].

### 2. How a tree in the forest is trained

Let the training set be $D=\{(x_i,y_i)\}_{i=1}^{n}$, with $x_i\in\mathbb{R}^{d}$ and $y_i\in\{1,\ldots,C\}$. Let $B$ be the number of trees and $m$ the number of candidate features at a node.

#### Bootstrap sampling

For each tree, draw $n$ training indices independently and uniformly **with replacement** from $\{1,\ldots,n\}$. Some observations appear multiple times; others are absent. For example, five draws from indices $\{1,2,3,4,5\}$ might give $(2,2,5,1,5)$.

For one observation, the probability of being absent from a tree's sample is

$$
P(\text{not sampled})=\left(1-\frac{1}{n}\right)^n\longrightarrow e^{-1}\approx0.368.
$$

Thus a bootstrap sample of size $n$ contains about $0.632n$ distinct observations **in expectation for large $n$**, not all $n$ distinct observations. The remaining observations are that tree's **out-of-bag (OOB)** observations.

#### Random feature selection at each node

At a node, sample a candidate subset $M\subseteq\{1,\ldots,d\}$ of size $m$. Search for the best feature-threshold split within $M$:

$$
(j^{\ast},t^{\ast})=
\underset{j\in M,t}{\mathrm{arg\,max}}
\left[
H(S)-\frac{|S_L|}{|S|}H(S_L)
-\frac{|S_R|}{|S|}H(S_R)
\right].
$$

where $S$ is the node's bootstrap data, $S_L=\{x\in S:x_j\le t\}$, and $S_R=\{x\in S:x_j>t\}$. For classification,

$$
H(S)=-\sum_{c=1}^{C}p_c\log_2p_c.
$$

Here $p_c$ is the class fraction, counting bootstrap multiplicities. This is the lecture's information-gain split, with a restricted feature search. Gini impurity, $1-\sum_cp_c^2$, is another common choice. Our example explicitly uses entropy [2].

**The feature subset is sampled again at each node**, not fixed once for the whole forest. A feature absent from one node's candidate subset can appear later. In scikit-learn, the split search may inspect additional features if needed to find a valid partition, so `max_features` is a nominal search limit rather than an absolute limit in every situation [2].

Trees can often be grown deeply, but constraints such as maximum depth and minimum leaf size remain useful.

#### Training pseudocode

```text
For b = 1,...,B:
    Draw n training observations with replacement.
    Grow tree b on that bootstrap sample:
        At each nonterminal node:
            Sample a new subset of candidate features.
            Search candidate thresholds on those features.
            Choose the best valid impurity-reducing split.
        Stop according to purity and tree-size constraints.
Return all B trees.
```

Sampling observations and averaging trees is **bagging**. A standard random forest adds feature randomization at each split. It does not choose all thresholds arbitrarily: it searches for a good threshold among the candidate features.

### 3. Combining predictions

#### Classification

A hard-voting description is

$$
\widehat y(x)=\underset{c}{\mathrm{argmax}}\sum_{b=1}^{B}\mathbf{1}\{T_b(x)=c\},
$$

where $T_b(x)$ is tree $b$'s predicted class. For example, predictions $(A,B,A,A,B)$ give class $A$.

**Implementation detail:** scikit-learn's `RandomForestClassifier` instead averages each tree's class-probability vector, then selects its largest entry [2]:

$$
\widehat p(c\mid x)=\frac{1}{B}\sum_{b=1}^{B}\widehat p_b(c\mid x),\qquad
\widehat y(x)=\underset{c}{\mathrm{argmax}}\widehat p(c\mid x).
$$

A tree's probability is the class fraction in the leaf reached by $x$. Probability averaging and hard voting need not agree when leaves are impure. Forest probabilities are estimates, not automatically calibrated confidence statements.

#### Regression

For a continuous response, average the numerical predictions:

$$
\widehat f(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x).
$$

For example, predictions $(10,12,11)$ produce $11$. The trees use a regression split criterion, such as squared-error reduction, rather than class entropy.

### 4. Why averaging and randomness help

The following calculation is an illustrative variance model, not a general classification-error formula. Fix an input $x$. Suppose predictions $Z_b=T_b(x)$ each have variance $\sigma_T^2$ and every distinct pair has correlation $\rho$. Then

$$
\begin{aligned}
\mathrm{Var}\left(\frac{1}{B}\sum_{b=1}^{B}Z_b\right)
&=\frac{1}{B^2}\left[B\sigma_T^2+B(B-1)\rho\sigma_T^2\right]\\
&=\sigma_T^2\left[\rho+\frac{1-\rho}{B}\right].
\end{aligned}
$$

The randomness here can represent changing training samples and tree randomization. The equal-variance/equal-correlation assumptions simplify the explanation.

- Increasing $B$ reduces the $(1-\rho)/B$ term.
- If $\rho=1$, averaging gives no variance reduction.
- If $\rho=0$, variance is $\sigma_T^2/B$.
- With fixed positive $\rho$, the variance approaches a nonzero floor.

For example, take $\sigma_T^2=1$ and $B=100$. Correlation $0.8$ gives variance $0.802$, while correlation $0.1$ gives $0.109$. **Diversity matters, not just tree count.** Bootstrap sampling and restricted feature searches aim to reduce correlation. Restricting features too aggressively can also weaken each tree.

For squared-error regression,

$$
\mathbb{E}\left[(Y-\widehat f(x))^2\mid X=x\right]
=\mathrm{Var}(Y\mid X=x)+\mathrm{Bias}(\widehat f(x))^2
+\mathrm{Var}(\widehat f(x)).
$$

This decomposition assumes prediction is evaluated on a fresh response independent of training, conditional on $x$. A forest primarily targets estimator variance; it cannot remove irreducible noise or repair every source of bias. For classification, averaging probability estimates offers a related intuition, but the variance identity is not itself a formula for misclassification error. Forests can still overfit or fail under distribution shift.

### 5. Advantages, tradeoffs, and important settings

| Aspect | One decision tree | Random forest |
|---|---|---|
| Explanation | A compact tree can be inspected as rules | Many trees make the whole model harder to inspect |
| Stability | Deep trees can be sensitive to training changes | Aggregation often improves stability |
| Interactions | Learns conditional, nonlinear rules | Combines many such rules |
| Computation | Usually faster and smaller | More trees require more time and memory |
| Preprocessing | Usually no feature scaling needed | Usually no feature scaling needed |
| Evaluation | Validation or cross-validation | Also supports OOB evaluation with bootstrapping |

Forests are useful baselines for tabular data. Standard axis-aligned regression forests predict averages of leaf responses and generally extrapolate poorly beyond the observed response range. Numerical category codes should not be mistaken for meaningful ordered measurements; encode categorical inputs appropriately for the implementation.

| scikit-learn setting | Meaning and practical effect |
|---|---|
| `n_estimators` | Number of trees. More trees reduce finite-ensemble randomness, with diminishing returns and increasing cost. |
| `max_features` | Candidate features per split. Smaller subsets may reduce correlation but weaken trees; `"sqrt"` is a common classification starting point. |
| `max_depth` | Caps path length. Shallower trees can reduce complexity but miss interactions. |
| `min_samples_leaf` | Minimum observations required in a leaf. Larger values discourage tiny, noisy leaves. |
| `bootstrap` | Enables resampling with replacement. Required for the OOB procedure below. |
| `max_samples` | Controls bootstrap sample size when bootstrapping is enabled. The derivation above assumes $n$ draws. |
| `criterion` | Split impurity, such as `"entropy"` or `"gini"` for classification. |
| `class_weight` | Changes class contributions during fitting; may help with imbalance but does not replace metric selection. |
| `random_state` | Reproducible pseudorandom choices. |
| `n_jobs` | Parallel execution; changes resource use rather than the statistical objective. |

These controls are documented in the estimator reference [2]. Increasing the number of trees does not progressively deepen each tree; it is different from increasing `max_depth`.

**Computational perspective.** If one fitted tree has $L$ nodes, storing $B$ comparable trees costs $O(BL)$. A prediction traverses each tree, giving roughly $O(Bh)$ node decisions for depth $h$. Training depends on split search, sample sizes at nodes, and depth. With $m$ candidate features and a presorted linear-scan assumption, balanced trees suggest $O(Bmn\log n)$ split work. Re-sorting values independently at every node can add another logarithmic factor; highly unbalanced trees can be worse. This is an explanatory bound, not a universal runtime guarantee for scikit-learn.

### 6. Out-of-bag evaluation and feature importance

#### Out-of-bag predictions

For observation $i$, let $\mathcal B_i$ contain only trees whose bootstrap samples did not include it. Average those trees' probabilities:

$$
\widehat p_{\mathrm{OOB}}(c\mid x_i)
=\frac{1}{|\mathcal B_i|}\sum_{b\in\mathcal B_i}\widehat p_b(c\mid x_i).
$$

Compare the resulting OOB predictions with the training labels. Every observation is evaluated using a subset of trees that did not train on it. With too few trees, some observations may have no OOB trees. OOB estimates are useful internal diagnostics; repeated tuning against them can still overfit the evaluation. Keep a final test set, and use grouped or temporal evaluation when observations are not iid [2, 3].

#### Two notions of feature importance

**Impurity importance** (`feature_importances_`) summarizes each feature's weighted impurity reductions across the trees. It is fast, but continuous or high-cardinality features can be favored because they offer more candidate splits.

**Permutation importance** measures how much a chosen evaluation score decreases when a feature's values are shuffled, ideally on held-out validation data. It describes reliance by this fitted model, not causation or an intrinsic property of the feature. Correlated features can substitute for each other, making individual permutation importances appear small [4]. If importance is used for feature selection, perform that selection inside training/validation folds rather than using the final test set.

### 7. Reproducible example: classify wine cultivars

The built-in Wine dataset contains 178 observations, 13 numerical features, and three classes [5]. It requires no external download. We compare one tree with a forest using the same train/test split and the same entropy criterion.

**Evaluation plan:** reserve a stratified 30% test set; run five-fold stratified cross-validation only on the remaining training set; then fit each fixed configuration on all training data and evaluate the held-out test set. Stratification approximately preserves class proportions. All settings below are fixed in advance: the test scores are not used to choose them. No feature scaling or preprocessing is needed here.

The code was run with scikit-learn 1.7.2. Fixed seeds support reproducibility, but exact numbers can change with library versions. To reproduce the reported environment, use Python 3.12, `scikit-learn==1.7.2`, and `numpy==2.5.4`.

```python
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Single tree": DecisionTreeClassifier(
        criterion="entropy", random_state=42,
    ),
    "Random forest": RandomForestClassifier(
        n_estimators=300,
        criterion="entropy",
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=1,
    ),
}
print(f"scikit-learn {sklearn.__version__}; NumPy {np.__version__}")
print(f"Train: {len(y_train)}; Test: {len(y_test)}")
for name, model in models.items():
    scores = cross_val_score(
        clone(model), X_train, y_train, cv=cv,
        scoring="accuracy", n_jobs=1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(
        f"{name}: CV={scores.mean():.3f} +/- {scores.std():.3f}; "
        f"train={model.score(X_train, y_train):.3f}; "
        f"test={accuracy_score(y_test, pred):.3f}; "
        f"macro-F1={f1_score(y_test, pred, average='macro'):.3f}"
    )
print(f"Forest OOB accuracy: {models['Random forest'].oob_score_:.3f}")
print("First three predictions:", models["Random forest"].predict(X_test[:3]))
print(
    "First probability vector:",
    np.round(models["Random forest"].predict_proba(X_test[:1])[0], 3),
)
```

`n_estimators=300` supplies many randomized trees. With 13 features, `max_features="sqrt"` nominally considers three at a split. `min_samples_leaf=2` avoids single-observation leaves; the forest therefore differs from the baseline in both aggregation and leaf-size control. This is a practical comparison, **not an ablation isolating one cause of improvement**. `n_jobs=1` keeps the example's resource use modest; trees can also be trained in parallel.

Training accuracy measures fit to observed data; cross-validation measures performance across training partitions. Test accuracy is the final held-out classification fraction. Macro-F1 computes F1 separately for each class and averages them equally. The standard deviation across five CV folds describes fold-to-fold variability; it is not a confidence interval [3].

#### Results from the verified run

The split contained 124 training and 54 test observations.

| Model | Training accuracy | Training-only CV accuracy (mean ± SD) | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Single tree | 1.000 | 0.854 ± 0.035 | 0.907 | 0.910 |
| Random forest | 1.000 | 0.976 ± 0.032 | 1.000 | 1.000 |

The forest's OOB accuracy was 0.976. The first three predicted labels were `[0, 1, 0]`; the first probability vector, rounded to three decimals, was `[0.949, 0.036, 0.015]`.

Both models fit the training observations perfectly, while their held-out behavior differed. The forest performed better in this run, and its training-only CV scores were also higher. Nevertheless, this is one small dataset and one test split: 100% test accuracy means all 54 held-out observations were correct, not that future error is zero. The forest and baseline also differ in minimum leaf size, so these results cannot attribute the difference solely to bootstrapping or feature randomization. The similar OOB and CV scores are reassuring here, but these estimates use different fitting/evaluation procedures and need not agree in general.

`predict` returns class labels; `predict_proba` returns one column per class in the order of `classes_`. For other applications, choose metrics reflecting the error costs, such as recall or balanced accuracy for an imbalanced task. If tuning parameters, perform the search within the training set (for example, with `GridSearchCV`) and evaluate the selected procedure once on the final test set.

### 8. Takeaways

- Bootstrap resampling varies the observations used by each tree; feature subsampling varies the candidate splits.
- Aggregation is most useful when individual trees are informative without making identical errors.
- More trees improve ensemble stability but do not remove all bias, overfitting, or distribution shift.
- OOB scores and feature importances are useful diagnostics with distinct limitations.
- A sound comparison needs a baseline, a reproducible split, appropriate metrics, and protection against test-set leakage.

### References

1. Leo Breiman (2001). [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf). *Machine Learning*, 45, 5-32. Original random-forest formulation and discussion of tree strength and dependence.
2. scikit-learn 1.7. [RandomForestClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Algorithm settings, probability aggregation, OOB attributes, and implementation details.
3. scikit-learn 1.7. [Cross-validation: evaluating estimator performance](https://scikit-learn.org/1.7/modules/cross_validation.html). Train/test separation and validation strategies.
4. scikit-learn. [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html). Impurity-importance bias, held-out permutation evaluation, and correlated-feature limitations.
5. scikit-learn 1.7. [load_wine](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_wine.html). Dataset dimensions, class labels, and original data attribution.
6. Kai Wang (2026). *Lecture 06: Feature Selection and Decision Tree*, CSE/ISyE 6740, Georgia Institute of Technology. Connection: greedy threshold splits, entropy/information gain, and the final slide's introduction of ensembles as an extension.

### Preparation disclosure

This draft was prepared with assistance from OpenAI Codex for organization, explanations, equations, and the example code. The example was executed and its reported results checked during preparation.
