---
semester: fall-2026
lecture_number: "06"
slot: course-2
slot_order: 2
role: course_material
role_label: Course materials
tab_title: Core explanation II
assignee: unassigned
issue: null
---
# 1. Where This Picks Up
Course Materials I (slides 1–18) covered **feature selection**: given a labeled dataset, how do we measure whether a single feature $X_i$ is informative about the label $Y$, using entropy $H(Y)$, conditional entropy $H(Y \mid X_i)$, and mutual information $I(X_i, Y) = H(Y) - H(Y \mid X_i)$? Slide 18 is a one-line transition slide with no new content. Starting at slide 19, the lecture reuses that exact same information-gain idea for a new purpose: building a **decision tree**, a full predictive model rather than just a ranking of features.

# 2. The Classification Setup
Formally, we are given a dataset
$$ 
D = \{(x^1, y^1), (x^2, y^2), \ldots, (x^n, y^n)\}, \qquad x \in \mathbb{R}^d, \quad y \in \{1, 2, \ldots, K\} 
$$
Each $x^i$ is a $d$-dimensional feature vector and $y^i$ is one of $K$ discrete class labels. The learning problem is: given a **new** feature vector $x$ that was not in the training set, infer its label $y$. Formally, we want to learn a function mapping $f : X \to Y$ from the training data, and then apply $f$ to new inputs.

This is a very generic setup: many different model families can be used to define $f$ (nearest neighbors, support vector machines, neural networks, decision trees, and so on), and they can disagree quite a bit on the *shape* of the decision boundary they draw between classes, even when they are trained on the exact same points. The lecture illustrates this with a scikit-learn comparison figure that runs ten classifiers on the same three toy 2D datasets: every method separates the red and blue points reasonably well, but a decision tree's boundary looks visibly different from, say, a nearest-neighbor or a Gaussian-process boundary: it is made of axis-aligned rectangular blocks rather than smooth curves (see the [scikit-learn classifier-comparison gallery](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html), cited in the lecture as the source of that figure). That "blocky, axis-aligned" shape is a direct consequence of *how* a decision tree makes its predictions, which is the subject of the rest of this note.

# 3. Decision Trees as Hypotheses
A decision tree represents $f$ as a tree of yes/no questions about the features:

- Every **internal node** tests one attribute/feature $x_i$.
- Every **branch** out of a node corresponds to one possible outcome of that test (for a real-valued feature, this will mean a branch for "above a threshold" and a branch for "at or below it"; see Section 6).
- Every **leaf** node stores a class label $y$.

To predict the label of a new input $x$, we **traverse the tree from the root to a leaf**: at each internal node we look at the value of the feature it tests, follow the matching branch, and repeat, until we land on a leaf, whose label is the tree's output.

A tiny example (not from the slides, just for illustration) makes the traversal concrete: suppose a 2-level tree first asks "is $x_i \le v$?" and then, on the "yes" branch, asks "is $x_j \le u$?":

```
                 [ x_i <= v ? ]
                 /            \
              yes              no
               |                |
      [ x_j <= u ? ]        (leaf: red)
       /          \
     yes            no
      |              |
  (leaf: blue)   (leaf: red)
```

A new point with $x_i \le v$ and $x_j > u$ would walk left, then right, and be classified **red**. This is exactly the "blocky" boundary from Section 2: every leaf owns a rectangular region of feature space carved out by the thresholds on the path leading to it.

# 4. Why Not Just Find the Smallest Tree?
Many different trees can be built that correctly classify the same training set, including, in the extreme, a tree that grows one branch per training point and simply memorizes labels. Between all the trees that fit the data, we would prefer the *smallest* one: a compact tree is easier to interpret, and (by the same intuition as picking the fewest useful features in Course Materials I) is less likely to be fitting noise.

Unfortunately, finding the **provably smallest** decision tree consistent with a dataset is an NP-complete problem (Hyafil & Rivest, 1976). Since we cannot afford to search over all possible tree shapes, the lecture instead uses a **greedy heuristic**:

1. Start from an empty tree.
2. Split on whichever remaining attribute looks like the single best choice *right now* (by information gain).
3. Recurse: repeat the same procedure independently inside each branch created by that split.

This does not guarantee the globally smallest tree: it is a "take the best next step and never look back" strategy, but it is cheap to compute and works well in practice, which is exactly the trade-off greedy algorithms usually make against exact/optimal ones.

# 5. Growing the Tree by Recursion
Concretely, at any node with the (sub)dataset $D$ it is responsible for, the algorithm is:

1. **Initialization:** start from an empty tree holding dataset $D$.
2. For every remaining attribute (feature) $X_i$, compute its information gain against the current $D$, reusing the exact same quantity from Course Materials I:
$$
I(X_i, Y) = H(Y) - H(Y \mid X_i)
$$
3. **Pick the best attribute:**
$$
i := \arg\max_i \, I(X_i, Y) = \arg\max_i \, \big[H(Y) - H(Y \mid X_i)\big]
$$
4. **Split** the tree (and the dataset) into one branch per value of $X_i$: if $X_i$ takes values $v_1, v_2, \ldots$ in $D$, create sub-datasets $D_{\{X_i = v_1\}}, D_{\{X_i = v_2\}}, \ldots$, one per branch.
5. **Recurse** independently on each sub-dataset, exactly as if it were a brand-new, smaller classification problem.

The lecture's running example is predicting Atlanta apartment rental price from a dataset $D$ of listings. Suppose "number of bedrooms" turns out to have the highest information gain at the root, so the tree first splits three ways on $X_i \in \{0, 1, 2\}$. Each branch now owns a sub-dataset (e.g. $D_{\{X_i = 2\}}$ contains only the 2-bedroom listings), and recursion treats that sub-dataset as its own fresh problem. If "number of bathrooms" $X_j$ has the best information gain *within that sub-dataset*, the $X_i = 2$ branch splits again on $X_j \in \{0, 1, 2\}$, and so on down the tree:

```
                        Root
                          |
             X_i: number of bedrooms?
        /              |               \
   X_i = 0          X_i = 1           X_i = 2
                                          |
                                X_j: number of bathrooms?
                                /         |         \
                           X_j = 0    X_j = 1     X_j = 2
```

Notice that the *same* information-gain rule from Course Materials I is being applied over and over, just on shrinking, increasingly specific subsets of the data: that reuse is exactly what makes this a recursive algorithm in the computer-science sense, not merely a repeated one.

# 6. Real-Valued Features Need a Different Kind of Split
The scheme above works cleanly for a feature like "number of bedrooms" that only takes a handful of distinct values. It breaks down for a **real-valued** feature such as room size or distance to school: giving every distinct numeric value its own branch would mean an unbounded number of branches (in principle, infinitely many). A hypothesis with that many branches is not really learning anything general: with enough branches, every leaf ends up containing at most one training example, which is memorization, not generalization, and it will overfit any dataset.

The fix is to give up on branching per exact value and instead use a **binary threshold split**: pick a threshold $t$ and split attribute $X_i$ into just two branches, $X_i < t$ and $X_i \ge t$. Because this only "costs" one comparison rather than enumerating every value, we can even split on the **same** feature more than once along a single root-to-leaf path, at different thresholds, e.g. first splitting room size at 1000 ft², and later, further down that branch, splitting the remaining listings again at 600 ft²:

```
                     Root
                       |
             X_i: number of bedrooms?
                       |
                   X_i = 2
                       |
           X_j: room size > 1000 ft²?
              /                  \
      (< 1000 ft²)           (>= 1000 ft², leaf)
             |
   X_k: room size > 600 ft²?
       /              \
 (< 600 ft², leaf)  (>= 600 ft², leaf)
```

# 7. Choosing the Threshold
Picking a good threshold $t$ sounds like it requires searching over the real line, but it does not: **only a finite number of candidate thresholds are ever worth checking**. Sort the training examples by the value of $X_i$; as $t$ sweeps from one sorted value to the next, the resulting left/right split (and therefore its information gain) only changes at points where $t$ crosses over a training example. Furthermore, if two *adjacent* sorted examples have the **same** label, there is nothing to gain by placing a threshold between them (both examples would already end up on the same side no matter which of the two neighboring gaps you pick, so the split's quality is identical). So the only candidate thresholds worth testing are the midpoints between adjacent, sorted examples whose labels **differ**.

With candidate thresholds in hand, the information gain formula from Section 5 simply gets an extra condition attached to it:
$$
I(X_i > t, Y) = H(Y) - H(Y \mid X_i > t)
$$
and we search jointly over both the feature *and* the threshold:
$$
i, t := \arg\max_{i,\,t} \; I(X_i > t, Y) = \arg\max_{i,\,t} \; \big[H(Y) - H(Y \mid X_i > t)\big]
$$

**Worked example.** To make this concrete, here is a small toy dataset (not from the slides) of 6 students, "hours studied" versus pass/fail on a quiz:

| Hours studied | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Result | Fail | Fail | Fail | Pass | Fail | Pass |

Overall there are 4 fails and 2 passes, so (using the same entropy formula as Course Materials I):
$$
H(Y) = -\tfrac{4}{6}\log_2\tfrac{4}{6} - \tfrac{2}{6}\log_2\tfrac{2}{6} \approx 0.918 \text{ bits}
$$
Reading down the sorted list, the label changes between hours 3 and 4 (Fail → Pass), between 4 and 5 (Pass → Fail), and between 5 and 6 (Fail → Pass): three label changes, so there are exactly three candidate thresholds: $t \in \{3.5,\ 4.5,\ 5.5\}$. The two gaps between same-label neighbors (1–2 and 2–3, both Fail) are skipped. Computing the gain at each candidate:

| Threshold $t$ | Left group ($<t$) | Right group ($\ge t$) | Weighted $H(Y \mid X>t)$ | Gain $I(X>t, Y)$ |
|---|---|---|---|---|
| 3.5 | {1,2,3} all Fail, $H=0$ | {4,5,6} = 2 Pass, 1 Fail, $H \approx 0.918$ | $\tfrac{3}{6}(0) + \tfrac{3}{6}(0.918) \approx 0.459$ | $\approx 0.459$ |
| 4.5 | {1,2,3,4} = 3 Fail, 1 Pass, $H \approx 0.811$ | {5,6} = 1 Fail, 1 Pass, $H = 1$ | $\tfrac{4}{6}(0.811) + \tfrac{2}{6}(1) \approx 0.874$ | $\approx 0.044$ |
| 5.5 | {1,2,3,4,5} = 4 Fail, 1 Pass, $H \approx 0.722$ | {6} pure Pass, $H=0$ | $\tfrac{5}{6}(0.722) + \tfrac{1}{6}(0) \approx 0.602$ | $\approx 0.316$ |

So $t = 3.5$ is chosen: it gives the largest reduction in entropy ($\approx 0.459$ bits), even though it does not perfectly separate the classes: the "5 hours, Fail" student ends up on the "Pass-leaning" side. That imperfection is exactly why the algorithm recurses: the right branch $\{4,5,6\}$ is not yet pure, so it becomes its own sub-problem and the same search (over remaining thresholds, or other features if there were any) runs again inside it, exactly as described in Section 5.

# 8. Putting It Together: the Full Algorithm
Combining Sections 5 through 7, the complete recursive learning procedure is:

1. **Initialization:** start from an empty tree with dataset $D$.
2. For every attribute $X_i$, testing every candidate threshold $t$ if it is real-valued, compute the information gain $I(X_i > t, Y) = H(Y) - H(Y \mid X_i > t)$ (or the plain $I(X_i, Y)$ from Section 5 if $X_i$ is categorical).
3. **Pick the best (attribute, threshold) pair:** $i, t := \arg\max_{i,t} I(X_i > t, Y)$.
4. **Split** the tree and the dataset at node $X_i > t$ (or by value, for a categorical attribute).
5. **Create subtrees and recurse** through all of them, until a stopping condition (Section 9) is met.

# 9. When Do We Stop Recursing?
Three candidate stopping rules come up, and only two of them are actually correct.

- **Base Case One (stop):** if every point remaining in this subtree already has the same label, there is nothing left to split, make a leaf with that label.
- **Base Case Two (stop):** if every point remaining in this subtree has *identical* input features (but, e.g., conflicting labels), no attribute can possibly separate them any further, make a leaf (for example, using the majority label).
- **Base Case Three (a bad idea):** "if every remaining attribute individually has small information gain, stop." This sounds reasonable (if nothing helps, why keep going?) but it is a trap.

**The XOR warning.** Consider a label defined as $Y = X_1 \operatorname{XOR} X_2$:

| $X_1$ | $X_2$ | $Y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Splitting on $X_1$ alone sends two 0-labels and two 1-labels into *each* branch (rows 1,2 vs. rows 3,4 are an even mix), so $H(Y \mid X_1) = H(Y)$ and the information gain is exactly zero. By symmetry, splitting on $X_2$ alone also gives zero gain. Judged one feature at a time, neither attribute looks even slightly useful. And yet splitting on **both** $X_1$ and $X_2$ in sequence perfectly separates the four rows into four pure leaves: the label is completely determined once both features are known. Base Case Three would stop right at the point where a single low-information split is about to reveal a perfectly informative *pair* of splits, because it only ever asks "does the next one split help?" instead of "could two splits from here help?". The lesson is that information can be hidden in the *interaction* between features even when no individual feature carries any of it, so stopping on individually-low gain silently throws away real signal.

# 10. Strengths, Typical Uses, and the Overfitting Problem
Decision trees are one of the most widely used tools in machine learning, for a few concrete reasons that follow directly from everything above:

- **Interpretable:** the tree itself (the sequence of feature/threshold tests) *is* a human-readable explanation of the prediction, unlike, say, the weights of a neural network.
- **Cheap to train and to use:** even though the *optimal* tree is NP-hard to find (Section 4), the greedy heuristic only ever computes information gain a bounded number of times per node, and prediction is just one root-to-leaf walk.
- **Flexible beyond plain classification:** the same "split on the best question, then recurse" idea generalizes to regression (a leaf can output the average of the $y$-values that land in it instead of a majority class, with the splitting criterion swapped from entropy to something like variance reduction) and even density estimation, though this extension is only briefly gestured at in the summary slide and is developed further in general references on decision trees, e.g. the [Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning) overview.

The same flexibility that makes threshold splits powerful (Sections 6–7) is also the source of their biggest weakness: **decision trees will overfit.** Nothing in the greedy algorithm stops it from growing branches deep enough to isolate individual noisy training points (much like the "one branch per value" hypothesis that motivated switching to threshold splits in the first place, just reached one threshold at a time instead of all at once), so a fully grown tree can reach 100% training accuracy while generalizing poorly. The lecture mentions three standard remedies:

- **Fixed depth / early stopping**: stop growing once the tree reaches a maximum depth (or a minimum number of examples per node), even if the leaves are not yet pure, deliberately trading some training accuracy for a simpler, less overfit tree.
- **Pruning**: grow the tree out fully first, then remove branches after the fact if they do not actually help accuracy on held-out data.
- **Ensembles of trees (random forests)**: train many different trees (e.g. on resampled data and/or feature subsets) and combine their predictions, rather than trusting any single greedy tree.

# 11. Recap
A decision tree predicts by walking root-to-leaf through a sequence of feature tests; because finding the smallest tree consistent with the data is NP-hard, it is built greedily, one information-gain-maximizing split at a time, recursing into each resulting sub-dataset exactly as in Course Materials I's feature-scoring idea. Real-valued features are handled with binary threshold splits rather than one branch per value, and although this looks like it requires searching infinitely many thresholds, only the midpoints between adjacent, differently-labeled sorted examples ever need to be checked. Recursion stops once a subtree is pure or has no further distinguishing features, but never simply because every individual attribute looks unhelpful, since interacting features (the XOR case) can hide gain that only appears after a second split. Trees are cheap, interpretable, and flexible, but their flexibility also lets them overfit, which is why fixed depth, pruning, and ensembles like random forests are used alongside them in practice.

# Sources
- Wang, K. *CSE/ISyE 6740: Computational Data Analysis*, Lecture 6: "Feature Selection and Decision Tree," Georgia Tech, Fall 2026 (09/14/2026), slides 18–29. All formulas, the Atlanta rental-price example, the threshold-split diagrams, and the XOR stopping-condition example are adapted from this lecture; the worked "hours studied" threshold table in Section 7 is an original example constructed for this write-up, not taken from the slides.
- Hyafil, L., & Rivest, R. L. (1976). "Constructing Optimal Binary Decision Trees is NP-Complete." *Information Processing Letters*, 5(1), 15–17. (Cited in the lecture as the source of the NP-completeness result in Section 4.)
- scikit-learn developers. "Classifier comparison." *scikit-learn documentation*. https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html: source of the multi-classifier decision-boundary comparison described in Section 2 (referenced, not reproduced, here).
- "Decision tree learning." *Wikipedia*. https://en.wikipedia.org/wiki/Decision_tree_learning : background reading on regression/density-estimation extensions mentioned in Section 10.
- "Decision tree pruning." *Wikipedia*. https://en.wikipedia.org/wiki/Decision_tree_pruning: background reading on the pruning remedy mentioned in Section 10.
