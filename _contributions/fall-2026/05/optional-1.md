---
semester: fall-2026
lecture_number: "05"
slot: optional-1
slot_order: 3
role: optional_topic
role_label: Optional topic
tab_title: Extension I
assignee: JackTNorris
issue: 10
---
# Lecture 5 Optional: Wu (1983) on EM Convergence
### Primary Reference & Citation

* **Wu, C. F. J. (1983).** *On the Convergence Properties of the EM Algorithm*. **The Annals of Statistics**, 11(1), 95–103.

  Stable DOI: <https://doi.org/10.1214/aos/1176346060>

* **Secondary Reference:**

  *Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977).* *Maximum Likelihood from Incomplete Data via the EM Algorithm*. **Journal of the Royal Statistical Society: Series B (Methodological)**, 39(1), 1–38. ([DOI: 10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)).

  *Role:* DLR introduced the general formulation of EM, but their general convergence proof contained mathematical oversights that Wu rectifies.

* **AI Usage**: Google gemini was used as a tool to format latex and probe my personal understanding of the paper before trying to make an accessible guide. All output was reviewed

### Glossary of Specialized Notation

* \(\mathcal{X}, \mathcal{Y}\)**:** The complete-data space (\(\mathcal{X}\)), which contains all items in the dataspace + their assigments, and observed-data space (\(\mathcal{Y}\)) (Wu, p. 95).

* \(\Omega\)**:** The parameter space (\(\Omega \subseteq \mathbb{R}^d\)).

* \(L(\theta)\)**:** The observed-data log-likelihood, \(\ln g(y \mid \theta)\) (Wu, p. 96).

* \(Q(\theta \mid \theta')\)**:** Expected complete-data log-likelihood under \(p(Z \mid y, \theta')\) (Wu, p. 96).

* \(H(\theta \mid \theta')\)**:** Expected conditional log-density of the latent variables, \(\mathbb{E}[\ln k(Z \mid y, \theta) \mid y, \theta']\) (Wu, p. 96).

* \(M(\theta)\)**:** Point-to-set algorithmic mapping defining the M-step: \(\arg\max_{\phi \in \Omega} Q(\phi \mid \theta)\) (Wu, p. 96).

* **Closed Map:** A set-valued map where \(x_k \to x\), \(y_k \in M(x_k)\), and \(y_k \to y\) implies \(y \in M(x)\) (Wu, p. 96).

* **Stationary Point (**\(\mathcal{S}\)**):** Any point \(\theta \in \Omega\) where directional derivatives satisfy first-order optimality (for interior points, \(\nabla L(\theta) = 0\)) (Wu, p. 97).

* **Zangwill’s Global Convergence Theorem:** General optimization theorem (Zangwill, 1969) establishing convergence of point-to-set algorithmic iterates (Wu, pp. 96–97).



## 1. General Introduction

The Expectation-Maximization (EM) algorithm was formally introduced in 1977 by Dempster, Laird, and Rubin (whom we'll call DLR from now on). We'll assume basic familiarity with the EM algorithm for this write-up, but feel free to refer to previous notes or [this](https://www.youtube.com/watch?v=REypj2sy_5U) video for a great foundation of building an intuitive understanding of how the EM algorithm works. Outside of formalizing the alternating expectation ("E") and maximization ("M") steps we've learned in class, DLR made four primary claims about their algorithm:

1. **Monotonic Increase of the Likelihood (Theorem 1)**

2. **General Convergence to Stationary Points (Theorem 2)**

3. **Asymptotic Rate of Convergence (Theorems 3 & 4)**

4. **Generalization to GEM (Generalized EM)**

We'll discuss these a little further shortly, but in practice, people frequently condense these results into the casual assertion that *"EM is guaranteed to converge."* However, this conclusion masks critical distinctions between whether the *likelihood values* converge, whether the *parameters* converge, and whether the destination of the EM alogirhtm is a *global maximum*.

Jeff Wu’s 1983 paper, *“On the Convergence Properties of the EM Algorithm,”* performs a rigorous mathematical verification of DLR's claims. Wu confirms that Claim 1 holds under standard conditions, but demonstrates that DLR’s proof for Claim 2 contained critical mathematical gaps. Before we dive too deep into how he does this though, we'll briefly discuss some definitions of convergence to better understand Wu's logic.

## 2. Four Levels of Convergence

To understand Wu’s paper, we'll first clarify four levels of convergence that can often get mixed up, and roughly state if EM guarantees them.

| Level | Formal Condition | Guaranteed by EM? | Required Conditions (Wu 1983) | Layman's Terms |
| :--- | :--- | :--- | :--- | :--- |
| **1. Monotonic Improvement** | \(L(\theta^{(t+1)}) \ge L(\theta^{(t)})\) | **Yes** | Jensen's inequality and valid M-step (p. 96). | For every iteration of the EM algorithm, the "bump" in our parameters increases (or at least maintains) the likelihood of our data. |
| **2. Convergence of Likelihood Values** | \(L(\theta^{(t)}) \to L^* < \infty\) | **Yes** | \(L(\theta)\) bounded above on \(\Omega\) (Theorem 1, p. 97). | The likelihood score is guaranteed to flatten out at some ceiling—it won't bounce around indefinitely or shoot off to infinity. |
| **3. Convergence of Parameter Sequence** | \(\Vert\theta^{(t)} - \theta^*\Vert \to 0\) | **No (Not generally)** | Requires isolated stationary points and vanishing step size (Theorem 3, p. 99). | Even if the likelihood score flatlines, your actual parameter knobs might keep wandering along a flat plateau or ridge without ever settling on a single spot. |
| **4. Convergence to Global MLE** | \(L(\theta^*) = \sup_{\theta} L(\theta)\) | **No** | Requires unimodality/strict concavity (Corollary 1, p. 99); false in general. | EM will reliably climb uphill, but you're only guaranteed to get stuck on *some* local hilltop or saddle point—not necessarily the highest peak on the map. |

## 3. Monotonic Improvement and Value Convergence

Diving into DLR's first claim, they propose that successive steps of the EM algorithm produce non-decreasing observed log-likelihood values. Expressing this mathematically:

\[
L(\theta^{(t+1)}) \ge L(\theta^{(t)})
\]


Wu (p. 96) confirms this identity by decomposing the observed log-likelihood, \(L(\theta) = \ln g(y \mid \theta)\), into a solvable proxy (\(Q\)) and an entropy term (\(H\)) using the complete data \(x = (y, z)\). Before we get there though, we'll use the definition of conditional probability to write the following, where y is the observed data and \(Z\) are the missing/latent variables (like cluster assignments):

\[k(Z \mid y, \theta) = \frac{f(y, Z \mid \theta)}{g(y \mid \theta)}\]

In the context of clustering, \(k(Z \mid y, \theta)\) assigns the posterior responsibility that points belong to their respective latent clusters. In this case, \(f(y, Z \mid \theta)\) is the complete-data joint density, \(g(y \mid \theta)\) is the marginal observed-data density, and \(k(Z \mid y, \theta)\) is the posterior distribution of the missing data given the observed data. Rearranging for the observed data density \(g(y \mid \theta)\):

\[g(y \mid \theta) = \frac{f(y, Z \mid \theta)}{k(Z \mid y, \theta)}\]
Taking the natural log of both sides and using the definition of log-likelihood:
\[\ln g(y \mid \theta) = \ln f(y, Z \mid \theta) - \ln k(Z \mid y, \theta)\]
\[L(\theta) = \ln f(y, Z \mid \theta) - \ln k(Z \mid y, \theta)\]

Because \(Z\) is unobserved, we take the expectation of both sides with respect to the distribution of \(Z\) given the observed data \(y\) and our current estimate of the parameter, \(\theta'\):

\[\mathbb{E}_{Z \mid y, \theta'}[L(\theta)] = \mathbb{E}_{Z \mid y, \theta'}[\ln f(y, Z \mid \theta)] - \mathbb{E}_{Z \mid y, \theta'}[\ln k(Z \mid y, \theta)]\]

Since \(L(\theta)\) doesn't contain \(Z\), we can rewrite the equation as the following:

\[L(\theta) = \mathbb{E}_{Z \mid y, \theta'}[\ln f(y, Z \mid \theta)] - \mathbb{E}_{Z \mid y, \theta'}[\ln k(Z \mid y, \theta)]\]

Substituting in some of definitions from the paper:

\[Q(\theta \mid \theta') \equiv \mathbb{E}_{Z \mid y, \theta'}[\ln f(y, Z \mid \theta)]\]
\[H(\theta \mid \theta') \equiv \mathbb{E}_{Z \mid y, \theta'}[\ln k(Z \mid y, \theta)]\]
\[L(\theta) = Q(\theta \mid \theta') - H(\theta \mid \theta')\]

The paper states that by Jensen’s inequality, \(H(\theta \mid \theta') \le H(\theta' \mid \theta')\). In the M-step, choosing \(\theta^{(t+1)}\) such that \(Q(\theta^{(t+1)} \mid \theta^{(t)}) \ge Q(\theta^{(t)} \mid \theta^{(t)})\) forces:

\[
L(\theta^{(t+1)}) - L(\theta^{(t)}) = \underbrace{\left[Q(\theta^{(t+1)} \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)})\right]}_{\ge 0} + \underbrace{\left[H(\theta^{(t)} \mid \theta^{(t)}) - H(\theta^{(t+1)} \mid \theta^{(t)})\right]}_{\ge 0} \ge 0
\]

Therefore, every step results in a non-decreasing change in the likelihood function. 

## 4. Why DLR’s Parameter Convergence Proof Broke Down

While likelihood values converge, DLR went further and claimed that the parameter vector sequence \(\{\theta^{(t)}\}\) itself converges to a stationary point. Wu points out two central defects in DLR's derivation:

1. **The Set-Valued Nature of the M-Step:**

    DLR treated the M-step update as an ordinary continuous mapping \(\theta^{(t+1)} = M(\theta^{(t)})\). When \(Q(\theta \mid \theta')\) has multiple maximizers, however, \(M\) becomes a point-to-set map (\(M: \Omega \to \mathcal{P}(\Omega)\)). In this setting, the classical point-to-point continuity theorems of calculus no longer apply; instead, convergence of the algorithm requires establishing that \(M\) is a closed point-to-set map under Zangwill's global convergence framework (Wu, 1983, p. 96). Put simply, we can no longer guarantee that our iterate moves along a single continuous path, so we rely on set-valued closedness to ensure that the limit of any sequence generated by the algorithm remains in the set of stationary points.

2. **Likelihood Convergence Does Not Imply Parameter Convergence:**

   In multidimensional optimization, a sequence of iterates \(\{\theta^{(t)}\}\) can traverse a continuous ridge or oscillate between distinct limit points while their objective values \(L(\theta^{(t)})\) plateau at \(L^*\). In other words, there's a possibility that \(\theta\) never "settles down" on a value, even if the value of \(L\) does. 

## 5. Main Theorems and Regularity Conditions in Wu (1983)
Beyond validating DLR's first theorem and resolving the gaps in the second, Wu (1983) establishes several formal convergence results under explicit regularity assumptions:

### Regularity Assumptions and Their Roles
* **Continuity & Closedness:** Wu assumes \(Q(\theta \mid \theta')\) and \(H(\theta \mid \theta')\) are continuous in both arguments (conditions (4) and (5), p. 96). This ensures the algorithmic map \(M\) is closed, a prerequisite for Zangwill's Global Convergence Theorem, which is needed due to the point-to-set characterstic of the \(M\) function. 

* **Differentiability:** In Theorem 2 (p. 98), Wu assumes \(Q(\theta \mid \theta')\) is continuously differentiable with respect to \(\theta\) in the interior of \(\Omega\). This allows interchanging limits and derivatives to establish that \(\nabla L(\theta^*) = 0\) at limit points.

* **Compactness:** In Theorems 1 and 2, Wu assumes the sequence \(\{\theta^{(t)}\}\) is contained in a compact (closed and bounded) subset of \(\Omega\). Compactness guarantees that every subsequence has a convergent subsequence, keeping iterates away from infinite boundaries.

* **Identifiability:** If a model is non-identifiable, stationary points form connected manifolds (ridges). This violates the requirement of isolated stationary points needed for parameter convergence (Section 3, pp. 98–99). In other words, if multiple parameter values give the exact same likelihood, you get a flat ridge instead of a single sharp peak

### Formal Conclusions, Expressed Mathematically

* **Theorem 1 & Theorem 2 (pp. 97–98):** If \(M\) is closed and \(\{\theta^{(t)}\}\) remains in a compact set, every limit point \(\theta^*\) of \(\{\theta^{(t)}\}\) is a stationary point of \(L(\theta)\).

* **Theorem 3 (p. 99):** If the stationary points with likelihood \(L^*\) are isolated and \(\Vert{}\theta^{(t+1)} - \theta^{(t)}\Vert{} \to 0\), the sequence \(\{\theta^{(t)}\}\) converges to a single stationary point \(\theta^*\).

* **Corollary 1 (p. 99):** If \(L(\theta)\) is unimodal with a unique stationary point, \(\theta^{(t)}\) converges to the unique global MLE.

### Formal Conclusions, Expressed Simply

* **Theorems 1 & 2 (pp. 97–98): If your likelihood function doesn't blow up to infinity, anywhere you settle is a flat spot.**
  * *What it means:* As long as your parameter guesses stay inside a bounded range (a "compact set") and the algorithm's update rule behaves nicely without glitching at the edges ("\(M\) is closed"), any point the algorithm circles or approaches is guaranteed to have a slope of zero. In other words, EM won't stall out halfway up a steep hill; it will always land on a stationary point (a peak, a saddle point, or a valley).

* **Theorem 3 (p. 99): If the peaks of your likelihood function stand alone and your steps shrink, you are forced to lock onto a single number.**
  * *What it means:* If the flat spots are separate summits rather than long, connected flat ridges ("isolated stationary points"), and the distance between consecutive updates slows to a crawl (\(\|\theta^{(t+1)} - \theta^{(t)}\| \to 0\)), the algorithm can't hop between hills or slide endlessly along a plateau. It has no choice but to freeze on one specific coordinate vector \(\theta^*\).

* **Corollary 1 (p. 99): If there is only one hill in the entire landscape of your likelihood function, you always hit the true summit.**
  * *What it means:* When the log-likelihood looks like an upside-down bowl with a single peak and zero other flat spots ("unimodal with a unique stationary point"), EM is guaranteed to climb straight to the top. Your parameters will converge directly to the single best answer possible, AKA the unique global Maximum Likelihood Estimate (MLE).

## 6. Local Maxima, Saddles, and Boundary Pathologies

Expanding on theorem 3, Wu emphasizes that stationary points are not necessarily local maxima:

* **Saddle Points:** The stationary set \(\mathcal{S} = \{\theta \in \Omega : \nabla L(\theta) = 0\}\) contains local maxima, local minima, and saddle points (p. 98). If initialized exactly on a saddle point or symmetric manifold, EM can remain stuck because the gradient of the surrogate function vanishes in the direction of improvement (Wu, pp. 100–101). If EM lands you somewhere on the likelihood graph with the gradient is zero, it provides no way for you to leave it. 

* **Boundary Collapse:** If an iterate approaches the boundary of \(\Omega\), interior differentiability fails and compactness is violated. In Gaussian mixtures, this occurs when a component covariance collapses (\(\sigma_k^2 \to 0\)), driving likelihood to infinity.

## 7. Connection to Lecture 5 (ELBO and Minorization-Maximization)
In Lecture 5, EM is presented through the Evidence Lower Bound (ELBO):

\[
L(\theta) \ge \mathcal{L}(q, \theta)
\]

where \(\mathcal{L}(q, \theta) = \mathbb{E}_{q}[\ln p(y, Z \mid \theta)] + \mathcal{H}(q)\) and \(\mathcal{H}(q) = -\mathbb{E}_{q}[\ln q(Z)]\).

* **Wu's Formulation:** Wu writes \(L(\theta) = Q(\theta \mid \theta') - H(\theta \mid \theta')\), where \(Q(\theta \mid \theta') = \mathbb{E}_{Z \mid y, \theta'}[\ln p(y, Z \mid \theta)]\) and \(H(\theta \mid \theta') = \mathbb{E}_{Z \mid y, \theta'}[\ln p(Z \mid y, \theta)]\). Setting \(q(Z) = p(Z \mid y, \theta')\) gives \(\mathcal{H}(q) = -H(\theta' \mid \theta')\), which yields:

  \[
  \mathcal{L}(q, \theta) = Q(\theta \mid \theta') - H(\theta' \mid \theta')
  \]

* **E-step:** Setting \(q(Z) = p(Z \mid y, \theta')\) makes the lower bound tight at \(\theta'\): \(\mathcal{L}(q, \theta') = Q(\theta' \mid \theta') - H(\theta' \mid \theta') = L(\theta')\).

* **M-step:** Maximizing \(\mathcal{L}(q, \theta)\) over \(\theta\) is equivalent to maximizing \(Q(\theta \mid \theta')\) because the entropy term \(H(\theta' \mid \theta')\) depends only on \(\theta'\) and does not depend on \(\theta\).
## 8. Practical Takeaways for Fitting a Gaussian Mixture Model (GMM)
Wu's work allows us to draw the following takeaways for more effective implementation of the EM algorithm: 

1. **Regularize Covariances to Preserve Compactness:**

   The GMM likelihood is unbounded as \(\sigma_k^2 \to 0\). Add a small diagonal regularizer (\(\Sigma_k + \epsilon I\)) to keep iterates inside a compact set in the interior of \(\Omega\).

2. **Always Use Multiple Restarts:**

   EM does not guarantee a global maximum. Run EM with 10–50 random initializations (or use k-means++) and select the solution with the highest log-likelihood. There's always the possibility you get stuck in a saddle or local optimum if you only run the algorithm once.

3. **Dual Stopping Criteria:**

   Do not stop solely when \(\vert{}L(\theta^{(t+1)}) - L(\theta^{(t)})\vert{} < \varepsilon\). Likelihood changes can become negligible on flat plateaus while parameters are still migrating. Monitor both likelihood change and parameter step size:
   

   \[
   \vert{}L(\theta^{(t+1)}) - L(\theta^{(t)})\vert{} < \varepsilon \quad \text{and} \quad \Vert{}\theta^{(t+1)} - \theta^{(t)}\Vert{} < \delta
   \]
