---
semester: fall-2026
lecture_number: "17-diffusion"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: HUST-AI-HYZ
issue: 53
---
## From Maximum Likelihood to the DDPM Loss

These notes cover slides 1–13 of Lecture 17. Slide 2 lists the lecture's four topics: the DDPM loss, the model architecture, the computation cost, and applications. This section covers the first topic and the demo on slide 13. Core explanation II starts at slide 14.

Lecture 16 ended with a training recipe that looks almost too simple: add noise to an image, ask a network to guess that noise, and minimize the squared error. Nothing in that recipe mentions probability, yet the network it trains is the core of a probabilistic generative model. Slide 4 asks the natural question: *why this loss function?*

The answer is one chain of reasoning, and these notes follow it link by link:

1. **Goal:** maximize the likelihood of the training data (slide 6).
2. **Obstacle:** backpropagating through the sampling chain is expensive, and the likelihood itself is an integral over every denoising path (slides 7–8).
3. **Bound:** Jensen's inequality gives an upper bound on the negative log-likelihood that we can estimate by sampling (slide 8).
4. **Split:** the Markov structure and Bayes' rule break the bound into one KL divergence per step (slides 9–10).
5. **Simplify:** each KL compares two Gaussians, so it reduces to a weighted squared error between noises (slide 11).
6. **Train:** dropping the weights gives the DDPM loss (slide 12), and slide 13 shows the noise model at work.

### Notation

The notation follows Lecture 16.

| Symbol | Meaning |
|---|---|
| $$x_0$$ | A clean sample from the training set $$D_{\text{train}}$$ |
| $$x_t$$ | The sample after $$t$$ noising steps, with $$x_t\in\mathbb{R}^d$$ |
| $$x_{1:T}$$ | The whole path $$x_1,\dots,x_T$$ |
| $$\beta_t$$ | The noise schedule: the variance of the noise added at step $$t$$ |
| $$\alpha_t=1-\beta_t$$ | The share of signal variance kept at step $$t$$ |
| $$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$$ | The share of signal variance kept over the first $$t$$ steps |
| $$q$$ | The forward (noising) process, which is fixed |
| $$p_\theta$$ | The reverse (denoising) model with learned parameters $$\theta$$ |
| $$\epsilon_t$$ | The standard Gaussian noise relating $$x_0$$ to $$x_t$$ (see below) |
| $$\epsilon_\theta(x_t,t)$$ | The noise model: a network's prediction of $$\epsilon_t$$ |
| $$\sigma_t^2$$ | The fixed variance of the reverse step at time $$t$$ |

Two facts from Lecture 16 do most of the work below. One forward step is a Gaussian, and the whole forward process has a closed form that jumps from $$x_0$$ to any $$x_t$$ in one move:

$$
\begin{aligned}
q(x_t\mid x_{t-1})&=\mathcal{N}\big(x_t;\ \sqrt{\alpha_t}\,x_{t-1},\ \beta_t I\big),\\
x_t&=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon_t .
\end{aligned}
$$

So $$\epsilon_t$$ is the total noise accumulated between $$x_0$$ and $$x_t$$, rescaled to unit variance. It is not only the fresh noise added at step $$t$$. The slides' loss formulas write $$\mathcal{N}(0,1)$$ for standard Gaussian noise in every coordinate; these notes write $$\mathcal{N}(0,I)$$.

### 1. The Recipe to Explain (slides 3–4)

One training step (Algorithm 1, drawn on slide 3) does the following:

1. Sample an image $$x_0$$ from the training set.
2. Sample a step $$t$$ uniformly from $$\{1,\dots,T\}$$ and noise $$\epsilon_t\sim\mathcal{N}(0,I)$$.
3. Jump to the noisy image $$x_t$$ with the closed form above.
4. Feed $$x_t$$ and $$t$$ to the network to get its guess $$\epsilon_\theta(x_t,t)$$.
5. Take a gradient step on the squared error between the true noise and the guess.

Averaged over all those random choices, this minimizes

$$
L_{\text{DDPM}}
:=\mathbb{E}_{t\sim[1,T],\ x_0\sim D_{\text{train}},\ \epsilon_t\sim\mathcal{N}(0,I)}
\Big[\lVert\epsilon_t-\epsilon_\theta(x_t,t)\rVert^2\Big].
$$

Slide 4 also puts a $$\sum_t$$ in front of this expectation. The expectation over $$t$$ already averages over the steps, so these notes use the form on slide 12, which has the same minimizer.

Generation (Algorithm 2) runs in the opposite direction. It starts from pure noise $$x_T\sim\mathcal{N}(0,I)$$ and repeats, for $$t=T,\dots,1$$,

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)+\sigma_t z,
$$

where $$z\sim\mathcal{N}(0,I)$$, except that $$z=0$$ at the final step $$t=1$$.

Two questions need answers. Why is the training target the noise? And why should a squared error be the right objective for a probabilistic model? Section 6 answers both.

### 2. The Goal: Maximum Likelihood (slide 6)

A generative model should make the training data likely. Slide 6 phrases this as follows: when the input is $$x_0$$, the model should be likely to produce that same $$x_0$$ as its output. Here $$p_\theta(x_0)$$ is the probability density with which the reverse chain, started from pure noise, ends at $$x_0$$. Maximizing it is the same as minimizing the negative log-likelihood:

$$
\min_\theta\ -\mathbb{E}_{x_0\sim D_{\text{train}}}\big[\log p_\theta(x_0)\big].
$$

There are two ways to attack this objective. Section 3 tries the direct route and shows why it is impractical. Sections 4–7 follow the route DDPM takes.

### 3. The Direct Route: Backpropagation Through Time (slide 7)

The most direct idea is to treat sampling itself as the model. Algorithm 2 calls the same network $$T$$ times, so the generated $$x_0$$ is the output of a very deep network that reuses its weights at every layer. If $$x_0$$ depends on $$\theta$$ through this chain, the chain rule gives two paths from $$\theta$$ to the objective:

$$
\frac{d}{d\theta}\log p_\theta(x_0)
=\frac{\partial}{\partial\theta}\log p_\theta(x_0)
+\frac{\partial\log p_\theta(x_0)}{\partial x_0}\,\frac{dx_0}{d\theta}.
$$

The expensive part is $$dx_0/d\theta$$. Each state depends on $$\theta$$ directly, through the network call in its own update, and indirectly, through the previous state in the chain:

$$
\frac{dx_{t-1}}{d\theta}
=\frac{\partial x_{t-1}}{\partial\theta}
+\frac{\partial x_{t-1}}{\partial x_t}\,\frac{dx_t}{d\theta},
\qquad t=1,\dots,T.
$$

The recursion ends at $$x_T$$, which is fresh noise and does not depend on $$\theta$$. Slide 7 shows the case $$t=1$$, with the first term printed as $$\partial x_0/\partial x_1$$; the chain rule gives $$\partial x_0/\partial\theta$$, as written here. Unrolling a shared network like this is called backpropagation through time (BPTT), the same method used to train recurrent networks.

Slide 7 then asks what this costs. For one rollout of the chain:

- **Computation:** rolling out the chain takes $$T$$ forward passes of the network, and backpropagating takes $$T$$ backward passes. The lecture counts each pass as on the order of $$\dim(\theta)$$ operations, so one gradient costs $$O(\dim(\theta)\,T)$$.
- **Memory:** plain backpropagation keeps each step's intermediate values until the backward pass reaches that step, so memory grows as $$T$$ times what one pass stores. The lecture put this at $$O(\dim(\theta)\,T)$$ as well.

Ho et al. use $$T=1000$$, so one rollout would take 1000 forward and 1000 backward passes. The Classroom Dialog below puts numbers on these costs.

Cost is not the only problem. The $$x_0$$ in the objective is a fixed training image, while the $$x_0$$ in $$dx_0/d\theta$$ is a sample generated by the chain, so these formulas do not yet describe a training step. They also assume we can evaluate $$\log p_\theta(x_0)$$, and Section 4 shows that we cannot. Slide 7 is best read as an argument about cost rather than as a working algorithm.

### 4. A Bound We Can Estimate (slide 8)

The reverse chain can reach the same $$x_0$$ through many different routes, and $$p_\theta(x_0)$$ adds up all of them:

$$
p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}.
$$

This is an integral over $$Td$$ dimensions with a neural network inside it, so it has no closed form.

The fix is the one used for Gaussian mixtures with the EM algorithm in Lecture 5: multiply and divide by a distribution over the hidden variables, then move the log inside the expectation. Here the hidden variables are the path $$x_{1:T}$$, and we also need a distribution we can sample from. The natural choice is the forward process $$q(x_{1:T}\mid x_0)$$, which serves as an approximate posterior over the path, the role the posterior played in EM:

$$
\begin{aligned}
-\mathbb{E}_{x_0\sim D_{\text{train}}}\big[\log p_\theta(x_0)\big]
&=-\mathbb{E}_{x_0}\Big[\log\int q(x_{1:T}\mid x_0)\,\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T}\Big]\\
&\le-\mathbb{E}_{x_0,\ x_{1:T}\sim q}\Big[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big]=:L.
\end{aligned}
$$

Here $$x_{1:T}\sim q$$ is shorthand for drawing the path from $$q(x_{1:T}\mid x_0)$$. The last step is Jensen's inequality. Since $$\log$$ is concave, $$\log\mathbb{E}[Y]\ge\mathbb{E}[\log Y]$$, and the leading minus sign flips the direction. We call the right-hand side $$L$$.

Using $$q$$ helps for two reasons. It is fixed and known, so we can sample paths and evaluate their probabilities without training anything. And its paths all start at the given $$x_0$$, so they are routes that actually connect to that image. Paths drawn from the reverse chain would rarely end near a particular training image.

Minimizing $$L$$ pushes down an upper bound on the negative log-likelihood. In EM, Jensen's inequality gave a lower bound on the log-likelihood to maximize; with the minus sign, the same step gives an upper bound to minimize. The connection to VAEs comes with slide 15 in Core explanation II.

### 5. One Term per Step (slides 9–10)

$$L$$ is still an expectation over whole paths. The next steps split it into pieces that each involve a single time step.

**Factor both chains (slide 9).** Both processes are Markov chains, so a path probability is a product of one-step terms:

$$
\begin{aligned}
q(x_{1:T}\mid x_0)&=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),\\
p_\theta(x_{0:T})&=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
\end{aligned}
$$

The slides write the starting distribution as $$p_\theta(x_T)$$, but it is the fixed prior $$\mathcal{N}(0,I)$$ with nothing to learn. Flipping the fraction in $$L$$ to absorb the minus sign and taking the log of the products gives

$$
L=\mathbb{E}_{x_0,\ x_{1:T}\sim q}\Big[-\log p(x_T)+\sum_{t=1}^{T}\log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}\Big].
$$

**Reverse the direction of $$q$$ (slide 10).** This sum is not yet a KL divergence, because the numerator is a distribution over $$x_t$$ and the denominator is a distribution over $$x_{t-1}$$. We need $$q$$ written as a backward step too. Lecture 16 (slide 17) noted that the plain reverse step $$q(x_{t-1}\mid x_t)$$ is hard. One reason is that it would average over every clean image that could have produced $$x_t$$. Conditioning on $$x_0$$ removes that difficulty, and the Markov property lets us add the condition for free: $$q(x_t\mid x_{t-1})=q(x_t\mid x_{t-1},x_0)$$. Bayes' rule then gives the rewriting used on slide 10:

$$
q(x_t\mid x_{t-1})=q(x_{t-1}\mid x_t,x_0)\cdot\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}.
$$

**Telescope.** Taking the log of the second factor and summing over the steps, each numerator cancels the next step's denominator. Slide 10's third line keeps the end at $$x_T$$, which adds $$\log q(x_T\mid x_0)$$ next to $$-\log p_\theta(x_T)$$. The Classroom Dialog below records how the other end was handled in class.

**Group into KL divergences.** Slide 10's last line collects the terms into one KL divergence per step. It writes the sum from $$t=1$$ and notes that $$L_0$$ is often separated and ignored. The precise statement, Eq. (5) of Ho et al., keeps the $$t=1$$ term separate as $$L_0$$:

$$
\begin{aligned}
L=\mathbb{E}_{x_0,\ x_{1:T}\sim q}\Big[
&\underbrace{D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\Vert\,p(x_T)\big)}_{L_T}\\
&+\sum_{t=2}^{T}\underbrace{D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}}\\
&\underbrace{-\log p_\theta(x_0\mid x_1)}_{L_0}\Big].
\end{aligned}
$$

The three kinds of term have clear meanings:

| Term | Question it asks | Depends on $$\theta$$? |
|---|---|---|
| $$L_T$$ | Does the fully noised sample look like the pure-noise prior? | No, so it is constant during training |
| $$L_{t-1}$$ | Does the learned reverse step match the true one given $$x_0$$? | Yes; these terms train the model |
| $$L_0$$ | Does the last reverse step land on $$x_0$$? | Yes, but slide 10 sets it aside |

$$L_T$$ has nothing to train because $$q$$ and the prior are both fixed. For the schedule used by Ho et al. it is also tiny, about $$10^{-5}$$ bits per dimension.

The payoff is that each $$L_{t-1}$$ depends only on $$x_t$$ and $$x_0$$. We no longer need to simulate a whole path, because the closed form from the Notation section produces $$x_t$$ in one jump. This removes the need to roll out and backpropagate through the chain. The sum still has one term per step, and Section 7 deals with that by sampling a single $$t$$.

### 6. From KL Divergence to Squared Error (slide 11)

Apart from $$L_0$$, all the learning now sits in the terms $$L_{t-1}$$, and each one compares two Gaussian distributions over $$x_{t-1}$$.

**The model's reverse step.** From Lecture 16, $$p_\theta(x_{t-1}\mid x_t)$$ is a Gaussian with fixed variance $$\sigma_t^2 I$$. Sampling from it adds fresh noise $$\sigma_t z$$ at every step except the last, so the denoising process stays stochastic; in class this variance was described as added on purpose. Its mean is the noise-free part of the sampling update from Section 1:

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big).
$$

**The true reverse step.** Slide 11 asks for the mean of $$q(x_{t-1}\mid x_t,x_0)$$. Bayes' rule gives

$$
q(x_{t-1}\mid x_t,x_0)=\frac{q(x_t\mid x_{t-1},x_0)\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}.
$$

Every factor on the right is a known Gaussian from Lecture 16, and the denominator does not involve $$x_{t-1}$$. As a function of $$x_{t-1}$$, the numerator is the exponential of a quadratic, so the true reverse step is Gaussian too. Its variance depends only on the noise schedule. Finding its mean and then replacing $$x_0$$ using the closed form for $$x_t$$ gives the result on the slide:

$$
\tilde\mu_t(x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_t\Big).
$$

The lecture skipped this algebra, and these notes do the same. The reading recommended on slide 12 works it out (see the "Reverse diffusion process" section of Weng's post).

Now compare the two means. They have exactly the same form, and the only difference is $$\epsilon_\theta(x_t,t)$$ in place of $$\epsilon_t$$. This is by design. The network already receives $$x_t$$ as input, so the one thing it does not know is the noise. DDPM therefore defines $$\mu_\theta$$ to mirror the true mean and asks the network only for the missing piece. **This is why the training target is the noise.**

**From KL to squared error.** Both Gaussians have fixed, isotropic variances. For two such Gaussians, the KL divergence is the squared distance between their means, scaled by $$\frac{1}{2\sigma_t^2}$$, plus a constant $$C_t$$ that involves only the two variances and not $$\theta$$. Substituting the two means, the $$x_t$$ terms cancel and only the noises remain:

$$
\begin{aligned}
L_{t-1}
&=\frac{1}{2\sigma_t^2}\Big\lVert
\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_t\Big)\\
&\qquad\qquad-\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)
\Big\rVert^2+C_t\\
&=\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\sigma_t^2}\,\lVert\epsilon_t-\epsilon_\theta(x_t,t)\rVert^2+C_t.
\end{aligned}
$$

Slide 11 leaves $$\frac{1}{2\sigma_t^2}$$ and $$C_t$$ out of its first line. The factor returns in its second line, which agrees with Eq. (12) of Ho et al. once $$C_t$$ is dropped.

**This is why the loss is a squared error.** A KL divergence between Gaussians with fixed variances only measures how far apart their means are, and here the means differ only through the noise.

### 7. Learning the Noise Model (slide 12)

Putting Sections 5 and 6 together, with $$\lambda_t=\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\sigma_t^2}$$,

$$
\begin{aligned}
-\mathbb{E}_{x_0\sim D_{\text{train}}}\big[\log p_\theta(x_0)\big]\le L
&=\mathbb{E}\Big[\sum_{t=2}^{T}\lambda_t\,\lVert\epsilon_t-\epsilon_\theta(x_t,t)\rVert^2\Big]\\
&\quad+\mathbb{E}[L_0]+\text{const},
\end{aligned}
$$

where the expectation is over $$x_0$$ and the noises $$\epsilon_t$$. Slide 12 sets $$L_0$$ and the constant aside and writes this sum as $$\sum_{t=1}^{T-1}L_t$$. Those are the same terms with the index shifted by one: $$L_t$$ is the term for step $$t+1$$. The slide's next expression keeps the subscript $$t$$ on the weight and noise; with those subscripts, the range should be $$t=2,\dots,T$$, as written above.

DDPM then makes one more move: it drops $$\lambda_t$$ and samples $$t$$ uniformly instead of summing. This gives the loss from Section 1:

$$
\begin{aligned}
L_{\text{DDPM}}
&:=\mathbb{E}_{t\sim[1,T],\ x_0\sim D_{\text{train}},\ \epsilon_t\sim\mathcal{N}(0,I)}
\Big[\lVert\epsilon_t-\epsilon_\theta(x_t,t)\rVert^2\Big],\\
x_t&=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon_t .
\end{aligned}
$$

In words, we train a noise model $$\epsilon_\theta(x_t,t)$$ to predict the noise mixed into the corrupted sample $$x_t$$, which tells the sampler how much noise to remove at each step. Four details are worth spelling out.

**Dropping the weights is a choice, not an identity.** In class it was presented as a simplification. $$L_{\text{DDPM}}$$ is a reweighted version of the bound, so it is no longer an upper bound on the negative log-likelihood. The table shows what was dropped, using the schedule of Ho et al. ($$T=1000$$, with $$\beta_t$$ rising linearly from $$10^{-4}$$ to $$0.02$$) and $$\sigma_t^2=\beta_t$$, one of the two variance choices they tested:

| Step $$t$$ | Signal multiplier $$\sqrt{\bar\alpha_t}$$ | Weight $$\lambda_t$$ |
|---|---|---|
| 2 | 0.9999 | 0.2727 |
| 10 | 0.9991 | 0.0737 |
| 100 | 0.9471 | 0.0101 |
| 250 | 0.7239 | 0.0053 |
| 500 | 0.2803 | 0.0055 |
| 1000 | 0.0064 | 0.0102 |

At $$t=2$$, where the image is almost clean, the bound gives roughly 50 times the weight of the middle steps, and by $$t=10$$ that ratio has dropped below 15. Setting every weight to 1 shifts training effort away from those first steps and toward the noisier, harder ones. Ho et al. report the trade-off on CIFAR-10: compared with training on the bound, the simplified loss gave much better sample quality (FID 3.17 versus 13.51) and a slightly worse upper bound on test negative log-likelihood (3.75 versus 3.70 bits per dimension).

**The weights matter because the network is shared.** Multiplying one term of the loss by a positive constant does not move that term's minimizer. So if each $$t$$ had its own network with separate parameters, every term could be minimized on its own and the weighting would not change the result (Song et al. make this point for DDIM). In practice one network is shared across all $$t$$, and the weights decide where its limited capacity goes.

**The step $$t=1$$ is still trained.** Algorithm 1 samples $$t$$ from $$\{1,\dots,T\}$$, so "ignoring" $$L_0$$ on slide 10 refers to the derivation. Ho et al. explain the $$t=1$$ term of the simplified loss as an approximation of $$L_0$$.

**No rollout is needed.** One uniform sample of $$t$$ gives an unbiased estimate of the average over all steps, and the closed-form jump produces $$x_t$$ without running the reverse chain. A training step therefore never unrolls the sampling process the way slide 7 does. Slide 21, in Core explanation II, works out what this saves per gradient step.

### 8. The Demo: What Noise Prediction Looks Like (slide 13)

Slide 13 shows the noise model at work on one image at a single noise level, in six panels:

| Panel | What it shows | In symbols |
|---|---|---|
| 1. Reference image | The clean image | $$x_0$$ |
| 2. Noisy input | What the network receives | $$x_t=0.9203\,x_0+0.3911\,\epsilon$$ |
| 3. Scaled noise added | The noise inside the input | $$0.3911\,\epsilon$$ |
| 4. True noise | The regression target | $$\epsilon$$ |
| 5. Predicted noise | The network's output | $$\epsilon_\theta(x_t,t)$$ |
| 6. Squared error heatmap | The error at each pixel | $$(\epsilon-\epsilon_\theta(x_t,t))^2$$ |

**The two multipliers.** The header values are the coefficients of the closed-form jump: the signal multiplier is $$\sqrt{\bar\alpha_t}=0.9203$$ and the noise multiplier is $$\sqrt{1-\bar\alpha_t}=0.3911$$. Their squares, 0.847 and 0.153, add to 1 up to rounding, as they must. So $$\bar\alpha_t\approx0.85$$ at this $$t$$, which is why the cat is still easy to see in panel 2. Panel 3 is panel 4 scaled by 0.3911, which is why it looks fainter.

**The error.** Panels 4 and 5 look identical by eye, so the reported numbers are more informative. The noise MSE is 0.0358 for this image and 0.0302 on average over the demo's three images at this $$t$$. For scale, if the MSE is averaged over pixels, a network that always predicted zero would score about 1, because $$\epsilon$$ has unit variance.

**The heatmap.** Panel 6 shows that the error is not spread evenly. It is close to zero in the smooth background and largest on the whiskers and fur. Smooth regions are easy because the image barely changes from pixel to pixel, so any fine-grained variation must be noise. The fur has fine-grained variation of its own, and that real detail is hard to tell apart from noise.

### Summary: Why This Loss?

| Slide | Step | What it gives us |
|---|---|---|
| 6 | Maximum likelihood | The right goal, but not computable directly |
| 7 | Backpropagation through time | A direct route whose compute and memory grow linearly in $$T$$ ($$\dim(\theta)\,T$$ in the lecture's count) |
| 8 | Forward-process paths and Jensen's inequality | An upper bound $$L$$ that we can estimate by sampling |
| 9–10 | Markov factorization, Bayes' rule, telescoping | One KL divergence per step, each needing only $$x_t$$ and $$x_0$$ |
| 11 | KL divergence between Gaussians | A weighted squared error between noises |
| 12 | Drop the weights and sample $$t$$ uniformly | $$L_{\text{DDPM}}$$, trained one sampled step at a time without unrolling the chain |
| 13 | Demo | Accurate noise prediction, with the remaining error on fine detail |

The squared noise error is therefore more than a convenient regression target. It is a reweighted form of a variational bound on the negative log-likelihood, and it can be estimated one sampled step at a time. Core explanation II picks up at slide 14 with the network that plays the role of $$\epsilon_\theta$$.

### Classroom Dialog and Discussion

- **Why start from maximum likelihood?** When it is not clear what model to use, maximum likelihood is the first thing to try, as it was for Gaussian mixtures in density estimation. For a diffusion model, it means that if a cat image goes through the diffusion process, the denoising process should be likely to bring the same cat back.
- **What do the costs on slide 7 come to?** In the lecture's count, both are on the order of $$\dim(\theta)\times T$$. If backpropagating through one network call took about 0.1 seconds, a single gradient step with $$T=1000$$ would take about 100 seconds, and training needs thousands of gradient steps or more. For memory, the same count puts a model with a billion parameters, unrolled over 1000 steps, at about $$10^{12}$$ stored values.
- **Does storing activations add another factor?** Only a constant one, depending on what is stored, such as the values from the forward pass and the gradients in the backward pass. In the lecture's count the order stays $$\dim(\theta)\times T$$, and even with a constant of 1 that is too large.
- **Where have we seen the trick on slide 8?** In EM for Gaussian mixtures, Jensen's inequality gave a lower bound on the log-likelihood; with the minus sign, the same step gives an upper bound here. The integral it bounds adds up every denoising route to the same image, so a route that passes through another cat, or even a dog, still counts.
- **How does the telescoping sum end?** In the handwritten steps on slide 10, the product started at $$t=1$$ with the factor $$q(x_1\mid x_0)/q(x_0\mid x_0)$$. Treating $$q(x_0\mid x_0)$$ as 1 leaves only $$q(x_T\mid x_0)$$.
- **What are the two Gaussians on slide 11?** The model's reverse step has the mean of the sampling update and a variance added on purpose to keep the denoising process stochastic. The true reverse step is also Gaussian, since it comes from a product of Gaussian densities; the lecture skipped its algebra. After substitution, the scalars and the $$x_t$$ terms of the two means match, so only the noise differs.
- **Why drop the weights on slide 12?** In class, the weights were dropped for simplicity. Section 7 adds the empirical reason that Ho et al. report.
- **What did the live demo show?** It walked through a single training step on the cat image. The predicted noise came out on the same scale as the true noise, the bright spots in the heatmap marked the pixels predicted incorrectly, and the training loss went down over the few iterations shown.

This section summarizes in-class discussions based on notes I took during the lecture and subsequently organized for inclusion here. The questions and answers are edited summaries, not verbatim quotations.

### References

1. Kai Wang. *CSE/ISyE 6740 Lecture 17: Diffusion Models*, October 28, 2026, slides 1–13, and the in-class explanations summarized in the Classroom Dialog. Main source for the derivation and the demo. [Course slides](https://github.com/GT-KOALA/Gatech-CSE6740/blob/main/assets/semesters/fall-2026/lectures/17/lecture-17-diffusion-models.pdf).
2. Kai Wang. *CSE/ISyE 6740 Lecture 16: Diffusion Models*, October 26, 2026. Source for the forward process, the noise schedule, and the reverse model. [Course slides](https://github.com/GT-KOALA/Gatech-CSE6740/blob/main/assets/semesters/fall-2026/lectures/16/lecture-16-diffusion-models.pdf).
3. Kai Wang. *CSE/ISyE 6740 Lecture 5: Gaussian Mixture Models (GMMs)*, September 9, 2026. Source for the Jensen lower bound and the role of the posterior in the EM algorithm. [Course slides](https://github.com/GT-KOALA/Gatech-CSE6740/blob/main/assets/semesters/fall-2026/lectures/05/lecture-05-expectation-maximization.pdf).
4. Jonathan Ho, Ajay Jain, and Pieter Abbeel. *Denoising Diffusion Probabilistic Models*, NeurIPS 2020. Source for Eqs. (5) and (12), the simplified objective, the noise schedule, the size of $$L_T$$, and the CIFAR-10 results. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
5. Lilian Weng. *What are Diffusion Models?*, Lil'Log, July 11, 2021. The reading recommended on slide 12. [Blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
6. Jiaming Song, Chenlin Meng, and Stefano Ermon. *Denoising Diffusion Implicit Models*, ICLR 2021, Section 3.2. Source for the remark that the optimum does not depend on the weights when parameters are not shared across $$t$$. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502).

---

**AI assistance:** I led the preparation of these notes and wrote most of the content. I subsequently used Claude (Anthropic) to polish the language, improve clarity, and help identify potential errors.
