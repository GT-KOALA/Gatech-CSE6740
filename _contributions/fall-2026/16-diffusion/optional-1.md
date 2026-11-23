---
semester: fall-2026
lecture_number: "16-diffusion"
slot: optional-1
slot_order: 3
role: optional_topic
role_label: Optional topic
tab_title: Extension I
assignee: egesamuray
issue: 52
---

# Conditional Diffusion Models: Classifier Guidance and Classifier-Free Guidance

Suppose a diffusion model has learned the ten handwritten digits. How does it draw any digit? How would you ask it for a **3**? And once it already accepts the label **3**, what would it mean to turn up the guidance? Lecture 16 gives us the noise predictor for the first question. Lecture 17 adds a class input for the second. The third question concerns what we do with the predictor **during sampling**.

**Learning objectives.** By the end of these notes, you should be able to derive the conditional score from Bayes' rule, convert it into a DDPM noise prediction, and trace one reverse step with either kind of guidance. You should also be able to read a guidance scale in a paper and identify which model predictions it combines.

## 1. Introduction and DDPM Recap

Let $$x_0$$ be a clean example, $$t\in\{1,\ldots,T\}$$ a noise step, and $$I$$ the identity matrix in data space. In the forward process, $$\beta_t$$ is the variance of the fresh noise added at step $$t$$. Following Lecture 16, slides 12–13, write $$\alpha_t=1-\beta_t$$ and $$\bar\alpha_t=\prod_{j=1}^t\alpha_j$$. One forward step and its cumulative effect are

<div markdown="1" style="overflow-x:auto">

$$
\begin{aligned}
q(x_t\mid x_{t-1})&=\mathcal N(\sqrt{\alpha_t}\,x_{t-1},\beta_t I),\\
x_t&=a_t x_0+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal N(0,I),\\
a_t&=\sqrt{\bar\alpha_t},\qquad \sigma_t=\sqrt{1-\bar\alpha_t}.
\end{aligned}
$$

</div>

The second line lets us jump directly to a chosen noise level. During training, draw a data point, a uniform timestep and independent Gaussian noise. A network with parameters $$\theta$$ sees $$x_t$$ and $$t$$. Its target is the noise used to construct this particular training pair [1, 2].

<div markdown="1" style="overflow-x:auto">

$$
\mathcal L_{\rm simple}
=\mathbb E_{x_0,t,\epsilon}
\big[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\big].
$$

</div>

During generation, the clean example is unknown. With a schedule that makes $$a_T$$ small, start from $$x_T\sim\mathcal N(0,I)$$. Use the learned reverse steps in the order $$T,T-1,\ldots,1$$ to obtain $$x_0$$. The label never enters this procedure, so any of the ten digits is a possible result.

<img src="https://www.calvinyluo.com/assets/images/diffusion/vdm_base.webp" alt="Forward arrows carry a dog image toward Gaussian noise. Reverse arrows point from each noisy state toward the preceding, cleaner state." width="680" style="max-width:100%;height:auto">

*Figure 1. Read the bottom arrows from left to right during corruption and the top arrows from right to left during generation. The middle picture is an example of what the denoiser receives.* [Luo, 2022, Fig. 3 (tutorial printout, p. 12)](https://arxiv.org/pdf/2208.11970#page=7) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## 2. What Is Conditioning?

### 2.1 Choosing a population

Imagine two equally common classes of points on a sheet of paper. Class 0 forms a cloud centered two units left of the origin. Class 1 forms an equally wide cloud two units right. If we cover the labels and draw a point at random, its density is the mixture

<div markdown="1" style="overflow-x:auto">

$$
p(x_0)=\tfrac12p(x_0\mid y=0)+\tfrac12p(x_0\mid y=1).
$$

</div>

Uncovering a label changes the population we draw from. A request for class 1 selects the entire right-hand cloud, including its spread. Returning its center every time would lose that variation. In the digit example, the analogous spread includes many ways of writing a 3.

<div markdown="1" style="overflow-x:auto">

| Population | Density along the horizontal axis, left to right |
| --- | --- |
| Both classes | ▁▂▂▃▄▄▅▄▄▃▃▂▂▂▃▃▄▄▅▄▄▃▂▂▁ |
| Class 0 | ▂▃▄▅▇███▇▅▄▃▂▁▁▁▁▁▁▁▁▁▁▁▁ |
| Class 1 | ▁▁▁▁▁▁▁▁▁▁▁▁▂▃▄▅▇███▇▅▄▃▂ |

</div>

*Figure 2. Density profiles for the two unit-variance Gaussian classes in Section 6. Each position samples the same horizontal coordinate, from −4 to 4. Bar heights share one scale. Averaging the last two rows produces the first row before rounding.*

### 2.2 A directly conditioned DDPM

The condition $$y$$ is information available at generation time. It might be a label, a text prompt, or a low-resolution observation. In the class-conditional DDPM considered here, we draw a labeled pair $$(x_0,y)$$ and add noise by the same rule as before. Hence $$q(x_t\mid x_0,y)=q(x_t\mid x_0)$$. The noisy *population* can still depend on the class because the clean examples came from that class.

Lecture 17, slide 18, adds $$y$$ to the inputs of the noise model. Think of a heavily blurred digit whose lower loop could belong to a 3 or an 8. A squared-error predictor averages the noise explanations compatible with what it sees. Giving it the label changes which clean digits can contribute to that average. The target in training is still the injected noise [1, 3].

<div markdown="1" style="overflow-x:auto">

$$
\mathcal L_{\rm cond}
=\mathbb E_{(x_0,y),t,\epsilon}
\big[\|\epsilon-\epsilon_\theta(x_t,t,y)\|_2^2\big].
$$

</div>

To see how the label affects a reverse step, suppose for a moment that the clean point were known. The Gaussian forward posterior has mean [2, Eq. 7]

<div markdown="1" style="overflow-x:auto">

$$
\mathbb E_q[x_{t-1}\mid x_t,x_0]
=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sigma_t^2}(x_t-a_t x_0)\right).
$$

</div>

The residual $$x_t-a_t x_0$$ equals $$\sigma_t\epsilon$$. When only $$x_t,t,y$$ are known, average over the possible clean origins. This replaces $$\epsilon$$ by its conditional expectation, which the trained network estimates. For a DDPM with fixed reverse variance, the resulting model mean is

<div markdown="1" style="overflow-x:auto">

$$
\mu_\theta(x_t,t,y)
=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sigma_t}\epsilon_\theta(x_t,t,y)\right).
$$

</div>

### 2.3 Where guidance enters

For an unconditional draw, evaluate the denoiser with only the noisy digit and time. For an ordinary conditional draw, supply “3” at every step and use the mean above as written. A **guided** draw introduces a further choice at sampling time. We can adjust each reverse step using evidence for “3”, with a scale controlling how much that evidence moves the step. To see what counts as evidence at a noisy input, it helps to work with scores.

## 3. Mathematical Foundations

### 3.1 The conditional score identity

At a fixed noise level, let $$p_t(x_t)$$ denote the density of all noisy examples and $$p_t(x_t\mid y)$$ that of the labeled class. Their **scores** are gradients of log density with respect to the input. Each score is a vector with the same shape as $$x_t$$. For a small displacement in a given direction, its dot product with that displacement gives the first-order change in log density [4].

Write the two scores as $$s_u=\nabla_{x_t}\log p_t(x_t)$$ and $$s_c=\nabla_{x_t}\log p_t(x_t\mid y)$$. Bayes' rule gives $$p_t(x_t\mid y)=p_t(x_t)p_t(y\mid x_t)/p(y)$$ for a class of positive prior probability. Take logarithms, then differentiate at fixed $$t$$. The log prior drops out because it is constant in $$x_t$$.

<div markdown="1" style="overflow-x:auto">

$$
s_c
=\underbrace{s_u}_{\text{all noisy examples}}
+\underbrace{\nabla_{x_t}\log p_t(y\mid x_t)}_{\text{evidence for the requested class}}.
$$

</div>

Near the boundary between two equally likely digit classes, the first arrow may favor neither class. The second can still be large if a small change would make one label easier to distinguish. A class probability is a number. Its gradient tells us *which pixels to change* and in which direction.

### 3.2 Why a noise predictor gives a score

For one fixed clean point, differentiating the forward Gaussian gives

<div markdown="1" style="overflow-x:auto">

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=-\frac{x_t-a_t x_0}{\sigma_t^2}
=-\frac{\epsilon}{\sigma_t}.
$$

</div>

At sampling time $$x_0$$ is unknown. The **population** density averages the forward kernel over clean examples of the requested class.

<div markdown="1" style="overflow-x:auto">

$$
p_t(x_t\mid y)=\int q(x_t\mid x_0)\,p(x_0\mid y)\,dx_0.
$$

</div>

Differentiate this expression and divide by the density. Bayes' rule turns the weight inside the integral into the posterior density of each possible clean origin [3, 4].

<div markdown="1" style="overflow-x:auto">

$$
\begin{aligned}
s_c
&=\int p(x_0\mid x_t,y)\,\nabla_{x_t}\log q(x_t\mid x_0)\,dx_0\\
&=-\frac{\mathbb E[\epsilon\mid x_t,t,y]}{\sigma_t}
\ \approx\ -\frac{\epsilon_\theta(x_t,t,y)}{\sigma_t}.
\end{aligned}
$$

</div>

The conditional expectation is the population minimizer of the squared-error loss. The approximation enters when we use a fitted network. With the label omitted, the same argument gives the unconditional score. The minus sign matters below. Adding a vector to the score requires subtracting its noise-scaled counterpart from the predicted noise.

<img src="https://www.calvinyluo.com/assets/images/diffusion/score_3d.jpeg" alt="Three density peaks with colored sampling paths starting at the same point and heading toward different peaks." width="46%" style="max-width:46%;height:auto"> <img src="https://www.calvinyluo.com/assets/images/diffusion/score_sample.webp" alt="Top view of the mixture with local score arrows and the same stochastic paths drawn over them." width="46%" style="max-width:46%;height:auto">

*Figure 3. On the left, height represents density. On the right, each black arrow gives a local uphill direction. The colored paths are Langevin samples from the same starting point. Their different destinations illustrate how random steps can carry samples to different modes even when the score field and starting point are fixed.* [Luo, 2022, Fig. 6 (tutorial printout, p. 26)](https://arxiv.org/pdf/2208.11970#page=18) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## 4. Classifier Guidance

### 4.1 Training a classifier on noisy inputs

Suppose we already have an unconditional denoiser and want to reuse it to draw 3s. Bayes' identity tells us which term is missing. Dhariwal and Nichol [5] estimate that term with a separate classifier $$p_\phi(y\mid x_t,t)$$. An accurate classifier of clean digits could give meaningless gradients halfway through denoising, where the strokes are barely visible. Train it on the same noise levels the sampler will encounter.

Draw a labeled example, sample $$t$$ and $$\epsilon$$, and construct $$x_t=a_t x_0+\sigma_t\epsilon$$. The supervised target is the original label. If a noisy state carries almost no class information, a well-calibrated classifier should return probabilities close to the class priors there. Minimizing cross-entropy gives

<div markdown="1" style="overflow-x:auto">

$$
\mathcal L_{\rm cls}(\phi)
=\mathbb E_{(x_0,y),t,\epsilon}\big[-\log p_\phi(y\mid x_t,t)\big].
$$

</div>

### 4.2 Steering one reverse step

Hold the classifier weights fixed and differentiate the requested class's **log probability** with respect to the current noisy input. Call this vector $$h_t$$. With an unconditional base score estimate $$\hat s_u$$, the sampler uses

<div markdown="1" style="overflow-x:auto">

$$
h_t=\nabla_{x_t}\log p_\phi(y\mid x_t,t),\qquad
\tilde s=\hat s_u+\gamma h_t.
$$

</div>

Here $$\gamma$$ is the classifier guidance scale. At zero, the sampler uses the base model. At one, an exact unconditional score and exact classifier recover the conditional score. Larger values amplify the label evidence. The paper also guides an already class-conditional base model. In that setting, ordinary conditioning is already present at $$\gamma=0$$.

How does an arrow move a Gaussian transition? Let $$\mu_\theta$$ and $$\Sigma_t$$ be the base reverse mean and covariance. Over a small step, approximate the classifier's log probability by a linear function. Multiplying the Gaussian by the exponential of that linear term adds a tilt to its log density. Completing the square shifts the center. In the authors' sampling algorithm the gradient is evaluated at the current noisy state [5, Alg. 1].

<div markdown="1" style="overflow-x:auto">

$$
\tilde\mu_\theta=\mu_\theta+\gamma\Sigma_t h_t,\qquad
x_{t-1}\sim\mathcal N(\tilde\mu_\theta,\Sigma_t).
$$

</div>

The covariance determines how far a transition can move along the classifier gradient. For a sampler written in noise-prediction form, the score conversion in Section 3 gives $$\tilde\epsilon=\epsilon_\theta(x_t,t)-\gamma\sigma_t h_t$$ [5, Alg. 2]. These are two sampler formulations. Use the correction specified for the transition being taken. Applying both in one step would count the class evidence twice.

**Look at the samples.** Dhariwal and Nichol asked an unconditional model for a Pembroke Welsh corgi. At classifier scale 1, several samples show partial animals or backgrounds. At 10, pointed ears and short muzzles recur across the grid. Both panels use a classifier. [Dhariwal and Nichol, Fig. 2, p. 5](https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf#page=5)

## 5. Classifier-Free Guidance

### 5.1 Learning two predictions with one network

The identity in Section 3 can also be read backwards. If we already have $$s_c$$ and $$s_u$$, then their difference gives the class-evidence direction. Ho and Salimans [6] train one denoiser to answer both questions. Independently for each training example, they replace the real condition with a reserved null condition $$\varnothing$$ with probability $$p_{\rm drop}$$, where $$0<p_{\rm drop}<1$$. A null condition means the label was withheld. Any digit, including 3, can occur in this subset.

<div markdown="1" style="overflow-x:auto">

$$
\begin{aligned}
\tilde y&=\begin{cases}
\varnothing & \text{with probability }p_{\rm drop},\\
y & \text{with probability }1-p_{\rm drop},
\end{cases}\\
\mathcal L_{\rm CFG}
&=\mathbb E\big[\|\epsilon-\epsilon_\theta(x_t,t,\tilde y)\|_2^2\big].
\end{aligned}
$$

</div>

Only one condition choice is needed per training example. Because dropout is independent of the data, the null branch sees the whole data distribution. The labeled branch can use the same network weights to learn class-specific denoising.

### 5.2 Interpolating and extrapolating at sampling time

Evaluate the denoiser twice on the **same** noisy state at the **same** time. Write $$\epsilon_u=\epsilon_\theta(x_t,t,\varnothing)$$ and $$\epsilon_c=\epsilon_\theta(x_t,t,y)$$. The change caused by revealing the label is $$\epsilon_c-\epsilon_u$$. Starting at the unconditional prediction, take $$g$$ copies of that change.

<div markdown="1" style="overflow-x:auto">

$$
\tilde\epsilon=\epsilon_u+g(\epsilon_c-\epsilon_u).
$$

</div>

Use this prediction in the reverse mean from Section 2. At $$g=0$$ the prediction is unconditional. At $$g=1$$ it is the ordinary conditional prediction. Values between zero and one interpolate, while $$g>1$$ continues past the conditional prediction in the same direction. Two evaluations can share a batch, although both predictions still have to be computed for general $$g$$.

**Reading the original paper.** Ho and Salimans write $$(1+w)\epsilon_c-w\epsilon_u$$. Expanding our equation shows that $$g=1+w$$. Thus **“no extra guidance” means $$w=0$$ in the paper and $$g=1$$ here**. For example, $$g=3$$ means $$3\epsilon_c-2\epsilon_u$$, equivalent to $$w=2$$ [6, Eq. 6].

### 5.3 What replaces the classifier gradient?

Convert both predictions to scores using the same $$-\!1/\sigma_t$$ factor. The resulting direction is $$\tilde s=\hat s_u+g(\hat s_c-\hat s_u)$$. With exact scores, the difference is precisely the classifier gradient in Section 4. With learned networks it is an estimate. A learned vector field can even have nonzero circulation around a loop, in which case it cannot be the gradient of a single log probability [6, Sec. 3.2]. CFG needs only the two predictions to take its next step.

## 6. Illustrative Example: A Gaussian Mixture

Return to Figure 2. Let the equally likely classes have clean distributions $$\mathcal N(m_y,I)$$ in two dimensions, with $$m_0=(-2,0)$$ and $$m_1=(2,0)$$. For easy arithmetic take $$\beta_j=1/2$$ at every step and examine $$t=2$$. Then $$a_t=1/2$$. Forward noising moves each center to $$a_t m_y$$, while its covariance stays $$a_t^2I+\sigma_t^2I=I$$. The noisy clouds are centered at $$(-1,0)$$ and $$(1,0)$$.

The score of a unit-covariance Gaussian at a point $$x$$ is its center minus $$x$$. For the full mixture, average those score vectors using the **posterior** class probabilities at $$x$$.

<div markdown="1" style="overflow-x:auto">

$$
\begin{aligned}
s_c(x,t,y)&=a_t m_y-x,\\
s_u(x,t)&=\sum_{k=0}^1 p_t(y=k\mid x)\,(a_t m_k-x).
\end{aligned}
$$

</div>

At the midpoint $$(0,0)$$ the two class probabilities are equal. The unconditional arrows cancel. Asking for class 1 leaves the rightward arrow $$(1,0)$$. Now move one unit **above** the midpoint. Symmetry still gives equal class probabilities, but both clouds pull downward. This separates the part of the prediction the label changes from the part the two classes agree on.

<div markdown="1" style="overflow-x:auto">

| Noisy location | Unconditional, $$g=0$$ | Class 1, $$g=1$$ | Amplified, $$g=2$$ |
| --- | --- | --- | --- |
| $$(0,0)$$ | $$(0,0)$$ | $$(1,0)$$ → | $$(2,0)$$ → |
| $$(0,1)$$ | $$(0,-1)$$ ↓ | $$(1,-1)$$ ↘ | $$(2,-1)$$ ↘ |

</div>

In the second row, guidance doubles the horizontal component and leaves the vertical component at −1. Doubling the whole conditional score would give $$(2,-2)$$, a different direction. Each entry describes a local score at one noise level. A generated endpoint also depends on the other reverse steps and their noise.

## 7. Comparison and Guidance Strength

### 7.1 What has to run at each step?

<div markdown="1" style="overflow-x:auto">

| Method | Training requirement | Work per reverse step | Practical trade-off |
| --- | --- | --- | --- |
| Direct conditioning | Denoiser trained on labeled pairs | One conditional prediction | Preserves the ordinary conditional sampling rule. |
| Classifier guidance | Base denoiser and a classifier trained on noisy labeled data | One denoiser call, one classifier evaluation and an input gradient | Reuses a base model. Steering depends on the noisy classifier's gradients. |
| Classifier-free guidance | Denoiser trained with condition dropout | Usually two denoiser predictions, optionally batched | Supports labels or prompts through the same network. Extra denoiser work is needed for amplification. |

</div>

### 7.2 How much guidance?

At a fixed noise level, Bayes' identity gives the guided score a density interpretation.

<div markdown="1" style="overflow-x:auto">

$$
s_u+g(s_c-s_u)
=\nabla_{x_t}\log\!\left[p_t(x_t)\,p_t(y\mid x_t)^g\right].
$$

</div>

A larger exponent favors locations where the class is easy to recognize. In a digit model that can suppress an unusual but perfectly valid handwritten 3. This identity describes each noise level separately. It does not specify the final distribution produced by a finite sampler [6, 7].

<img src="https://sander.ai/images/panda1.jpg" alt="Unguided text-conditional GLIDE samples for a stained-glass panda prompt, including diffuse shapes and varied styles." width="46%" style="max-width:46%;height:auto"> <img src="https://sander.ai/images/panda3.jpg" alt="GLIDE samples for the same prompt at CFG scale three. Panda faces and strong stained-glass outlines appear repeatedly." width="46%" style="max-width:46%;height:auto">

*Figure 4. The prompt in both grids asks for a stained-glass window depicting a panda eating bamboo. On the left ($$g=1$$) the prompt is already present. On the right ($$g=3$$), look for the black outlines and the recurring black-and-white faces. These are separate sets of samples.* [Nichol et al., 2022, Fig. 17b,d](https://proceedings.mlr.press/v162/nichol22a.html) · [Panels as shown in Dieleman, 2022, printout p. 6](https://sander.ai/2022/05/26/guidance.html) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

There is a measurable trade-off behind the pictures. For their class-conditional 256 × 256 ImageNet model, Dhariwal and Nichol increased classifier scale from 1 to 10. Precision rose from **0.82 to 0.88**, while recall fell from **0.52 to 0.32** [5, Table 3]. In this evaluation, precision measures how often samples fall near the real-image distribution. Recall measures how much of that distribution the samples cover. A grid of convincing individual images can thus hide missing kinds of images.

What happens if the estimated direction is wrong? A misleading noisy classifier gradient gets multiplied by $$\gamma$$. CFG can amplify errors in the difference between its two predictions. Ho and Salimans show examples with saturated colors under strong guidance [6, Fig. 3]. The appropriate scale depends on the model, noise schedule, sampler and variation one wants to preserve. To judge it, examine several draws for the **same** condition as well as the fidelity of each draw [5–7].

## 8. Key Takeaways

1. **Conditioning** selects a distribution. A denoiser trained with a label can already draw varied examples of that class using the ordinary DDPM reverse step.
2. **Classifier guidance** obtains the class-evidence direction from a noisy classifier. **CFG** estimates the same direction by comparing two denoiser predictions at one noisy state.
3. **Guidance scale** multiplies the part of the prediction that changes with the condition. Stronger guidance can make a class easier to recognize while reducing the range of examples generated.

## 9. References

1. Wang, K. (2026). *Diffusion Models*, CSE/ISyE 6740. Forward corruption, noise prediction and sampling in [Lecture 16, slides 12–13 and 21–25](https://gt-koala.github.io/Gatech-CSE6740/assets/semesters/fall-2026/lectures/16/lecture-16-diffusion-models.pdf#page=12). Conditional denoiser in [Lecture 17, slide 18](https://gt-koala.github.io/Gatech-CSE6740/assets/semesters/fall-2026/lectures/17/lecture-17-diffusion-models.pdf#page=18).
2. Ho, J., Jain, A., and Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. See Eqs. 4, 11 and 14 and Algorithms 1–2. [Paper](https://arxiv.org/abs/2006.11239)
3. Luo, C. (2022). *Understanding Diffusion Models: A Unified Perspective*. See the noise and score interpretations, and the section on guidance. [Tutorial](https://www.calvinyluo.com/2022/08/26/diffusion-tutorial.html) · [Paper](https://arxiv.org/abs/2208.11970)
4. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR. See Sec. 3.3 on estimating scores and Sec. 5 on conditional sampling. [Paper](https://arxiv.org/abs/2011.13456)
5. Dhariwal, P., and Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS. See Sec. 4, Algorithms 1–2, Fig. 2 and Table 3. [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)
6. Ho, J., and Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. See Eq. 6 and Algorithms 1–2. Their own toy experiment uses three Gaussian classes. [Paper](https://arxiv.org/abs/2207.12598) · [Mixture illustration, Fig. 2](https://arxiv.org/pdf/2207.12598v1#page=2)
7. Dieleman, S. (2022). *Guidance: a cheat code for diffusion models*. See the visual comparisons and the May 29 correction on time-dependent densities. [Article](https://sander.ai/2022/05/26/guidance.html)
8. Nichol, A. Q., et al. (2022). *GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models*. ICML. Figure 17 compares samplers on the same text prompt. [Paper](https://proceedings.mlr.press/v162/nichol22a.html)
9. Weng, L. (2021). *What are Diffusion Models?* Additional illustrated reading on the DDPM forward and reverse processes. [Article](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

*AI assistance disclosure. An AI assistant helped draft and edit these notes, work through the examples, locate figure sources and check the mathematical formatting.*
