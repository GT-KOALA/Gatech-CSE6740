---
semester: fall-2026
lecture_number: "16-diffusion"
slot: course-2
slot_order: 2
role: course_material
role_label: Course materials
tab_title: Core explanation II
assignee: Duckbluee
issue: 51
---

## Denoising and DDPM: From Reverse Transitions to Noise Prediction

These notes cover Lecture 16, slides 16–27 [1]. They continue from the forward diffusion process and explain how a learned reverse process generates data. Denoising Diffusion Probabilistic Models (DDPM) implement this idea through a network that predicts noise. The main connection is between three descriptions of the same task: estimating a reverse transition, predicting its mean, and predicting the noise in a corrupted sample.

### 1. Setup and notation

We only need a short recap of the forward process. Let $$x_0\in\mathbb R^d$$ be a training example, such as an image represented as a vector, and let $$I$$ denote the identity matrix of the matching dimension. A fixed noise schedule specifies $$\beta_1,\ldots,\beta_T$$, with $$0<\beta_t<1$$. Define

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,
\qquad
\bar\alpha_0=1.
$$

The forward transition and its closed-form marginal are

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(x_t;\sqrt{\alpha_t}\,x_{t-1},\beta_t I\right),
$$

$$
q(x_t\mid x_0)
=\mathcal N\!\left(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\right).
$$

Consequently, we can construct a sample at any chosen timestep directly:

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

This direct construction will make training efficient: we do not have to simulate every earlier forward step to obtain $$x_t$$.

| Symbol | Meaning |
|---|---|
| $$x_0$$ | A clean data sample; during generation, the final generated sample |
| $$x_t$$ | The state at noise level $$t$$ |
| $$T$$ | Number of diffusion steps |
| $$\beta_t$$ | Variance of the noise added in one forward step |
| $$\alpha_t,\bar\alpha_t$$ | One-step and cumulative signal-retention factors |
| $$\epsilon$$ | Standard Gaussian noise used in the direct construction of $$x_t$$ |
| $$\epsilon_\theta(x_t,t)$$ | Neural network prediction of that noise |
| $$\mu_\theta(x_t,t)$$ | Mean of a learned reverse transition |
| $$\sigma_t^2$$ | Variance chosen for a reverse transition |
| $$\theta$$ | Parameters shared by the denoising network across timesteps |

The coefficients $$\sqrt{\bar\alpha_t}$$ and $$\sqrt{1-\bar\alpha_t}$$ multiply the signal and noise, respectively. Their squares determine the corresponding variance factors. In particular, $$\beta_t$$ is a variance, whereas the standard deviation of the one-step forward noise is $$\sqrt{\beta_t}$$.

### 2. Learning the reverse process

#### 2.1 Starting from noise

The schedule is chosen so that the signal remaining at the final timestep is very small. Then $$q(x_T\mid x_0)$$ is approximately $$\mathcal N(0,I)$$ for typical data samples. This motivates the generation procedure in slides 17–19:

1. Draw $$x_T\sim\mathcal N(0,I)$$.
2. Repeatedly sample an earlier state from a reverse transition.
3. After the last step, use $$x_0$$ as the generated sample.

The learned joint distribution is

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),
\qquad
p(x_T)=\mathcal N(0,I).
$$

The product describes a sequence of local transitions. Learning these transitions allows the model to transform a simple starting distribution into a complex data distribution.

#### 2.2 Why the reverse direction needs a model

The forward rule is specified by us, but the true reverse conditional $$q(x_{t-1}\mid x_t)$$ depends on the unknown data distribution. A noisy state can be compatible with several cleaner states, so reversing the process involves uncertainty.

Following slide 18, we model each reverse transition with a Gaussian:

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N\!\left(
x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I
\right).
$$

The mean specifies the center of the proposed earlier state. The variance determines the remaining randomness around that center. Small forward steps motivate this Gaussian approximation; the true reverse conditional is not generally an exact Gaussian at an arbitrary finite step.

The timestep is an essential input. The same observed value can require different denoising behavior at different noise levels. A single network is used across timesteps, with $$t$$ identifying which denoising problem it is solving.

#### 2.3 A tractable posterior during training

There is a useful distinction between the reverse conditional needed for generation and a conditional available during training. When the clean sample $$x_0$$ is known, the posterior has a closed form. For $$2\le t\le T$$,

$$
q(x_{t-1}\mid x_t,x_0)
=\mathcal N\!\left(
x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I
\right),
$$

where

$$
\tilde\mu_t(x_t,x_0)
=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
$$

$$
\tilde\beta_t
=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.
$$

These expressions follow by multiplying the forward Gaussian transition by the Gaussian distribution of $$x_{t-1}$$ given $$x_0$$ and completing the square. Appendix A gives the calculation.

Knowing this posterior does not solve generation directly: a new generated sample has no known clean $$x_0$$ to condition on. In fact,

$$
q(x_{t-1}\mid x_t)
=\int q(x_{t-1}\mid x_t,x_0)\,q(x_0\mid x_t)\,dx_0.
$$

Even though the conditional inside the integral is Gaussian, averaging over possible clean samples need not give a Gaussian. The posterior conditioned on $$x_0$$ provides a tractable training construction; the learned model must operate using $$x_t$$ and $$t$$ alone [2, Section 2].

### 3. From predicting a mean to predicting noise

Slide 21 changes the network's output from a mean estimate to a noise estimate. The connection can be made explicit.

From the direct forward construction,

$$
x_0
=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon}
{\sqrt{\bar\alpha_t}}.
$$

Substituting this expression into the posterior mean and using $$\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$$ gives

$$
\tilde\mu_t(x_t,x_0)
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon
\right).
$$

The model therefore predicts $$\epsilon_\theta(x_t,t)$$ and uses the corresponding reverse mean

$$
\mu_\theta(x_t,t)
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,
\epsilon_\theta(x_t,t)
\right).
$$

Once the noise estimate is available, this formula determines the mean. Mean prediction and noise prediction are two parameterizations of the reverse model, rather than two separate networks that must both be trained [2, Section 3.2].

For interpretation, the noise estimate also gives a clean-sample estimate:

$$
\hat x_0(x_t,t)
=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

For example, consider a scalar with $$x_0=1.5$$, $$\bar\alpha_t=0.64$$, and sampled noise $$\epsilon=-0.5$$. Then

$$
x_t=0.8(1.5)+0.6(-0.5)=0.9.
$$

If the predicted noise equals that sampled noise, the clean-sample estimate is

$$
\hat x_0=\frac{0.9-0.6(-0.5)}{0.8}=1.5.
$$

This example illustrates the algebra. A learned prediction will generally be imperfect, and $$\hat x_0$$ is an estimate. DDPM sampling uses the reverse transition to move from $$x_t$$ to $$x_{t-1}$$; it does not replace the whole sampling procedure with this one clean-sample estimate.

The label $$\epsilon$$ in these equations is the effective noise in the direct construction from $$x_0$$ to $$x_t$$. It is generally different from the fresh noise introduced by only the last forward step from $$x_{t-1}$$ to $$x_t$$.


### 4. The DDPM training objective

#### 4.1 A known target at a randomly selected noise level

During training, we have access to clean examples. We independently sample a training example $$x_0$$, a timestep $$t\sim\mathrm{Uniform}(\{1,\ldots,T\})$$, and noise $$\epsilon\sim\mathcal N(0,I)$$. We then construct $$x_t$$ using the forward closed form.

Because we sampled $$\epsilon$$ ourselves, it is an available supervision target. The simplified DDPM objective in slides 22–23 can be written as

$$
L_{\mathrm{simple}}(\theta)
=\mathbb E_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-
\epsilon_\theta\!\left(
\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
t
\right)
\right\|_2^2
\right].
$$

The expectation averages over examples, noise levels, and Gaussian corruptions. The squared norm compares the predicted and sampled noise across all coordinates. A minibatch estimates this expectation; using a mean over coordinates instead of a sum changes the loss by a fixed dimension-dependent factor.

This form uses a single expectation over the randomly selected timestep. Equivalently, it is the average of the timestep-specific expected losses. Summing those losses instead of averaging multiplies the objective by $$T$$. We should choose one convention and apply it consistently.

The supervision target is noise, while the learned capability is denoising. The noisy image contains information about both the clean-data distribution and its corruption. Learning to estimate the corruption therefore provides the mean needed by the reverse process.

#### 4.2 Connection to fitting the reverse transition

The noise-prediction loss is connected to the variational training objective of a diffusion model [2, Sections 2–3.4]. For $$t\ge2$$, one term compares the known training posterior with the model transition:

$$
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\,\Vert\,
p_\theta(x_{t-1}\mid x_t)
\right).
$$

With fixed reverse variance $$\sigma_t^2 I$$, the part depending on the model parameters is a squared mean error:

$$
D_{\mathrm{KL}}
=\frac{1}{2\sigma_t^2}
\left\|
\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)
\right\|_2^2+C_t,
$$

where $$C_t$$ does not depend on $$\theta$$. Substituting the noise parameterization gives

$$
D_{\mathrm{KL}}
=\frac{\beta_t^2}
{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2+C_t.
$$

Thus, fitting the reverse mean leads to a weighted noise-prediction error. The practical objective $$L_{\mathrm{simple}}$$ uses equal weights for uniformly sampled timesteps rather than retaining these timestep-dependent weights. It is a reweighted training objective, not a statement that unweighted noise MSE is exactly the data negative log likelihood. The full variational bound also includes the terminal-prior and final reconstruction terms.

This connection explains why a local noise-prediction task can train a model used in a multistep generative process.

### 5. Training the denoiser

Slides 23–24 give the following procedure. The operations on schedule coefficients are broadcast over each sample's coordinates.

```text
repeat until training stops:
    x0 = sample a minibatch of clean training examples
    t = sample one integer in {1, ..., T} for each example
    epsilon = sample independent standard Gaussian noise, shaped like x0

    xt = sqrt(alpha_bar[t]) * x0
         + sqrt(1 - alpha_bar[t]) * epsilon

    epsilon_pred = epsilon_theta(xt, t)
    loss = mean_squared_error(epsilon_pred, epsilon)
    update theta using the gradient of loss
```

Across updates, random timestep selection trains the same network on many noise levels. Sampling a new timestep on each update does not mean the training procedure must visit timesteps in chronological order.

The gradient passes through the denoising network used for the sampled timestep. The forward corruption is fixed and does not contain the model parameters. Standard DDPM training with this objective therefore does not require unfolding all $$T$$ reverse sampling steps and backpropagating through the entire generated chain.

The training illustration in slide 24 can be read from left to right: select a clean image, corrupt it at a sampled noise level, predict the sampled noise, and update the model using their squared difference.

### 6. Generating samples

After training, the model parameters are held fixed. We start from fresh Gaussian noise and apply the reverse transition repeatedly.

At step $$t$$, first calculate the mean

$$
\mu_\theta(x_t,t)
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,
\epsilon_\theta(x_t,t)
\right).
$$

Then sample

$$
x_{t-1}
=\mu_\theta(x_t,t)+\sigma_t z,
$$

where $$z\sim\mathcal N(0,I)$$ is freshly drawn for a stochastic reverse step. This $$z$$ represents the remaining uncertainty in the reverse transition. It is different from the training target $$\epsilon$$.

For the fixed-variance DDPM formulation, common choices for $$t>1$$ are $$\sigma_t^2=\beta_t$$ or $$\sigma_t^2=\tilde\beta_t$$ [2, Section 3.2]. The choice is part of the sampling specification; the posterior variance conditioned on a known clean image should not automatically be identified with the variance of every learned reverse model.

The sampler shown in slide 23 sets $$z=0$$ at $$t=1$$. Its last output is therefore the predicted mean without another noise draw.

```text
x = sample standard Gaussian noise with the desired sample shape

for t = T, T-1, ..., 1:
    epsilon_pred = epsilon_theta(x, t)
    mean = (x - beta[t] / sqrt(1 - alpha_bar[t]) * epsilon_pred)
           / sqrt(alpha[t])

    if t > 1:
        z = sample fresh standard Gaussian noise, shaped like x
        x = mean + sigma[t] * z
    else:
        x = mean

return x
```

The schedule, indexing, input scaling, and noise parameterization must be consistent with training. The network is reused at every step; we do not train a different network for each timestep.

Slide 25 illustrates this iteration. One model call proposes an earlier state, and repeating the procedure produces the final image. The coefficient multiplying the predicted noise and the rescaling by $$1/\sqrt{\alpha_t}$$ are both necessary: the update is not simply the current image minus the raw network output.

| Aspect | Training | Sampling |
|---|---|---|
| Starting information | A clean training example is available | Start with $$x_T\sim\mathcal N(0,I)$$ |
| Timestep choice | Random timestep for each example | Descend from $$T$$ to $$1$$ |
| Network task | Estimate a known sampled noise target | Estimate noise to obtain a reverse-transition mean |
| Parameter updates | Update $$\theta$$ by gradient descent | Keep $$\theta$$ fixed |
| Network evaluations | One denoiser evaluation per example per loss estimate | Usually $$T$$ evaluations per generated sample in this sampler |
| Role of randomness | Creates training corruptions and targets | Creates the starting state and intermediate stochastic transitions |

The sampling goal is to produce a new sample from the learned data distribution. It does not require recovering a particular training image from its original forward trajectory.


### 7. Reading the classroom examples and demonstrations

Slides 19 and 26 use MNIST and ImageNet to illustrate image generation. The dataset details below describe the classroom demonstrations [1, slide 26], rather than a requirement that all diffusion models use these image sizes.

| Classroom example | Data described in the slide | What it helps illustrate |
|---|---|---|
| MNIST | 60,000 training images, $$28\times28$$ pixels, 10 digit classes | Learning image structure on a comparatively simple dataset |
| ImageNet | About 1.28 million training images, 1,000 classes; images resized to $$32\times32\times3$$ for this demonstration | Applying the same denoising formulation to varied color images |

Both cases use a network that predicts image-shaped noise. The mathematical task is the same even though the image distribution, dimensionality, and training difficulty differ.

#### 7.1 Interpreting the training curves

Slide 26 plots noise-prediction MSE, including raw training history and a trailing mean. The initial decrease shows that the network becomes better at its sampled denoising tasks. The later-training view makes smaller changes easier to see.

Individual losses fluctuate because the examples, timesteps, and noise draws vary. A smoothed curve makes the trend clearer, but its exact level also depends on input scaling, loss reduction, the schedule, and the data. Loss values from the two datasets should therefore not be read as a direct ranking of generative quality.

The curves are training losses. They are not classification accuracies, and the number of classes in a dataset does not by itself turn the denoiser into a classifier.

#### 7.2 Conditional and unconditional generation

The demonstration also includes curves labeled unconditional and conditional. The algorithms above describe an unconditional model: no requested class is supplied.

For class-conditional generation, a class label $$y$$ can be supplied as additional information to the noise network:

$$
\epsilon_\theta(x_t,t,y).
$$

The corresponding model is $$p_\theta(x_{t-1}\mid x_t,y)$$. Training still predicts the sampled Gaussian noise, while the label provides information about the desired data distribution. At generation time, supplying a label allows the denoising process to be conditioned on that class. This is a short extension needed to interpret the demonstration; the reverse-process and noise-prediction relationships remain the same.

#### 7.3 Interpreting the generated samples

A useful way to inspect generation is to follow the whole trajectory: an initial noisy state, intermediate states with emerging structure, and the final sample. A static noisy frame alone does not describe the final result.

Noise-prediction loss and generated samples provide different evidence. A decreasing loss indicates improvement on the training objective; final samples must also be inspected for plausible structure and variety.

There is also a distinction between the states used in training and sampling. Training constructs $$x_t$$ from real examples through the known forward process. Sampling uses states produced by the learned reverse chain. Imperfect transition estimates can therefore affect later steps, which is one reason to examine the complete sampler rather than judge it solely from a local training loss.

### Appendix A. Deriving the Gaussian training posterior

This derivation supplies the algebra behind Section 2.3. It applies for $$2\le t\le T$$. Write $$z=x_{t-1}$$ and $$y=x_t$$. Conditional on $$x_0$$, the two relevant Gaussian factors are

$$
q(z\mid x_0)=\mathcal N(z;m,vI),
\qquad
m=\sqrt{\bar\alpha_{t-1}}\,x_0,
\qquad
v=1-\bar\alpha_{t-1},
$$

$$
q(y\mid z)=\mathcal N(y;\sqrt{\alpha_t}\,z,\beta_t I).
$$

The forward Markov property and Bayes' rule give

$$
q(z\mid y,x_0)
\propto
q(y\mid z)\,q(z\mid x_0).
$$

Terms independent of $$z$$ can be absorbed into a constant. Expanding the log of this product gives

$$
\begin{aligned}
\log q(z\mid y,x_0)
&=C-\frac{1}{2v}\|z-m\|_2^2
-\frac{1}{2\beta_t}\|y-\sqrt{\alpha_t}\,z\|_2^2\\
&=C'-\frac12\left(A\|z\|_2^2-2b^\top z\right),
\end{aligned}
$$

with

$$
A=\frac1v+\frac{\alpha_t}{\beta_t},
\qquad
b=\frac{m}{v}+\frac{\sqrt{\alpha_t}}{\beta_t}\,y.
$$

Completing the square,

$$
A\|z\|_2^2-2b^\top z
=A\left\|z-\frac{b}{A}\right\|_2^2
-\frac{\|b\|_2^2}{A}.
$$

The final term does not depend on $$z$$. Comparing with a Gaussian exponent, the posterior covariance is $$A^{-1}I$$ and its mean is $$b/A$$.

The variance simplifies as follows:

$$
A^{-1}
=\left(\frac1v+\frac{\alpha_t}{\beta_t}\right)^{-1}
=\frac{v\beta_t}{\beta_t+\alpha_t v}.
$$

Since

$$
\beta_t+\alpha_t v
=(1-\alpha_t)+\alpha_t(1-\bar\alpha_{t-1})
=1-\bar\alpha_t,
$$

we obtain

$$
A^{-1}
=\frac{(1-\bar\alpha_{t-1})\beta_t}
{1-\bar\alpha_t}
=\tilde\beta_t.
$$

Similarly,

$$
\frac{b}{A}
=\frac{\beta_t m+\sqrt{\alpha_t}\,v\,y}
{\beta_t+\alpha_t v}
=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\,x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t.
$$

This is $$\tilde\mu_t(x_t,x_0)$$.

To see the final connection to noise prediction, substitute
$$
x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}
$$.

The coefficient of $$x_t$$ becomes

$$
\frac{\beta_t+\alpha_t(1-\bar\alpha_{t-1})}
{\sqrt{\alpha_t}(1-\bar\alpha_t)}
=\frac1{\sqrt{\alpha_t}},
$$

and the coefficient of $$\epsilon$$ becomes

$$
-\frac{\sqrt{\bar\alpha_{t-1}}\beta_t\sqrt{1-\bar\alpha_t}}
{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}}
=-\frac{\beta_t}
{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}.
$$

Together, these give the noise-parameterized posterior mean used in Section 3. The derivation conditions on a known clean sample; the learned reverse model replaces the unavailable noise by its network prediction.

### References

1. Kai Wang. *CSE/ISyE 6740: Diffusion Models*, October 26, 2026, slides 16–27. The forward-process notation is briefly recalled from the earlier slides. [Lecture slides](https://gt-koala.github.io/Gatech-CSE6740/assets/semesters/fall-2026/lectures/16/lecture-16-diffusion-models.pdf).
2. Jonathan Ho, Ajay Jain, and Pieter Abbeel. *Denoising Diffusion Probabilistic Models*, 2020. See Section 2, Sections 3.2 and 3.4, and Algorithms 1–2. [Paper](https://arxiv.org/abs/2006.11239).

### AI assistance

ChatGPT was used to help organize and draft the explanations and to assist with derivation checks. The mathematical expressions and algorithm steps were cross-checked against the cited lecture slides and DDPM paper. The scalar example and explanatory prose were prepared for these notes; no new model-training results are reported.
