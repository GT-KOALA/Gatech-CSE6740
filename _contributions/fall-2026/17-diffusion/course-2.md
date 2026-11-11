---
semester: fall-2026
lecture_number: "17-diffusion"
slot: course-2
slot_order: 2
role: course_material
role_label: Course materials
tab_title: Core explanation II
assignee: 33Geniusss
issue: 54
---
## Diffusion model architecture, computation, and applications

These notes cover [Lecture 17](https://gt-koala.github.io/Gatech-CSE6740/assets/semesters/fall-2026/lectures/17/lecture-17-diffusion-models.pdf), PDF pages 14–28: how to build the noise predictor, what it costs to run, and where we can use it. Page references use PDF pages; after page 23, printed slide numbers are one larger.

### 1. Connection to a hierarchical VAE (pages 14–15)

A VAE learns an encoder that maps data to a latent distribution and a decoder that maps latents back to data. A hierarchical VAE has several latent levels. We can think of the diffusion states $$x_1,\ldots,x_T$$ as these levels. [Kingma and Welling](https://arxiv.org/abs/1312.6114)

The forward chain acts like an encoder and the reverse chain like a decoder, but:

- The forward process is fixed.
- In pixel-space diffusion, noisy states keep the image's dimensions.
- The variational loss separates across timesteps, so we can sample timesteps during training.

This explains the slide's connection to a weighted ELBO. The usual noise MSE changes the timestep weights, so it is not exactly the original negative ELBO. [Ho et al., Section 3.4](https://arxiv.org/abs/2006.11239)

### 2. Choosing the noise-prediction network (page 16)

The network predicts one noise value per input component. If $$x_t$$ has shape $$H\times W\times C$$, then $$\epsilon_\theta(x_t,t)$$ must have the same shape.

A fully connected network can flatten the image and reshape its output. This gets expensive: for a $$256\times256\times3$$ image, one dense layer with 1,024 outputs needs about 201 million weights.

A **U-Net** keeps the spatial structure. Downsampling builds features at different scales, and upsampling restores resolution. Skip connections bring back earlier features to help recover detail. The lecture uses Conv2D and ConvTranspose2D layers. The original [U-Net](https://arxiv.org/abs/1505.04597) predicts segmentation labels; here it predicts noise.

The U-Net's “encoder” is learned. The **fixed encoder** mentioned above means the forward noising process.

### 3. Time and context inputs (pages 17–18)

Recall the noising formula, where $$\beta_t$$ is the forward noise variance, $$\alpha_t=1-\beta_t$$, and $$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$$:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

For unit-variance data, the signal-to-noise ratio is

$$
\operatorname{SNR}(t)
=\frac{(\sqrt{\bar\alpha_t})^2}{(\sqrt{1-\bar\alpha_t})^2}
=\frac{\bar\alpha_t}{1-\bar\alpha_t}.
$$

The timestep tells the network how much noise to expect. The target $$\epsilon$$ stays standard Gaussian, but its scale in $$x_t$$ changes.

Page 17 passes a time embedding through fully connected layers into U-Net blocks. DDPM uses sinusoidal embeddings, so one shared network can handle different timesteps. [Ho et al., Appendix B](https://arxiv.org/abs/2006.11239)

For conditional generation, we add a class or context $$y$$:

$$
\widehat\epsilon=\epsilon_\theta(x_t,t,y).
$$

For example, we can feed in a class embedding along with the time embedding. The model then learns $$p_\theta(x_0\mid y)$$ from observation-context pairs. At sampling time, we supply the desired context. The output is still image-shaped noise.

### 4. Training versus sampling cost (pages 19–23)

**Training.** Sample one timestep per example and form the noisy input directly. Let $$B$$ be the batch size, $$d$$ the data dimension, and $$T$$ the number of diffusion steps. Forming the batch costs $$O(Bd)$$. Adding one forward and backward pass gives, ignoring optimizer overhead,

$$
C_{\mathrm{train}}
\approx C_{\mathrm{fwd}}(B)+C_{\mathrm{bwd}}(B)+O(Bd).
$$

Here $$C_{\mathrm{fwd}}(B)$$ and $$C_{\mathrm{bwd}}(B)$$ are the batched network costs. We do not unroll all $$T$$ steps or store their activations. The schedule table may still take $$O(T)$$ space. This describes one update, not the total time needed to train the model.

**Sampling.** Start from $$x_T\sim\mathcal N(0,I)$$ and apply the DDPM update:

$$
x_{t-1}
=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)\right)+\sigma_t z.
$$

Here $$\sigma_t$$ is the reverse standard deviation. Use $$z\sim\mathcal N(0,I)$$ for $$t>1$$ and $$z=0$$ at the last step. Each output feeds into the next step, so we need $$T$$ sequential network calls:

$$
C_{\mathrm{sample}}
\approx\sum_{t=1}^{T}\left(C_{\mathrm{fwd}}(1)+O(d)\right)
=T\,C_{\mathrm{fwd}}(1)+O(Td).
$$

Parameters stay fixed, so there is no backward pass. Without saving the trajectory, we only keep the current state and temporary activations. Sampling time grows with $$T$$, but working memory need not. Faster samplers can use fewer calls.

| Operation | Network work | Use of time |
| --- | --- | --- |
| One training update | One batched forward and backward pass | Sample timesteps |
| One generated sample | A sequence of forward passes | Follow the reverse chain |
{: style="width: 100%; border-spacing: 12px 8px;"}

Sampling cost is per generated sample, not per optimizer update.

**Classroom demo.** The M3 Pro MacBook Pro has a 12-core CPU, an 18-core GPU, and 36 GB memory. The demo uses $$256\times256\times3$$ images, three Conv2D and three ConvTranspose2D layers, and 200 denoising steps. Page 23 shows noise turning into an image in 56.5 seconds. This is the slide's timing; we did not rerun the demo.

### 5. Applications as density estimation (pages 24–26)

The generated faces and image grid are examples of learning $$p(x)$$, or $$p(x\mid y)$$ when context is given. Like Gaussians and GMMs, VAEs and diffusion models estimate data distributions. They use neural networks to represent complex structure, though exact likelihoods can be hard to compute.

The lecture lists several other application areas. Here are possible ways to set up these problems, not experimental results from the slides:

| Area | What we could generate | What we could condition on |
| --- | --- | --- |
| Traffic or energy | Future traffic or demand | Recent observations and calendar information |
| Climate and weather | Future weather sequences | Historical measurements |
| Audio | A waveform or acoustic representation | Text or an observed signal |
| EEG, AIOps, or finance | Missing sequence values | Observed values and a missing-value mask |
| Recommendation | Candidate item representations | User history |
{: style="width: 100%; border-spacing: 12px 8px;"}

For traffic forecasting, the same history can lead to several plausible futures. Sampling gives us these alternatives, but we still need held-out data to check their accuracy and uncertainty. Each domain needs a suitable representation and noise process. See the [CVPR 2022 tutorial](https://cvpr2022-tutorial-diffusion-models.github.io/) for more examples.

### 6. VAE versus diffusion (page 27)

For the conventional VAE and pixel-space DDPM discussed in class:

| Aspect | VAE | DDPM |
| --- | --- | --- |
| Inference | Learned approximate posterior | Fixed forward noising process |
| Latents | Often a compact code useful as an embedding | Image-sized noisy states |
| Generation | Sample a latent and run the decoder | Run successive denoising steps |
| Sampling cost | Usually lower for comparable model scale | Higher because of repeated network calls |
{: style="width: 100%; border-spacing: 12px 8px;"}

The slides describe diffusion as more stable and higher-resolution than VAEs. This is a general comparison, not a rule: results depend on the model, data, and training setup. The main tradeoff here is a compact learned encoding and fast decoding versus repeated denoising.

### References and further reading (page 28)

1. Kai Wang. [Lecture 17: Diffusion Models](https://gt-koala.github.io/Gatech-CSE6740/assets/semesters/fall-2026/lectures/17/lecture-17-diffusion-models.pdf). CSE/ISyE 6740, October 28, 2026, PDF pages 14–28.
2. Jonathan Ho, Ajay Jain, and Pieter Abbeel. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS, 2020. Algorithms 1–2, Section 3.4, and Appendix B.
3. Diederik P. Kingma and Max Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). ICLR, 2014.
4. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI, 2015.
5. Arash Vahdat, Karsten Kreis, and Ruiqi Gao. [Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://cvpr2022-tutorial-diffusion-models.github.io/). CVPR tutorial, 2022; also credited for the diagrams on lecture pages 17–18.
6. Lilian Weng. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). 2021, with later updates. Further reading listed in the lecture.
7. Stanford CS231n. [2023 Lecture 15 slides](https://cs231n.stanford.edu/slides/2023/lecture_15.pdf). Further reading listed in the lecture.

**AI assistance disclosure:** OpenAI Codex helped check the logic and grammar in these notes, and check equations and references.
