---
semester: fall-2026
lecture_number: "15"
slot: course-2
slot_order: 2
role: course_material
role_label: Course materials
tab_title: Core explanation II
assignee: yousertin
issue: 48
---
# Variational Autoencoders

These notes cover slides 17–26 of Lecture 15: the VAE model, its loss and derivation, the PyTorch implementation, and applications (generation, interpolation, conditional VAE). Slides 1–16 (generative models, autoencoders, PCA) are in Core explanation I.

## Notation

- $$x_i \in \mathbb{R}^{d}$$: the $$i$$-th input; $$z_i \in \mathbb{R}^{d'}$$: its latent vector, with $$d' \ll d$$
- $$\phi$$, $$\theta$$: encoder and decoder parameters
- $$\mu_\phi(x_i)$$, $$\sigma^2_\phi(x_i)$$: mean and variance vectors predicted by the encoder
- $$q(z \mid x_i, \phi)$$: encoder distribution; the slides shorten it to $$q(z)$$
- $$p(z) = \mathcal{N}(0, I)$$: target (prior) latent distribution; the slides write $$\mathcal{N}(0,1)$$
- $$G_\theta(z)$$: decoder output, the mean of $$p(x \mid z, \theta)$$
- $$\eta$$: variance of the decoder's Gaussian noise
- $$y$$: context information (e.g., a class label) in the conditional VAE

## From Autoencoder to VAE

An autoencoder maps each input to a single code, so adding noise to a code can produce a meaningless output (slide 16). A VAE instead encodes each input into a **distribution** (slide 18):

- **Autoencoder:** $$x^i \mapsto z^i$$
- **VAE:** $$x^i \mapsto \big(\mu_\phi(x^i),\ \sigma^2_\phi(x^i)\big)$$

$$
z_i \sim q(z_i \mid x_i, \phi) = \mathcal{N}\big(\mu_\phi(x_i),\ \sigma^2_\phi(x_i)\big), \qquad x_i \sim p(x_i \mid z_i, \theta).
$$

- **Encoder:** a probabilistic model that predicts the mean and variance. It learns the distribution of the latent space given the observation.
- **Decoder:** a probabilistic model that predicts the mean $$G_\theta(z_i)$$. It learns the generative process given the latent vector.

For a $$d'$$-dimensional latent vector, the Gaussian has diagonal covariance: each coordinate has its own mean and variance.

**Intuition:** each input now occupies a small cloud of codes instead of a single point. The decoder is trained on samples from the whole cloud, so nearby codes decode to similar outputs.

## The VAE Loss

$$
L = \sum_i \frac{1}{2\eta}\big\lVert x_i - G_\theta(z_i)\big\rVert^2 + D_{\mathrm{KL}}\big(q(z) \,\Vert\, p(z)\big)
$$

This is the slide's shorthand. Written out, both terms are inside the sum over training examples:

$$
L = \sum_i \Big[ \frac{1}{2\eta}\big\lVert x_i - G_\theta(z_i)\big\rVert^2 + D_{\mathrm{KL}}\big(q(z \mid x_i, \phi) \,\Vert\, p(z)\big) \Big].
$$

- **Reconstruction loss:** measures how much the reconstruction $$x' = G_\theta(z_i)$$ differs from $$x_i$$, as in an ordinary autoencoder.
- **KL divergence loss:** measures how far the encoder's distribution $$\mathcal{N}\big(\mu_\phi(x_i), \sigma^2_\phi(x_i)\big)$$ is from the target $$\mathcal{N}(0, I)$$. Since $$q$$ depends on $$x_i$$, there is one KL term per example.

The KL divergence is defined as

$$
D_{\mathrm{KL}}(q \,\Vert\, p) = \mathbb{E}_{z \sim q}\left[\log \frac{q(z)}{p(z)}\right].
$$

It is $$\ge 0$$, equals $$0$$ only when $$q = p$$, and is not symmetric.

The two terms compete. On its own, reconstruction is easiest when codes are distinct and have small variance. KL pulls every code distribution toward $$\mathcal{N}(0, I)$$. Intuitively, the result is codes that still carry information about the input but stay concentrated around the origin, with overlapping clouds and fewer large empty regions than an ordinary autoencoder.

## Deriving the Loss (slides 19–21)

### Why the decoder must be noisy

Training a generator by maximum likelihood means maximizing

$$
p(x) = \int p(z)\, p(x \mid z)\, dz .
$$

If $$z$$ is low-dimensional and the decoder is deterministic, its outputs lie on a $$d'$$-dimensional surface in $$\mathbb{R}^d$$, so $$p(x) = 0$$ almost everywhere. (In class: a 2-D sheet in 3-D space almost never contains a random point.) The fix is a noisy observation:

$$
p(x \mid z) = \mathcal{N}\big(x;\ G_\theta(z),\ \eta I\big), \qquad \text{mean } G_\theta(z),\ \text{covariance } \eta I .
$$

### Lower bound by Jensen's inequality

As in the EM derivation, multiply and divide by any distribution $$q(z)$$, then use the concavity of $$\log$$:

$$
\begin{aligned}
\log p(x) &= \log \int q(z)\, \frac{p(z)}{q(z)}\, p(x \mid z)\, dz \\
&\ge \int q(z) \log\left(\frac{p(z)}{q(z)}\, p(x \mid z)\right) dz \\
&= \underbrace{\mathbb{E}_q\left[\log \frac{p(z)}{q(z)}\right]}_{=\,-D_{\mathrm{KL}}(q \,\Vert\, p)} + \underbrace{\mathbb{E}_q\big[\log p(x \mid z)\big]}_{\text{reconstruction}} .
\end{aligned}
$$

We choose $$p = \mathcal{N}(0, I)$$ and $$q = \mathcal{N}\big(\mu_\phi(x_i), \sigma^2_\phi(x_i)\big)$$, and maximize the bound over $$\phi$$ and $$\theta$$. Flipping the sign gives the loss $$L$$.

This bound is usually called the evidence lower bound (ELBO). It is tight when $$q$$ equals the true posterior $$p(z \mid x)$$ (Kingma & Welling, 2014).

### Reconstruction term

Taking the log of the Gaussian density $$\frac{1}{(2\pi\eta)^{d/2}} \exp\left(-\frac{\lVert x - G_\theta(z)\rVert^2}{2\eta}\right)$$ gives

$$
\mathbb{E}_q\big[\log p(x \mid z)\big] = -\frac{1}{2\eta}\, \mathbb{E}_q\big\lVert x - G_\theta(z)\big\rVert^2 - \frac{d}{2}\log(2\pi\eta).
$$

The slide drops the constant $$-\frac{d}{2}\log(2\pi\eta)$$, which does not affect training. In practice, the expectation is estimated with a sampled $$z$$. So, for fixed $$\eta$$, minimizing the squared error is equivalent to maximizing the Gaussian decoder's log-likelihood: the two differ only by the factor $$-\frac{1}{2\eta}$$ and an additive constant.

### KL term

For one latent coordinate, write $$\mu$$ and $$\sigma^2$$ for the encoder's mean and variance. With $$p = \mathcal{N}(0,1)$$ and $$q = \mathcal{N}(\mu, \sigma^2)$$:

$$
\log \frac{p(z)}{q(z)} = \frac{1}{2}\log\sigma^2 - \frac{z^2}{2} + \frac{(z-\mu)^2}{2\sigma^2}.
$$

Using $$\mathbb{E}_q\big[(z-\mu)^2\big] = \sigma^2$$ and $$\mathbb{E}_q\big[z^2\big] = \mu^2 + \sigma^2$$:

$$
\mathbb{E}_q\left[\log \frac{p(z)}{q(z)}\right] = \frac{1}{2}\left(\log\sigma^2 - \mu^2 - \sigma^2 + 1\right).
$$

Coordinates are independent, so the KL penalty sums over them:

$$
D_{\mathrm{KL}} = -\frac{1}{2}\sum_{j=1}^{d'}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right).
$$

For example, $$(\mu, \sigma^2) = (0, 1)$$ gives $$0$$; $$(1, 1)$$ gives $$0.5$$; $$(0, 0.25)$$ gives about $$0.32$$. The $$\mu^2$$ term pulls codes toward the origin, and the penalty grows without bound as $$\sigma^2 \to 0$$, so the encoder cannot collapse back to a single-point code.

## PyTorch Implementation (slides 20–23)

The code is from [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE).

**Loss** (slides 20–21):

```python
recons_loss = F.mse_loss(recons, input)
kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
```

`kld_loss` is the formula above: the sum over `dim = 1` runs over latent coordinates, and the mean over `dim = 0` averages over the batch. In the repository the terms are combined as `recons_loss + kld_weight * kld_loss`, where `kld_weight` sets the relative weight of the KL term. It is not the same quantity as $$\eta$$ (the decoder noise variance), but it has a similar effect: in the formula, a larger $$\eta$$ also gives the KL term relatively more weight.

**Encoder** (slide 23): convolutional layers extract features, the features are flattened, and then two fully connected heads predict the Gaussian parameters.

```python
result = self.encoder(input)
result = torch.flatten(result, start_dim=1)
mu = self.fc_mu(result)        # mu_phi(x)
log_var = self.fc_var(result)  # log sigma^2_phi(x)
```

The encoder predicts the log variance because the KL formula uses $$\log\sigma^2$$ directly, and exponentiating `log_var` always gives a positive variance.

**Decoder** (slide 23): a linear layer maps $$z$$ to $$512 \times 2 \times 2$$ values, which are reshaped into feature maps and upsampled by transpose convolutions. The variance is assumed to be 1 ($$\eta = 1$$), so only the mean network is needed.

```python
result = self.decoder_input(z)
result = result.view(-1, 512, 2, 2)
result = self.decoder(result)
result = self.final_layer(result)
```

**Sampling:** `reparameterize(mu, log_var)` (visible on slide 26) computes $$z = \mu + \sigma \odot \varepsilon$$ with $$\varepsilon \sim \mathcal{N}(0, I)$$. The randomness sits in $$\varepsilon$$, so gradients can flow back to the encoder.

## Applications (slides 24–26)

### Generation

After training, keep only the decoder:

1. Sample $$z \sim \mathcal{N}(0, I)$$.
2. Decode: $$x' \sim p(x \mid z, \theta)$$.

For example, a VAE trained on cat images generates new cats this way. Intuitively, this works because the KL term keeps the training codes close to $$\mathcal{N}(0, I)$$, so random prior samples tend to fall in regions the decoder saw during training. This is not a guarantee, and samples can still be blurry or unrealistic.

### Latent interpolation

Given two codes $$z$$ and $$z'$$, decode the points along the line between them:

$$
z_t = (1-t)\,z + t\,z', \qquad t \in [0, 1].
$$

The outputs change gradually from one endpoint to the other (slide 25, sketch drawings from Ha & Eck, 2018). The interpolation happens **in latent space**, not in pixel space; averaging pixels would just overlay the two images.

### Conditional VAE

A plain VAE cannot choose what it generates. If we know extra information $$y$$, such as the class label "cat", we add it to **both** the encoder and the decoder:

$$
z_i \sim q(z_i \mid x_i, y, \phi), \qquad x_i \sim p(x_i \mid z_i, y, \theta).
$$

Training is the same as for a vanilla VAE. In the code on slide 26, the label is embedded as an extra image channel and concatenated with the input before encoding (`torch.cat([embedded_input, embedded_class], dim = 1)`). It is also concatenated with $$z$$ before decoding (`torch.cat([z, y], dim = 1)`).

To generate, choose $$y$$, sample $$z$$, and decode both. Since $$y$$ carries the class, $$z$$ can capture style instead. This enables applications such as style transfer and causal inference (Kingma et al., 2014).

## Classroom Dialog

- **How can we protect a latent code from noise?** One idea from compression is to add checks for corrupted data. The VAE's idea is to make the encoding itself a distribution, so the model learns to handle perturbed codes.
- **A neural network is deterministic, so how can the encoder be stochastic?** Let the network predict the *parameters* of a distribution (a mean and a variance), then sample from that distribution.
- **Why $$D_{\mathrm{KL}}(q \,\Vert\, p)$$ and not $$D_{\mathrm{KL}}(p \,\Vert\, q)$$?** The two directions are not the same, because the order changes both the ratio and the distribution the expectation is taken over. This direction comes out of the likelihood derivation. It is also practical: we can sample from $$q$$ (the encoder) to compute the expectation.

## References

- K. Wang, [*Autoencoder and Variational Autoencoder*](/Gatech-CSE6740/assets/semesters/fall-2026/lectures/15/lecture-15-autoencoders-and-variational-autoencoders.pdf), CSE 6740 Computational Data Analysis, Lecture 15, Georgia Tech, October 21, 2026, slides 17–26, and the in-class explanations. Slide numbers in these notes are the numbers printed on the slides.
- D. P. Kingma and M. Welling, ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114), ICLR 2014.
- D. Ha and D. Eck, ["A Neural Representation of Sketch Drawings"](https://arxiv.org/abs/1704.03477), ICLR 2018.
- D. P. Kingma, D. J. Rezende, S. Mohamed, and M. Welling, ["Semi-Supervised Learning with Deep Generative Models"](https://arxiv.org/abs/1406.5298), NeurIPS 2014.
- AntixK, [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) ([`vanilla_vae.py`](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py), [`cvae.py`](https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py)).

*AI assistance: Used Claude to convert my tex notes to md and edit them. The equations, derivations, and code were checked against the slides and the cited sources.*
