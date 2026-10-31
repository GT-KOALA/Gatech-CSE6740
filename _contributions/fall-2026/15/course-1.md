---
semester: fall-2026
lecture_number: "15"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: KyliinX
issue: 47
---
# Generative Models and Autoencoders (Lecture 15, Slides 1–17)

Up to this lecture, most models we learned took an input and produced a prediction: given an image, is it a cat or a dog? This lecture reverses the question. Instead of only recognizing patterns in existing data, can we learn the patterns well enough to produce new data? For example, can we generate images of cats based on the given label? In this lecture, we first connect this goal to density estimation, then introduce autoencoders through the idea of compression, and finally see why reconstruction alone is not enough for generation. The variational autoencoder (VAE) that fixes this starts at slide 17 and is covered in the next note section.

## 1. From prediction to generation (slides 1–6)

In supervised learning, a model maps an input $x$ to a label $y$. A generative model goes backward: it is trained on samples from a data distribution and then asked to produce a new sample from the same distribution (slide 4). The sample can be conditioned on a label or a prompt (e.g., "a cat"), or unconditioned.

Slides 5–6 show some applications: converting a photo to a Ghibli-style drawing, generating an illustration from a text prompt, and the commercial systems built on these ideas (Runway Gen-2, DALL·E, Imagen, Stability AI, Leonardo.Ai). The applications are the motivation, while the underlying question is statistical. What distribution could have produced the examples we observe, and how do we sample from it?

## 2. Density estimation already gives us a generator (slides 7–8)

We have met this question before, in the density estimation and Gaussian mixture model (GMM) lectures. Given a dataset $D=\{x^{(1)},\ldots,x^{(n)}\}$, we choose a parametric density $p(x\mid\theta)$ and fit $\theta$ by maximum likelihood:

$$
\theta^{*}=\arg\max_{\theta}\,\log P(D\mid\theta)=\arg\max_{\theta}\sum_{i=1}^{n}\log p(x^{(i)}\mid\theta).
$$

For a GMM, the density is a weighted sum of Gaussian components,

$$
p(x\mid\theta)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k),\qquad \sum_{k=1}^{K}\pi_k=1,\quad \pi_k\ge 0,
$$

where $\pi_k$ is the mixing proportion of component $k$ and $(\mu_k,\Sigma_k)$ are its mean and covariance.

Slide 7 makes the key observation: once $\theta$ is learned, the same model can be run in reverse. Sample a component $k$ with probability $\pi_k$, then sample $x\sim\mathcal{N}(\mu_k,\Sigma_k)$. Density estimation (data $\to$ $\theta$) and generation ($\theta$ $\to$ new $x$) are the two directions of one model.

So there are two separate ideas in "neural generative models":

1. **Reverse the density model.** Learn a distribution from samples, then draw new samples from it. This is not specific to neural networks; a GMM already does it.
2. **Replace the parametric family with a neural network.** For high-dimensional data such as images, a mixture of Gaussians is far too restrictive. Neural generative models replace the simple parametric density with a network. The cost is that we usually lose the closed-form density $p(x)$ (slide 7, "how to extend to high-dimensional data?"). Different methods make different trade-offs, but all keep the ability to sample.

Slide 8 gives the history. Before 2012, no effective generative models existed for images. Then came the variational autoencoder (VAE, 2013), the generative adversarial network (GAN, 2014), and diffusion models (2020), which are currently the mainstream choice for high-resolution image generation. Whether diffusion is the final answer is an open research question. Our entry point is the autoencoder.

## 3. Autoencoders: compress, then decompress (slides 9–10)

The basic idea of autoencoders is compression. Take an input, squeeze it into a short code, then decompress the code back into the input. An autoencoder learns both halves of this process with neural networks:

* The encoder $f_\phi$ maps the input $x\in\mathbb{R}^{d}$ to a latent code $z\in\mathbb{R}^{d'}$.
* The decoder $g_\theta$ maps $z$ back to a reconstruction $x'$ in the original input space.

$$
x \;\xrightarrow{\;f_\phi\;}\; z=f_\phi(x) \;\xrightarrow{\;g_\theta\;}\; x'=g_\theta(z).
$$

![Schematic of an autoencoder: the input passes through an encoder into a small latent code, and a decoder maps the code back to a reconstruction of the input.](https://commons.wikimedia.org/wiki/Special:FilePath/Autoencoder_schema.png)

*Autoencoder schematic by Michela Massi, Wikimedia Commons, CC BY-SA 4.0.*

The latent layer is a bottleneck, with $d'\ll d$ (slide 10). Without a bottleneck, the task is trivial: the network can just copy its input, and we learn nothing. Forcing the code to be small means the encoder has to keep only the information the decoder needs most.

Training compares $x'$ with $x$. With squared error, the reconstruction loss is

$$
\mathcal{L}_{\mathrm{rec}}(\phi,\theta)=\frac{1}{n}\sum_{i=1}^{n}\bigl\|x^{(i)}-g_\theta\bigl(f_\phi(x^{(i)})\bigr)\bigr\|_2^{2},
$$

and $\phi,\theta$ are learned jointly by minimizing it. A perfect copy is not expected; some information is lost in the bottleneck, and the network learns which information to keep.

## 4. PCA is a linear autoencoder; deep autoencoders are the nonlinear version (slides 11–13)

The process, compress then decompress, looks familiar from lecture 3. PCA finds the directions of largest variance and represents each point by its coordinates along the principal components. Slide 11 takes this as an autoencoder with one linear layer on each side and no activation function. For data with mean $\mu$, principal directions $w^1,\ldots,w^{d'}$ and eigenvalues $\lambda_1,\ldots,\lambda_{d'}$, the encoder is

$$
z^{i}_j=\frac{w^{j\top}(x^{i}-\mu)}{\sqrt{\lambda_j}},\qquad j=1,\ldots,d',
$$

and the decoder reverses the scaling and projects back:

$$
x^{i\prime}=\mu+\sum_{j=1}^{d'}z^{i}_j\,\sqrt{\lambda_j}\,w^{j}.
$$

In the slide's notation, $z=W(x-\mu)$ and $x'=W'^{\top}z+\mu$. So PCA is exactly the same as the idea of encoding to smaller data and decoding, built from linear maps. Conversely, a linear autoencoder trained with squared error recovers the same principal subspace as PCA, although its internal coordinates need not be the PCA coordinates (Baldi and Hornik, 1989).

PCA's limitation is that it can only describe the data within a linear subspace, while images vary in nonlinear ways. Using PCA as an autoencoder may not yield satisfying results. A deep autoencoder stacks multiple layers with nonlinear activations in both encoder and decoder, so the learned compression can follow this nonlinear structure. Slide 13 states the two cases side by side:

* shallow one-layer autoencoder $=$ principal component analysis,
* deep autoencoder $=$ nonlinear dimensionality reduction.

The figure on slide 13 (from Hinton and Salakhutdinov, 2006) shows handwritten digits reconstructed from a 30-dimensional code. The 30-D PCA reconstructions are blurry, and some digits are hard to tell apart; the 30-D deep autoencoder reconstructions are nearly indistinguishable from the originals.

Slide 12 asks the practical question: what networks should the encoder and decoder be? For a plain vector, fully connected layers are enough. For images, the natural encoder is a stack of convolutional layers (as in the CNN lecture), which shrink the spatial size and then flatten to a code. The decoder has to do the opposite, growing a code back into an image, which needs a layer type that increases spatial size.

## 5. Deconvolution (slides 14–15)

Slides 14–15 introduce the transposed convolution, often called "deconvolution" in deep-learning materials, as a learnable upsampling layer. Slide 14 shows a $3\times3$ learnable filter applied with stride 2 and padding 1 (applied on the top and left only) to a $2\times2$ input: each input value scales a copy of the filter, the copies are placed in the output at stride-2 offsets and summed where they overlap, giving a $4\times4$ output. Slide 15 shows how padding and stride change the output size, mirroring the padding/stride animation from the convolution lecture.

For the autoencoder, this is simply the tool that makes a convolutional decoder possible. It is not the inverse of the encoder's convolution in any exact sense; it is a separate learned layer whose output happens to be larger than its input. The instructor noted this is a side detail for our purposes; the autoencoder ideas do not depend on it.

## 6. Why a good reconstruction is not yet a generator (slides 13 and 16)

So far the autoencoder reconstructs inputs it is given. Our original goal was different: produce a new image without starting from one.

After training, we can encode every dataset image to a code $z=f_\phi(x)$ and decode it back. To generate something new, we would pick a fresh $z$ and decode it. But nothing in the training objective tells us which $z$ to pick. The autoencoder learned a reconstruction map, not a probability distribution over codes. Slide 13 lists this as the first limitation: autoencoders are not generative models because they do not define a distribution. (The second limitation on the slide is practical: there is no principled rule for choosing the latent dimension $d'$.)

Slide 16 makes the problem concrete by perturbing the code:

$$
z=f_\phi(x),\qquad \tilde z=z+\epsilon,\qquad x'=g_\theta(\tilde z).
$$

The decoder now receives $\tilde z$ instead of $z$. The decoder would amplify the noise and make the output meaningless. This situation is similar to flipping a few bytes in a zip file and trying to unzip it. A corrupted compressed representation usually does not decompress to a slightly corrupted file.

**In-class discussion.** Students were asked how to make the representation robust to this corruption. Two directions came up:

1. Detect the corruption, as in signal processing or file integrity checks (checksums, error-detecting codes), and refuse or repair.
2. Inject noise into the code during training, so the decoder learns to reconstruct from perturbed codes. This is closely related to denoising autoencoders (which corrupt the input instead), and in the latent-space form it is what the VAE does.

The second direction is the one the course follows. Compression gives us useful representations and depth gives us a nonlinear reconstruction map, but as long as the decoder only ever sees exact codes $f_\phi(x)$, it cannot be trusted on any other code, and generation needs exactly that.

## 7. Transition to variational autoencoders (slide 17)

Slide 16 is the real motivation for what comes next. A plain autoencoder compresses and decompresses well on the training data, but the decoding is fragile: perturb the code and the output is unrecoverable. That is exactly why it is not usable as a generative model, because generating means feeding the decoder a code it has never seen.

The variational autoencoder is the fix. Instead of encoding $x$ to a single point $z$, it encodes $x$ to a distribution over codes and samples $z$ from it during training. The VAE formulation begins on slide 18 and is covered in the next course-material section.

## References

* Kai Wang, CSE/ISyE 6740 Computational Data Analysis: Autoencoder and Variational Autoencoder, Georgia Institute of Technology, Oct 21, 2026, slides 1–17. Primary source for this section.
* Kai Wang, CSE/ISyE 6740: Dimensionality Reduction, Aug 31, 2026. PCA encoding and reconstruction used in Section 4.
* Kai Wang, CSE/ISyE 6740: Density Estimation, Sep 2, 2026, and Gaussian Mixture Models, Sep 9, 2026. Likelihood, density estimation, and GMMs used in Section 2.
* Kai Wang, CSE/ISyE 6740: Convolutional Neural Networks, Oct 19, 2026. Convolution, padding, and stride are referenced in Sections 4–5.
* P. Baldi and K. Hornik, "Neural networks and principal component analysis: Learning from examples without local minima," Neural Networks 2(1), 1989. Linear autoencoders and the PCA subspace.
* G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of data with neural networks," Science 313(5786), 2006. Source of the digit-reconstruction figure on slide 13.
* V. Dumoulin and F. Visin, "A guide to convolution arithmetic for deep learning," arXiv:1603.07285, 2016. Transposed convolution output sizes.
* D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," ICLR 2014 (arXiv 2013); I. Goodfellow et al., "Generative Adversarial Nets," NeurIPS 2014; J. Ho, A. Jain and P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020. The three milestones on the slide 8 timeline.
* Autoencoder schematic: Michela Massi, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Autoencoder_schema.png), CC BY-SA 4.0.

*AI assistance: Claude (Anthropic) was used to check this draft against the lecture slides and to fix Markdown/MathJax formatting; the content was written and verified by the author.*
