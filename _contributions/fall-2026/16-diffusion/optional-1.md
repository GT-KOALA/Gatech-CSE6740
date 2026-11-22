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
> **Work in progress.** This is an initial scaffold for the Lecture 16 optional topic on conditional diffusion and guidance. Sections marked *TODO* are not yet written. AI assistance was used to prepare this outline; all content will be reviewed, verified, and revised before the final submission.

## Conditional diffusion models

An unconditional diffusion model learns to generate samples from a data distribution $$p(\mathbf{x})$$ by learning to reverse a gradual noising process (Sohl-Dickstein et al., 2015; Ho et al., 2020). In many applications we instead want samples that agree with a condition $$y$$, such as a class label or a text prompt. A *conditional* diffusion model targets the conditional distribution $$p(\mathbf{x} \mid y)$$.

*Guidance* methods add a tunable strength parameter that trades sample diversity for closer agreement with the condition. These notes cover the two most widely used approaches: classifier guidance (Dhariwal & Nichol, 2021) and classifier-free guidance (Ho & Salimans, 2022).

## The conditioning objective

Let $$p_t$$ denote the marginal density of the noisy sample $$\mathbf{x}_t$$ at noise level $$t$$. In the score-based view of diffusion models (Song et al., 2021), sampling runs a reverse-time process driven by the score $$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$. For conditional generation, the reverse process needs the conditional score $$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid y)$$ instead.

By Bayes' rule, $$p_t(\mathbf{x}_t \mid y) = p_t(y \mid \mathbf{x}_t)\, p_t(\mathbf{x}_t) / p_t(y)$$. Because $$p_t(y)$$ does not depend on $$\mathbf{x}_t$$, its gradient vanishes and

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid y) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p_t(y \mid \mathbf{x}_t).
$$

In words: the conditional score is the unconditional score plus the gradient of the log-likelihood of the condition given the *noisy* sample. Both guidance methods start from this decomposition. Classifier guidance estimates the second term with a separately trained classifier; classifier-free guidance avoids an explicit classifier.

## Classifier guidance

*TODO.* Planned content:

- Training a classifier $$p_\phi(y \mid \mathbf{x}_t, t)$$ on noisy samples.
- The guided score and the guidance scale.
- Training and sampling requirements.
- How the guidance scale affects sample fidelity and diversity.

## Classifier-free guidance

*TODO.* Planned content:

- Training one network for both conditional and unconditional prediction by randomly dropping the condition during training.
- The guided prediction as a combination of conditional and unconditional predictions.
- Training and sampling requirements.
- How the guidance weight affects sample fidelity and diversity.

## Comparison

*TODO.* Planned outline:

- What must be trained, and whether an existing conditional model can be reused.
- Network evaluations per sampling step.
- Which kinds of conditions each method supports.
- Fidelity–diversity trade-off as the guidance strength increases.
- Limitations and failure modes.
- Intuition through a small illustrative example.

## References

1. J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." *ICML*, 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
2. J. Ho, A. Jain, and P. Abbeel. "Denoising Diffusion Probabilistic Models." *NeurIPS*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
3. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*, 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
4. P. Dhariwal and A. Nichol. "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS*, 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
5. J. Ho and T. Salimans. "Classifier-Free Diffusion Guidance." *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*; extended version on arXiv, 2022. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
