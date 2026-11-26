---
semester: fall-2026
lecture_number: "19"
slot: optional-1
slot_order: 3
role: optional_topic
role_label: Optional topic
tab_title: Extension I
assignee: jsheng45
issue: 73
---
# Supplementary Notes: Vision Transformers (ViT) and CLIP

Lecture 19 — Transformers

## From text tokens to images

The self-attention mechanism discussed in Lecture 19 lets each token combine information from other tokens in a sequence:

$$
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Here, $$Q$$, $$K$$, and $$V$$ are learned projections of the token representations, and $$d_k$$ is the dimension of each key vector. Multi-head attention, a position-wise MLP, residual connections, and layer normalization form a Transformer encoder block.

This structure does not require the tokens to represent words. Vision Transformer (ViT) applies it to image patches. CLIP addresses a different question: how can we train image and text representations so that matching images and descriptions are close together? The two ideas fit together because a ViT can serve as CLIP's image encoder.

## Vision Transformer: treating an image as a sequence

### Patch tokens and positional embeddings

In the original ViT [1], an image $$I\in\mathbb{R}^{H\times W\times C}$$ is divided into non-overlapping patches of size $$P\times P$$. Assuming $$H$$ and $$W$$ are divisible by $$P$$, the number of patches is

$$
N=\frac{H}{P}\frac{W}{P}.
$$

Each patch is flattened into a vector $$x_i\in\mathbb{R}^{P^2C}$$ and mapped to a token embedding with a shared learned linear projection:

$$
e_i=E x_i,\qquad E\in\mathbb{R}^{d\times P^2C}.
$$

For example, a $$224\times224$$ RGB image split into $$16\times16$$ patches produces $$14\times14=196$$ patch tokens. Each patch contains $$16\cdot16\cdot3=768$$ values before projection. The embedding dimension $$d$$ is a model choice and need not equal 768.

A learned classification token, usually written as `[CLS]`, is placed before the patch tokens. Each token also receives a learned positional embedding:

$$
z_{0,i}=
\begin{cases}
x_{\mathrm{cls}}+p_0, & i=0,\\
e_i+p_i, & i=1,\ldots,N.
\end{cases}
$$

Positional information matters because self-attention alone does not distinguish the order of the patches. The original ViT uses learned one-dimensional embeddings for the positions in the flattened sequence, rather than explicit two-dimensional coordinate embeddings. These positions correspond to locations in the image grid, but their spatial relationships must be learned [1].

### Processing the sequence

Let $$Z_0$$ denote the full sequence of $$N+1$$ tokens. ViT passes this sequence through Transformer encoder blocks. Using the pre-normalization arrangement in the original paper, block $$\ell$$ computes

$$
\begin{aligned}
Z'_\ell &= Z_{\ell-1}
+\mathrm{MSA}\!\left(\mathrm{LN}(Z_{\ell-1})\right),\\
Z_\ell &= Z'_\ell
+\mathrm{MLP}\!\left(\mathrm{LN}(Z'_\ell)\right),
\qquad \ell=1,\ldots,L.
\end{aligned}
$$

Here, MSA is multi-head self-attention and LN is layer normalization. The MLP is applied separately to each token, while attention mixes information across tokens. After the final block and layer normalization, the `[CLS]` representation is passed to a classification head [1].

The connection to Lecture 19 is direct: the attention operation is unchanged; the main difference is how the input tokens are constructed.

### How ViT differs from a CNN

A CNN builds in assumptions about images through local convolutions and shared filters. Stacking convolutional layers lets information travel across progressively larger regions. In a standard ViT, each patch can attend to every other patch from the first attention layer, so distant regions can interact directly.

ViT therefore has fewer image-specific inductive biases than a CNN, although patch extraction still introduces local structure. This flexibility is a strength at scale, but it also affects training. In the original experiments, ViTs trained from scratch on smaller datasets without strong regularization performed worse than comparable ResNets; large-scale pre-training substantially improved their results [1]. This does not mean that a pre-trained ViT always needs a large downstream dataset.

Full attention also involves pairwise interactions among tokens. Its attention computation scales quadratically with token count, so using smaller patches or a higher image resolution can become expensive. The computational cost relative to a CNN depends on the model and input resolution.

## CLIP: learning to match images and text

### Two encoders, one embedding space

CLIP, short for Contrastive Language–Image Pre-training, learns from paired images and text descriptions [2]. It uses separate image and text encoders, followed by learned linear projections into a shared embedding space. Their outputs are L2-normalized to give image vectors $$v$$ and text vectors $$t$$ in $$\mathbb{R}^D$$.

The original CLIP experiments use either a modified ResNet or a ViT as the image encoder. The text encoder is a Transformer with causal, or masked, self-attention: a token can attend to itself and preceding tokens, but not later tokens. Its final `[EOS]` token representation is normalized and projected to form the text embedding [2].

Because the projected vectors are normalized, their dot product equals cosine similarity:

$$
\mathrm{sim}(I,T)=v^\top t.
$$

The encoders process the two modalities separately; they interact through these similarity scores during training.

### Contrastive training

Consider a batch of $$B$$ image–text pairs $$\{(I_i,T_i)\}_{i=1}^B$$. Define the scaled similarity score

$$
s_{ij}=\frac{v_i^\top t_j}{\tau},
$$

where $$\tau>0$$ is a learned temperature, implemented through a learned logit scale. The diagonal entries correspond to the observed pairs. For each image, the model tries to identify its paired text among the texts in the batch:

$$
\mathcal{L}_{I\to T}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp(s_{ii})}{\sum_{j=1}^{B}\exp(s_{ij})}.
$$

CLIP also trains in the reverse direction, asking each text to identify its paired image:

$$
\mathcal{L}_{T\to I}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp(s_{ii})}{\sum_{j=1}^{B}\exp(s_{ji})}.
$$

The full objective averages the two directions [2]:

$$
\mathcal{L}_{\mathrm{CLIP}}
=\frac{1}{2}\left(\mathcal{L}_{I\to T}+\mathcal{L}_{T\to I}\right).
$$

Intuitively, the model learns to score the observed image–text pairs above the other combinations in the batch. Those other combinations serve as negatives, although some may also be semantically related.

### Zero-shot classification with prompts

After pre-training, CLIP can classify an image by comparing it with text descriptions of candidate classes. For each class, we write a prompt such as `a photo of a banana` and compute its text embedding. The predicted class is the one whose prompt has the highest similarity to the image:

$$
\hat c=\arg\max_{c\in\mathcal C}v(I)^\top t(T_c).
$$

For example, suppose the candidate classes are apple, banana, and orange. An image of a banana might produce the following scores:

| Text prompt | Illustrative cosine similarity |
| --- | ---: |
| `a photo of an apple` | 0.21 |
| `a photo of a banana` | 0.34 |
| `a photo of an orange` | 0.18 |

The predicted class is banana because its prompt has the largest score. These numbers are hypothetical, not measured model outputs or class probabilities.

“Zero-shot” means that no labeled examples from the target classification task are used to train or fine-tune the classifier in this procedure. CLIP still requires image–text supervision during pre-training. Prompt wording also matters: different descriptions of the same class can produce different predictions [2].

### Strengths and limitations

CLIP makes it possible to specify candidate classes in natural language and to compare images with text for retrieval. However, its predictions depend on the candidate descriptions and on what it learned during pre-training. The original paper reports weaknesses on tasks such as counting objects and some specialized recognition problems, and discusses biases inherited from internet data [2]. A high similarity score should therefore not be treated as proof that the model understands every detail of an image.

## Putting ViT and CLIP together

ViT specifies an image-processing architecture; CLIP specifies a way to train image and text encoders together. A ViT can be trained with ordinary supervised image labels, and CLIP can use an image encoder other than ViT. When combined, the ViT supplies the image representation and CLIP's objective aligns that representation with text.

Both extend the Transformer ideas from Lecture 19: ViT changes the tokens that self-attention processes, while CLIP changes the supervision used to learn useful representations across modalities.

## References

1. Dosovitskiy, A., et al. (2020 preprint; ICLR 2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929). See Section 3.1 for the architecture and Section 4.3 for the data-scale experiments.
2. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. [Paper and publication details](https://proceedings.mlr.press/v139/radford21a.html). See Sections 2.2–2.3 for the objective and encoders, Section 3.1 for zero-shot transfer, and Sections 4–5 for limitations and broader impacts.

## AI assistance

ChatGPT assisted with wording revisions and technical checks of this draft. The numerical example above was constructed for illustration and was not obtained by running CLIP.

