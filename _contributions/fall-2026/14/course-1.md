---
semester: fall-2026
lecture_number: "14"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: YoEv
issue: 41
---

## CNN Foundations: From Pixels to Learned Features

### 1. Why use convolutional neural networks for images?

Image classification maps an image to a category. The ImageNet example in the lecture illustrates the scale of this problem: a model must recognize objects across many classes despite variations in position, appearance, and background.

A major milestone was the convolutional network introduced by Krizhevsky, Sutskever, and Hinton in 2012. Its five convolutional layers, followed by fully connected layers and a softmax classifier, demonstrated the effectiveness of learning image features directly from data. Some convolutional layers were followed by max pooling. [Reference: original paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).

To understand why this architecture is useful, we first compare a fully connected layer with a convolutional layer.

### 2. A fully connected layer treats the image as a vector

Consider the lecture's RGB image with shape $32 \times 32 \times 3$. The first two dimensions describe its spatial size; the third contains its red, green, and blue channels.

Flattening rearranges these values into a vector:

$$
x \in \mathbb{R}^{3072},
\qquad
3072 = 32 \cdot 32 \cdot 3.
$$

A fully connected layer with 10 outputs computes

$$
z = Wx + b,
\qquad
W \in \mathbb{R}^{10 \times 3072},
\qquad
b \in \mathbb{R}^{10}.
$$

Each output is a weighted sum over the entire image:

$$
z_i = \sum_{j=1}^{3072} W_{ij}x_j+b_i.
$$

An activation function may then produce $a_i=\sigma(z_i)$. This layer has $30{,}720$ weights and 10 biases.

Flattening does not discard pixel values, but a generic fully connected layer does not explicitly exploit their two-dimensional neighborhood structure. It also assigns separate weights to different image positions. For example, recognizing the same edge in two locations requires learning separate connections.

Convolution introduces two useful constraints:

- **Local connectivity:** each output depends on a small spatial neighborhood.
- **Weight sharing:** the same filter is applied at different positions.

Together, these let a network search for a local pattern throughout an image without learning a separate detector for every location.

### 3. One filter produces one feature map

For the $32 \times 32 \times 3$ input, the lecture uses a $5 \times 5 \times 3$ filter. It covers a small spatial patch but spans all three input channels.

At one position, the filter and the corresponding image patch each contain 75 values. Their dot product, plus a bias, produces one scalar:

$$
z = w^\top x_{\text{patch}} + b.
$$

Moving the filter across the image repeats this calculation with the **same weights and bias**. Arranging the results by position produces a feature map: a spatial record of where that filter responds strongly.

More formally, let $X$ be the padded input, indexed from zero. For stride $S$ and output channel $k$,

$$
Z_{i,j,k}
=
b_k+
\sum_{u=0}^{K-1}
\sum_{v=0}^{K-1}
\sum_{c=0}^{C_{\mathrm{in}}-1}
W_{u,v,c,k}\,X_{iS+u,jS+v,c}.
$$

Here, $K$ is the spatial filter size and $C_{\mathrm{in}}$ is the number of input channels. If followed by ReLU, the output activation is

$$
A_{i,j,k} = \max(0,Z_{i,j,k}).
$$

A terminology detail: deep-learning libraries commonly call this operation "convolution," although they implement cross-correlation, without reversing the filter. [Reference: PyTorch `Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

### 4. Padding and stride determine the spatial output size

**Stride** specifies how far the filter moves between neighboring output positions. A larger stride samples fewer positions.

**Padding** adds a border around the input, commonly filled with zeros. This allows the filter to operate near image boundaries and can help control spatial shrinkage.

For a square input of width $N$, a square filter of width $K$, symmetric padding $P$, stride $S$, and no dilation,

$$
N_{\mathrm{out}}
=
\left\lfloor
\frac{N+2P-K}{S}
\right\rfloor+1.
$$

The same calculation applies independently to height and width.

For the lecture's $32 \times 32$ input and $5 \times 5$ filter:

| Padding $P$ | Stride $S$ | Output height and width |
|---|---:|---:|
| 0 | 1 | $28 \times 28$ |
| 1 | 1 | $30 \times 30$ |
| 0 | 2 | $14 \times 14$ |
| 1 | 2 | $15 \times 15$ |

With no padding and stride 1, there are $32-5+1=28$ valid positions along each axis. Therefore, **one** filter produces a $28 \times 28 \times 1$ output.

Padding 1 does not preserve the input size for this filter. A $5 \times 5$ filter needs padding 2 to preserve spatial size when stride is 1.

### 5. Multiple filters produce multiple channels

One filter detects one learned pattern. A convolutional layer usually contains several independently learned filters.

In the lecture example, six $5 \times 5 \times 3$ filters produce six feature maps. Stacking them along the channel dimension gives

$$
32 \times 32 \times 3
\quad\longrightarrow\quad
28 \times 28 \times 6.
$$

The output depth 6 is the **number of filters**, not the number of layers. Each filter combines information across all three input channels and produces one output channel.

The next layer receives six input channels. To produce 12 output channels with $5 \times 5$ filters, it therefore uses 12 filters of shape $5 \times 5 \times 6$:

$$
28 \times 28 \times 6
\quad\longrightarrow\quad
24 \times 24 \times 12,
$$

again assuming stride 1 and no padding.

After the first layer, channels represent learned features rather than RGB colors.

### 6. Counting trainable parameters

For a standard convolutional layer with no channel grouping, the number of weights is

$$
K_h K_w C_{\mathrm{in}} C_{\mathrm{out}}.
$$

With one bias per output channel, the total number of trainable parameters is

$$
(K_h K_w C_{\mathrm{in}}+1)C_{\mathrm{out}}.
$$

Applying this to the lecture:

| Layer | Weights | Biases | Total |
|---|---:|---:|---:|
| $3 \rightarrow 6$ channels, $5 \times 5$ filters | $5\cdot5\cdot3\cdot6=450$ | 6 | 456 |
| $6 \rightarrow 12$ channels, $5 \times 5$ filters | $5\cdot5\cdot6\cdot12=1800$ | 12 | 1812 |

The slide's counts of 450 and 1800 include weights only.

Weight sharing explains why we do **not** multiply these counts by the number of spatial output positions. Producing more output values requires more computation, but it does not introduce a new filter at each position.

ReLU adds no trainable parameters.

### 7. Pooling summarizes nearby responses

Pooling reduces spatial resolution by summarizing a neighborhood within each channel.

For example, consider a $2 \times 2$ region:

$$
\begin{bmatrix}
1 & 3\\
2 & 4
\end{bmatrix}.
$$

**Max pooling** returns the largest value, $4$. It retains the strongest response in that region.

**Average pooling** returns the mean, $2.5$. It summarizes the overall response.

The pooling diagram on slide 15 contrasts these two operations. Both reduce each local window to one value, but they preserve different information. Standard pooling operates separately on each channel, so it changes spatial dimensions while preserving the channel count.

A $2 \times 2$ pooling window with stride 2 and no padding transforms

$$
28 \times 28 \times 6
\quad\longrightarrow\quad
14 \times 14 \times 6.
$$

The pooling operation has no trainable weights. Its window size and stride are chosen hyperparameters. [Reference: PyTorch `MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).

Pooling can make a representation less sensitive to small changes in feature position, but it also discards spatial detail. It does not guarantee translation invariance.

### 8. Following the lecture's convolution-pooling pipeline

The dimensions in slide 16 can be explained by two convolution-ReLU-pooling blocks:

| Operation | Output shape |
|---|---|
| Input | $32 \times 32 \times 3$ |
| $5 \times 5$ convolution, 6 filters, stride 1, no padding | $28 \times 28 \times 6$ |
| ReLU | $28 \times 28 \times 6$ |
| $2 \times 2$ max pooling, stride 2 | $14 \times 14 \times 6$ |
| $5 \times 5$ convolution, 12 filters, stride 1, no padding | $10 \times 10 \times 12$ |
| ReLU | $10 \times 10 \times 12$ |
| $2 \times 2$ max pooling, stride 2 | $5 \times 5 \times 12$ |

This distinguishes the example from the earlier stack without pooling, whose spatial sizes decrease from 32 to 28 to 24.

Slide 17 shows selected feature maps from successive layers for a cat image. These maps are responses of individual channels, rather than ordinary grayscale reconstructions of the input. A dark region indicates a weak response in the displayed channel; another channel may respond differently at the same location. As convolution and pooling accumulate, individual output units depend on increasingly large regions of the original image.

### 9. Why learn filters from data?

A manually designed filter can detect a chosen pattern, such as an edge. However, deciding in advance which filters are useful for a complex recognition task is difficult.

CNNs instead treat filter coefficients as trainable parameters. During supervised training, backpropagation computes how changing each coefficient would affect the task loss, and optimization updates those coefficients.

Because a filter is shared across locations, its gradient combines contributions from all the positions where it was used.

Stacking layers also allows the model to build more complex features from simpler ones. Slide 18 illustrates this progression with edges, facial parts, and facial structure. This is an intuition for hierarchical representation learning, rather than a guarantee that every channel corresponds to a recognizable object part.

The resulting feature maps provide the foundation for a complete image classifier. The next section develops how these building blocks are assembled into CNN models.

### Sources and AI assistance

- Kai Wang, *Lecture 14: Convolutional Neural Networks*, CSE/ISyE 6740, October 19, 2026, slides 1-18.
- Krizhevsky, Sutskever, and Hinton, [*ImageNet Classification with Deep Convolutional Neural Networks*](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), 2012.
- PyTorch documentation: [`Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) and [`MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).

**AI assistance:** OpenAI Codex assisted with drafting and mathematical checks. The contributor reviewed the final content.
