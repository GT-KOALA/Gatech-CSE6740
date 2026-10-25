---
semester: fall-2026
lecture_number: "14"
slot: course-2
slot_order: 2
role: course_material
role_label: Course materials
tab_title: Core explanation II
assignee: allensctong
issue: 42
---
## Lecture 14, Part II: From learned filters to CNN models

This part covers slides 18–27. Part I builds the convolutional layer, padding and stride, and pooling. Here we use those pieces to build a complete network, work out what it costs in parameters, memory, and computation, and then follow the architectures (AlexNet, VGGNet, ResNet) and training tools (ReLU, batch normalization) that let CNNs grow from 8 layers to more than 100.

**Notation.** An activation volume has shape $H \times W \times C$ (height, width, channels); PyTorch stores the same tensor as (batch, channels, height, width). A convolutional layer has $K$ filters of size $F \times F \times C$, padding $P$, and stride $S$. The training loss is $\mathcal{L}$ (the slides write $l$).

## 1. Why learn the filters? (slide 18)

Part I ended with the convolutional layer. A small filter slides over the input, and at each location it computes a dot product plus a bias, $w^\top x + b$, with the patch beneath it. The results form an activation map. One formula from Part I keeps track of sizes throughout this section:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2P - F}{S} \right\rfloor + 1
$$

That leaves an obvious question: **where do the numbers inside a filter come from?** For decades, computer vision's answer was that a person designs them. The Sobel filter is a classic example:

$$
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
$$

It responds strongly wherever brightness changes from left to right, which happens at vertical edges. Slide 18 gives three reasons why designing filters by hand does not scale:

- **Time-consuming:** every new task needs an expert to design and tune a new set of filters.
- **Brittle:** filters tuned for one setting (lighting, scale, camera) stop working when the conditions change.
- **Not scalable:** an edge is easy to write down as a $3 \times 3$ matrix, but "an ear" or "a cat's face" is not, and a 1000-class problem would need thousands of such detectors.

The slide's alternative is to **learn the filters from data**. A filter is just a set of weights, so every filter entry can be treated as a trainable parameter and fitted by gradient descent on the loss, the same way as the weights of the fully connected networks in Lecture 11. Seen this way, a convolutional layer is an ordinary neural-network layer with two constraints. Each output looks only at a small local patch (*local connectivity*), and the same weights are reused at every location (*weight sharing*).

The figure on slide 18 shows what learned filters end up looking like:

| Depth in the network | What the filters respond to |
|---|---|
| Low-level features | lines and edges at different orientations |
| Mid-level features | parts: eyes, noses, ears |
| High-level features | whole facial structures |

The first layer *rediscovers* edge detectors much like the hand-designed ones, even though nobody told it to. Zeiler and Fergus (2014) visualized CNNs trained on ImageNet and found the same progression: edges and color blobs in the first layer, textures and simple patterns after that, and object parts and whole objects in the deeper layers.

**Why a hierarchy?** Each layer's units look at a larger region of the original image than the layer before. This region is called the *receptive field*. To track it, let $r$ be the receptive field and $j$ the *jump*, the distance in input pixels between neighboring units. A layer with kernel size $k$ and stride $s$ updates them as follows:

$$
r_{\ell} = r_{\ell-1} + (k_{\ell} - 1)\, j_{\ell-1}, \qquad j_{\ell} = j_{\ell-1}\, s_{\ell}, \qquad r_0 = j_0 = 1
$$

Here is the calculation for the network in the next section, which has two $5 \times 5$ convolutions, each followed by $2 \times 2$ max pooling:

| After layer | Kernel / stride | Receptive field | Jump |
|---|---|---|---|
| conv1 | 5 / 1 | 5 × 5 | 1 |
| pool1 | 2 / 2 | 6 × 6 | 2 |
| conv2 | 5 / 1 | 14 × 14 | 2 |
| pool2 | 2 / 2 | 16 × 16 | 4 |

A conv1 unit sees a $5 \times 5$ patch, which is only enough for an edge. Each final unit summarizes a $16 \times 16$ region, half the width of a $32 \times 32$ image, which is enough for a part. Deeper networks keep growing the receptive field until units can respond to whole objects. Edges combine into parts, and parts combine into objects.

## 2. Assembling a complete CNN (slides 19–20)

### 2.1 The running example

Slide 20 combines everything into one model for $32 \times 32 \times 3$ color images and 10 classes (the format of the CIFAR-10 dataset):

```
32x32x3 image
  -> Conv2d + ReLU -> 28x28x6  -> max pool -> 14x14x6
  -> Conv2d + ReLU -> 10x10x12 -> max pool -> 5x5x12
  -> Flatten -> 300
  -> FC + ReLU -> 4096 -> FC + ReLU -> 4096 -> FC -> 10 scores -> softmax + loss
```

| Stage | Operation | Output shape |
|---|---|---|
| Input | color image | 32 × 32 × 3 |
| Conv1 | 6 filters of size 5 × 5 × 3, stride 1, no padding, then ReLU | 28 × 28 × 6 |
| Pool1 | 2 × 2 max pooling, stride 2 | 14 × 14 × 6 |
| Conv2 | 12 filters of size 5 × 5 × 6, stride 1, no padding, then ReLU | 10 × 10 × 12 |
| Pool2 | 2 × 2 max pooling, stride 2 | 5 × 5 × 12 |
| Flatten | reshape the volume into a vector | 300 |
| FC1 | fully connected, then ReLU | 4096 |
| FC2 | fully connected, then ReLU | 4096 |
| FC3 | fully connected (class scores) | 10 |
| Loss | softmax + cross-entropy | 1 number |

The slide labels only the shapes. $5 \times 5$ filters with stride 1 and $2 \times 2$ pooling with stride 2 are the choices that reproduce them exactly, and they match the filter counts on slide 14: $(32 - 5) + 1 = 28$, then $28 / 2 = 14$, then $(14 - 5) + 1 = 10$, then $10 / 2 = 5$.

This network is a close relative of **LeNet-5** (LeCun et al., 1998). LeNet-5 used the same $32 \times 32$ input, $5 \times 5$ convolutions, two rounds of downsampling, and 6 maps in the first layer. However, it had 16 maps in the second layer and much smaller fully connected layers (120 and 84 units). Section 3 shows that the 4096-wide layers are what make the slide's version expensive.

### 2.2 "How to use this???": flattening into fully connected layers

After pool2 we have 12 feature maps, each $5 \times 5$, which is a 3D volume. A fully connected (FC) layer expects a vector. The simplest bridge is to **flatten**: read the $5 \cdot 5 \cdot 12 = 300$ numbers in a fixed order into a vector $h_0 \in \mathbb{R}^{300}$. Flattening has no parameters and loses no information, since it only rearranges the same numbers. From there on, the model is the ordinary multilayer network from earlier lectures.

It helps to view the network as two halves:

- **Feature extractor** (conv + ReLU + pool): turns raw pixels into 300 learned features, each summarizing a $16 \times 16$ region (Section 1).
- **Classifier** (FC layers + softmax): maps those features to class probabilities. The last FC layer plus softmax is exactly **multiclass logistic regression**, which slide 5 also says about AlexNet. Here it runs on learned features rather than raw pixels.

The two halves are not trained separately. One loss at the end trains both together, *end to end*. Flattening is not the only way to connect them; Section 7 shows the *global average pooling* alternative used by ResNet.

### 2.3 The forward pass in equations

Take a conv layer $\ell$ with stride 1 and no padding. Its input $A^{(\ell-1)}$ has size $H \times W \times C$, and filter $k$ has weights $W_{u,v,c,k}$ and bias $b_k$. The layer computes

$$
Z^{(\ell)}_{i,j,k} = b_k + \sum_{c=1}^{C} \sum_{u=1}^{F} \sum_{v=1}^{F} W_{u,v,c,k}\, A^{(\ell-1)}_{i+u-1,\; j+v-1,\; c},
\qquad
A^{(\ell)}_{i,j,k} = \max\big(0,\; Z^{(\ell)}_{i,j,k}\big).
$$

This is slide 8's $w^\top x + b$ written out in full: $(i, j)$ is the output location and $k$ picks the filter. A $2 \times 2$ max-pooling layer with stride 2 keeps the largest value in each window:

$$
P_{i,j,k} = \max_{u,\,v \,\in\, \lbrace 0, 1 \rbrace} A_{2i-1+u,\; 2j-1+v,\; k}
$$

The classifier half is a standard network. Here $W_1, W_2, W_3$ and $b_1, b_2, b_3$ are the FC weights and biases, separate from the conv filters:

$$
h_0 = \operatorname{vec}\big(P^{(2)}\big) \in \mathbb{R}^{300}, \quad
h_1 = \mathrm{ReLU}(W_1 h_0 + b_1), \quad
h_2 = \mathrm{ReLU}(W_2 h_1 + b_2), \quad
s = W_3 h_2 + b_3 \in \mathbb{R}^{10}
$$

$$
p_k = \frac{e^{s_k}}{\sum_{m=1}^{10} e^{s_m}}, \qquad \mathcal{L} = -\log p_y
$$

where $y$ is the true class.

### 2.4 Training end to end

The trainable parameters $\theta$ are every filter weight and bias in the conv layers plus every weight and bias in the FC layers. Training minimizes the average loss over the $n$ training images with minibatch stochastic gradient descent (Lecture 12):

$$
\min_{\theta}\; \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\big(\theta;\, x_i, y_i\big)
$$

Backpropagation passes through the new layer types as follows:

- **ReLU** passes the gradient through where its input was positive and blocks it where the input was negative.
- **Max pooling** has no weights. Its gradient goes only to the input that was the maximum in each window, and the other three inputs get zero.
- **Convolution** reuses each weight at every location, so the weight's gradient is the *sum* of the contributions from all locations:

$$
\frac{\partial \mathcal{L}}{\partial W_{u,v,c,k}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial Z^{(\ell)}_{i,j,k}} \; A^{(\ell-1)}_{i+u-1,\; j+v-1,\; c}
$$

This sum is weight sharing seen from the training side: every location in every training image contributes evidence about the same small filter.

The whole model takes a few lines of PyTorch, and the last print confirms the parameter count derived in Section 3:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(),    # 32x32x3 -> 28x28x6
    nn.MaxPool2d(2),                              # -> 14x14x6
    nn.Conv2d(6, 12, kernel_size=5), nn.ReLU(),   # -> 10x10x12
    nn.MaxPool2d(2),                              # -> 5x5x12
    nn.Flatten(),                                 # -> 300
    nn.Linear(300, 4096), nn.ReLU(),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Linear(4096, 10),                          # 10 class scores
)

x = torch.randn(1, 3, 32, 32)                     # (batch, channels, height, width)
print(model(x).shape)                             # torch.Size([1, 10])
print(sum(p.numel() for p in model.parameters())) # 18057446

loss_fn = nn.CrossEntropyLoss()                   # applies softmax + log internally
```

## 3. What does it cost? (the question on slide 20)

Slide 20 asks: *what are the trainable parameters, and what do memory and computation cost?* We answer it for the running example. The pattern we find carries over to real CNNs (Sections 4–5).

### 3.1 Trainable parameters

Three counting rules cover every layer:

- A **conv layer** with $K$ filters of size $F \times F \times C$ has $(F^2 C + 1)\,K$ parameters: $F^2 C$ weights plus one bias per filter. This number **does not depend on the image size**, because the same filters are reused at every location.
- An **FC layer** from $n_{\text{in}}$ to $n_{\text{out}}$ units has $(n_{\text{in}} + 1)\, n_{\text{out}}$ parameters.
- **Pooling, ReLU, and flatten** have no parameters.

| Layer | Weights | Biases | Parameters |
|---|---|---|---|
| Conv1 (6 filters, 5 × 5 × 3) | 5 · 5 · 3 · 6 = 450 | 6 | 456 |
| Conv2 (12 filters, 5 × 5 × 6) | 5 · 5 · 6 · 12 = 1,800 | 12 | 1,812 |
| Pool1, Pool2, ReLU, Flatten | 0 | 0 | 0 |
| FC1 (300 → 4096) | 300 · 4096 = 1,228,800 | 4,096 | 1,232,896 |
| FC2 (4096 → 4096) | 4096 · 4096 = 16,777,216 | 4,096 | 16,781,312 |
| FC3 (4096 → 10) | 4096 · 10 = 40,960 | 10 | 40,970 |
| **Total** | | | **18,057,446** |

Slide 14's counts of 450 and 1,800 are the weights only; the biases add 6 and 12. Three things stand out:

1. The two conv layers together have only 2,268 parameters, about **0.013%** of the model.
2. FC2 alone has 16.8 million, about **93%** of the model. An FC layer costs the product of its two widths, and $4096 \times 4096$ is large.
3. Weight sharing makes an enormous difference. An FC layer that produced conv1's 4,704 outputs directly from the 3,072 input pixels would need $3{,}072 \times 4{,}704 + 4{,}704 = 14{,}455{,}392$ parameters, about 31,700 times conv1's 456.

### 3.2 Memory

Two different things take up memory.

**Parameter memory.** Each 32-bit float takes 4 bytes, so the weights take $18{,}057{,}446 \times 4 \approx 72$ MB. Training needs more. It stores one gradient per parameter (another 72 MB), and the Adam optimizer keeps two running averages per parameter (another 144 MB). That comes to about 289 MB before any activations are stored.

**Activation memory.** The forward pass produces an intermediate tensor at every layer. Backpropagation needs them (the gradient formula in Section 2.4 uses $A^{(\ell-1)}$), so they are kept until the backward pass. Per image:

| Tensor | Shape | Numbers stored |
|---|---|---|
| Input | 32 × 32 × 3 | 3,072 |
| Conv1 output (after ReLU) | 28 × 28 × 6 | 4,704 |
| Pool1 output | 14 × 14 × 6 | 1,176 |
| Conv2 output (after ReLU) | 10 × 10 × 12 | 1,200 |
| Pool2 output (= the flattened vector) | 5 × 5 × 12 | 300 |
| FC1 output | 4096 | 4,096 |
| FC2 output | 4096 | 4,096 |
| Class scores | 10 | 10 |
| **Total** | | **18,654 (about 75 KB)** |

This is a rough count, since frameworks may keep a few extra copies such as the values before ReLU. Activation memory grows linearly with the batch size: a batch of 256 images needs about 19 MB.

In this example, parameters dominate memory because the images are tiny. For full-size images the balance flips. VGG-16 on a $224 \times 224$ image produces about 24 million activation values in the forward pass, about 93 MB per image and roughly double that for training (CS231n notes). Most of that comes from the early conv layers, where the feature maps are large. Meanwhile, most of VGG-16's 138 million parameters sit in its FC layers. As a rule of thumb, **early conv layers dominate activation memory, and late FC layers dominate parameter memory.**

### 3.3 Computation

We count **multiply-accumulate operations (MACs)**, where one MAC is one multiplication plus one addition. Each output value of a conv layer is a dot product of length $F^2 C$, and each output of an FC layer is a dot product of length $n_{\text{in}}$:

$$
\text{MACs}_{\text{conv}} = \underbrace{H_{\text{out}} \cdot W_{\text{out}} \cdot K}_{\text{output values}} \times \underbrace{F^2 C}_{\text{filter size}},
\qquad
\text{MACs}_{\text{FC}} = n_{\text{in}} \cdot n_{\text{out}}
$$

| Layer | Calculation | MACs per image |
|---|---|---|
| Conv1 | (28 · 28 · 6) × (5 · 5 · 3) = 4,704 × 75 | 352,800 |
| Conv2 | (10 · 10 · 12) × (5 · 5 · 6) = 1,200 × 150 | 180,000 |
| FC1 | 300 × 4,096 | 1,228,800 |
| FC2 | 4,096 × 4,096 | 16,777,216 |
| FC3 | 4,096 × 10 | 40,960 |
| **Total** | | **about 18.6 million** |

That is about 37 million floating-point operations (FLOPs) for one forward pass, counting the multiply and the add separately. Pooling and ReLU only perform comparisons, a few thousand in total, which is negligible. For training, the backward pass costs roughly twice the forward pass, because it computes gradients for both the inputs and the weights. One training step therefore costs about three forward passes per image.

**Comparing 3.1 and 3.3.** Conv1 has only 456 parameters but performs 352,800 MACs, because each weight is reused at all $28 \times 28 = 784$ output positions. Every FC weight is used exactly once per image. **Conv layers are cheap in parameters but do a lot of computation per parameter.** In this toy network the FC layers still dominate computation (the conv layers do 2.9% of it), because a $4096 \times 4096$ layer is huge next to two small convolutions on a $32 \times 32$ image. In real ImageNet-scale CNNs it is the other way around. Krizhevsky (2014) estimated that conv layers hold about 5% of the parameters but do about 90–95% of the computation, and that FC layers are the reverse. Section 4 checks this for AlexNet.

### 3.4 Summary of the costs

| | Conv layers | FC layers |
|---|---|---|
| Parameters | few, because the filters are shared across locations | most |
| Activation memory | most, because the early feature maps are large | little |
| Computation | most in ImageNet-scale CNNs (about 90–95% in AlexNet-style models) | the rest; most only in small-image toy models like this one |

## 4. AlexNet (slide 21)

AlexNet (Krizhevsky, Sutskever, and Hinton, 2012) won the 2012 ImageNet challenge (ILSVRC-2012), which has 1000 classes, and started the deep-learning era in computer vision.

**What "84.7% vs. 73.8%" means.** These are *top-5* accuracies: a prediction counts as correct if the true label is among the model's five highest-scoring classes. ImageNet images often contain more than one object, so the challenge scored the top five guesses. The paper reports the same result as a top-5 test error of 15.3%, against 26.2% for the second-best entry.

**Architecture.** The diagram on slide 21 has five conv layers followed by three FC layers. Using the formula from Section 1 (and leaving out the paper's local response normalization layers):

| Layer | Operation | Output shape |
|---|---|---|
| Input | color image | 227 × 227 × 3 |
| C1 | 96 filters, 11 × 11, stride 4, + ReLU | 55 × 55 × 96 |
| Max pool | 3 × 3, stride 2 | 27 × 27 × 96 |
| C2 | 256 filters, 5 × 5, padding 2, + ReLU | 27 × 27 × 256 |
| Max pool | 3 × 3, stride 2 | 13 × 13 × 256 |
| C3 | 384 filters, 3 × 3, padding 1, + ReLU | 13 × 13 × 384 |
| C4 | 384 filters, 3 × 3, padding 1, + ReLU | 13 × 13 × 384 |
| C5 | 256 filters, 3 × 3, padding 1, + ReLU | 13 × 13 × 256 |
| Max pool | 3 × 3, stride 2 | 6 × 6 × 256 = 9,216 values |
| FC1, FC2 | 4096 units each, + ReLU | 4096 |
| FC3 | 1000 units, then softmax | 1000 |

The paper describes a $224 \times 224$ input, but $(224 - 11)/4 + 1$ is not a whole number. The sizes work out with $227 \times 227$, which is why the slide and most reimplementations use 227. The pooling windows overlap ($3 \times 3$ windows with stride 2), as in the AlexNet feature maps on slide 17.

**Parameters and computation.** Applying Section 3's counting rules:

- **Conv layers:** about 2.3 million parameters in the original model, where C2, C4, and C5 are split into two groups (one per GPU) so that each filter sees only half the input channels. Without that split, the count is about 3.7 million. The slide's "~3M" falls between the two.
- **FC layers:** about 58.6 million parameters. FC1 alone, which maps 9,216 values to 4,096, has 37.8 million.
- **Total:** about 61 million parameters. The paper's own figure is "60 million".
- **Compute:** by Section 3.3's formula, the original model performs about 666 million MACs in its conv layers and 59 million in its FC layers. That is about 0.72 billion MACs per image, 92% of them in the conv layers, consistent with Krizhevsky's 90–95% estimate.

This is the same lesson as the toy example: **the conv layers do the work, and the FC layers hold the parameters.**

**ReLU.** AlexNet used ReLU, $\max(0, u)$, instead of tanh or the sigmoid. In the paper, a four-layer CNN with ReLUs reached 25% training error on CIFAR-10 *six times faster* than the same network with tanh. Section 6 explains why.

**How long did it take?** The model was trained on 1.2 million images for about 90 epochs, which took 5–6 days on two NVIDIA GTX 580 GPUs with 3 GB of memory each. Here is a back-of-the-envelope check:

$$
\underbrace{0.72 \times 10^{9}}_{\text{MACs per image}} \times \underbrace{3}_{\text{fwd + bwd}} \times \underbrace{1.2 \times 10^{6}}_{\text{images}} \times \underbrace{90}_{\text{epochs}} \approx 2.3 \times 10^{17} \text{ MACs} \approx 4.7 \times 10^{17} \text{ FLOPs}
$$

Spread over about 5.5 days, that is roughly $10^{12}$ FLOPs per second across the two GPUs. The memory question from slide 20 was a real constraint here. The paper explains that a single 3 GB GPU limited how large a network could be trained, so the network was split across two GPUs, which is where the grouped layers come from. With 60 million parameters and only 1.2 million images, overfitting was the other big risk. The paper fought it with *dropout* in the first two FC layers and *data augmentation* (random crops, horizontal flips, and color changes).

## 5. VGGNet: depth from small filters (slide 22)

VGGNet (Simonyan and Zisserman, 2015, from Oxford's Visual Geometry Group) asked a simple question: what happens if we keep the design uniform and just make the network deeper?

**The design.** Every conv layer uses a $3 \times 3$ filter with stride 1 and padding 1, so the spatial size never changes inside a block. Every pooling layer is $2 \times 2$ with stride 2, which halves the height and width. The channel count doubles after each pooling step (64, 128, 256, 512) until it reaches 512. In the slide's diagram the volumes go $224 \times 224 \times 64 \to 112 \times 112 \times 128 \to 56 \times 56 \times 256 \to 28 \times 28 \times 512 \to 14 \times 14 \times 512 \to 7 \times 7 \times 512$, followed by FC layers of 4096, 4096, and 1000 units. Halving $H$ and $W$ divides the number of positions by 4, and doubling both the input and output channels multiplies the per-position work by 4. So these blocks do a roughly equal amount of computation each.

**Why only $3 \times 3$?** AlexNet mixed $11 \times 11$, $5 \times 5$, and $3 \times 3$ filters. VGG replaces a large filter with a stack of small ones. Two stacked $3 \times 3$ layers see a $5 \times 5$ region of their input, three see $7 \times 7$, and in general $n$ stacked layers see $(2n + 1) \times (2n + 1)$. The stack is cheaper. With $C$ input and output channels and ignoring biases:

$$
\underbrace{7 \cdot 7 \cdot C \cdot C}_{\text{one } 7 \times 7 \text{ layer}} = 49\, C^2
\qquad \text{vs.} \qquad
\underbrace{3 \cdot (3 \cdot 3 \cdot C \cdot C)}_{\text{three } 3 \times 3 \text{ layers}} = 27\, C^2
$$

That is 45% fewer weights; the paper puts it the other way, saying the single $7 \times 7$ layer needs 81% more. The stack also applies three ReLUs instead of one. The authors argue that this makes the function the network computes "more discriminative", meaning more expressive for the same receptive field.

**Cost.** The slide's highlighted configuration D is **VGG-16**: 13 conv layers and 3 FC layers, with **138 million parameters**. Applying Section 3's rules gives 14.7 million parameters in the conv layers and 123.6 million (89%) in the FC layers. FC1 alone, which maps the $7 \times 7 \times 512 = 25{,}088$ flattened values to 4096 units, has 102.8 million. VGG-16 needs about 15.3 billion multiply-adds per image, about 20 times as many as AlexNet.

**"The deeper the better!"** The paper's configurations A through E grow from 11 to 19 weight layers, and the error dropped as depth increased. The authors also note that the error *saturated* at 19 layers. Stacking more plain layers stops helping and eventually makes training harder. The next two slides explain why, and how ResNet got around it.

## 6. The vanishing gradient problem (slide 23)

Consider a 100-layer network, with $z_0$ the first layer's output and $z_{99}$ the last hidden layer's, and let $w$ be a weight in the first layer. The chain rule gives

$$
\frac{\partial \mathcal{L}}{\partial w}
= \frac{\partial \mathcal{L}}{\partial z_{99}}
\cdot \frac{\partial z_{99}}{\partial z_{98}}
\cdot \frac{\partial z_{98}}{\partial z_{97}}
\cdots
\frac{\partial z_{1}}{\partial z_{0}}
\cdot \frac{\partial z_{0}}{\partial w}
$$

To see what each factor looks like, use one unit per layer, with $z_t = \sigma(u_t)$ and $u_t = w_t z_{t-1} + b_t$. Then

$$
\frac{\partial z_t}{\partial z_{t-1}} = \sigma^{\prime}(u_t)\, w_t
\qquad \Longrightarrow \qquad
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z_{99}} \left( \prod_{t=1}^{99} \sigma^{\prime}(u_t)\, w_t \right) \frac{\partial z_0}{\partial w}
$$

With many units per layer, each factor becomes a matrix, $\operatorname{diag}\big(\sigma^{\prime}(u_t)\big)\, W_t$, but the story is the same. The gradient for an early layer is a **product of 99 factors**. If the factors are typically smaller than 1 in size, the product shrinks exponentially and the gradient *vanishes*. If they are larger than 1, it *explodes*. As the slide puts it, any factor that is nearly 0 ruins the whole chain. The activation function's derivative appears in every factor, which is why the choice of activation matters so much.

| Activation | $\sigma(u)$ | $\sigma^{\prime}(u)$ | Effect on the chain |
|---|---|---|---|
| Hard limiter (step) | 1 if $u \ge 0$, else 0 | 0 wherever it is defined | no gradient at all, so it cannot be trained by gradient descent |
| Piecewise linear | 0, then a linear ramp, then 1 | 1 on the ramp, 0 outside it | the gradient dies once a unit saturates |
| Sigmoid | $1/(1+e^{-u})$ | $\sigma(u)(1-\sigma(u)) \le 1/4$ | shrinks every factor by at least 4×, and is about 0 when saturated |
| Hyperbolic tangent | $\tanh u$ | $1-\tanh^2 u \le 1$ | about 0 when saturated |
| ReLU | $\max(0, u)$ | 1 if $u > 0$, 0 if $u < 0$ | passes the gradient through unchanged on the active side |

The red boxes on the slide mark the flat, *saturated* regions where $\sigma^{\prime}(u) \approx 0$. The sigmoid is a problem even when no unit is saturated. Its slope is at most $1/4$, reached at $u = 0$, so even in the best case the activation factors alone shrink the gradient by

$$
\left(\tfrac{1}{4}\right)^{5} \approx 10^{-3}, \qquad
\left(\tfrac{1}{4}\right)^{10} \approx 9.5 \times 10^{-7}, \qquad
\left(\tfrac{1}{4}\right)^{20} \approx 9.1 \times 10^{-13}
$$

over 5, 10, and 20 layers. Making the weights larger to compensate does not help much, because larger weights push the pre-activations $u_t$ into the flat regions.

**ReLU.** For an active unit the derivative of ReLU is exactly 1, so it does not shrink the gradient, and it is cheap to compute (one comparison). Its gradient is preserved only on one side, though. For $u < 0$ the gradient is 0, and a unit whose input is negative for every example stops learning altogether, a problem known as a "dead ReLU". Variants keep a little gradient on the negative side. *Leaky ReLU* is $\max(\alpha u, u)$ with a small $\alpha$ such as 0.01. *GELU* is $u\,\Phi(u)$, where $\Phi$ is the standard normal CDF (Hendrycks and Gimpel, 2016). GELU is the green curve in the slide's plot: it is smooth and dips slightly below zero, with a minimum of about $-0.17$ near $u \approx -0.75$, and it is now standard in Transformers.

The activation function is only half the story, because the weights appear in every factor too. Initialization schemes scale the random starting weights so that signals neither grow nor shrink from layer to layer; for ReLU networks the usual choice is variance $2 / n_{\text{in}}$ (He et al., 2015). The next two slides add two architectural fixes: shortcut connections and normalization.

## 7. ResNet: shortcut connections (slide 24)

ResNet (He, Zhang, Ren, and Sun, 2016) changed the basic building block. A **residual block** computes a correction $\mathcal{F}(x)$ and adds it back to its input:

$$
y = \mathrm{ReLU}\big(\mathcal{F}(x) + x\big), \qquad
\mathcal{F}(x) = \mathrm{conv}_2\Big(\mathrm{ReLU}\big(\mathrm{conv}_1(x)\big)\Big)
$$

Here both convolutions are $3 \times 3$, and the paper also applies batch normalization after each one. The curved arrow labeled "identity" on the slide is the **shortcut**. It has no parameters, and the only computation it adds is one addition.

**The problem it addresses.** The slide's training curves come from Figure 1 of the paper. On CIFAR-10, a 56-layer "plain" network (no shortcuts) has *higher training error* than a 20-layer one, not just higher test error. That rules out overfitting, which would show up as lower training error and higher test error. The deeper network is simply harder to optimize, and the authors call this the **degradation problem**. It is surprising, because a 56-layer network could in principle copy the 20-layer one and set its extra 36 layers to the identity. Plain layers apparently find the identity hard to learn. In a residual block, the identity just means $\mathcal{F}(x) = 0$, which the block reaches by driving its weights toward zero.

**Why gradients survive.** Ignoring the final ReLU, the derivative of a block is

$$
\frac{\partial y}{\partial x} = I + \frac{\partial \mathcal{F}}{\partial x}
$$

A follow-up paper (He et al., 2016b) moves the ReLU inside the branch so that the shortcuts are pure identities. The output of block $L$ is then the output of an earlier block $l$ plus a sum of corrections:

$$
x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i)
\qquad \Longrightarrow \qquad
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i) \right)
$$

The $I$ term carries the gradient from block $L$ straight back to block $l$ without multiplying it by any weight matrix or activation derivative. Unlike the 99-factor product in Section 6, this path cannot vanish just because some factor along it is small.

> **A nuance worth knowing.** The slide presents shortcut connections as the fix for vanishing gradients. The ResNet paper itself is more careful. Its plain networks already used batch normalization, and the authors argue the degradation is "unlikely to be caused by vanishing gradients", calling it an optimization difficulty instead. The gradient argument above comes from the follow-up paper. Both views agree on the practical result: with shortcuts, much deeper networks train well.

**The architecture diagram** (right side of the slide) compares VGG-19, a 34-layer plain network, and a 34-layer residual network. The dotted shortcuts mark the places where the height and width halve and the channel count doubles. There the shortcut must change shape too, which the paper does with a $1 \times 1$ convolution of stride 2 (a *projection*) or by padding with zeros. The ResNet ends with **global average pooling**, which averages each final $7 \times 7$ map into a single number, followed by one 1000-way FC layer. This is a second answer to slide 20's "How to use this???" question. Instead of flattening into a huge FC layer, global average pooling gives a 512-dimensional vector with no parameters, so the FC layer has only $512 \times 1000 + 1000 = 513{,}000$ parameters. As a result, ResNet-34 has about 21.8 million parameters, compared with about 143.7 million for VGG-19. It needs 3.6 billion multiply-adds per image, compared with 19.6 billion for VGG-19.

**Results.** With shortcuts, depth became usable. The paper trained a 152-layer network on ImageNet, which at 11.3 billion multiply-adds is still cheaper than VGG-16 or VGG-19. An ensemble of ResNets won ILSVRC 2015 with a 3.57% top-5 error, compared with AlexNet's 15.3% three years earlier. On CIFAR-10 the authors trained a 1202-layer network without optimization trouble: its training error was similar to a 110-layer network's. It tested worse, though, which the authors attribute to overfitting rather than optimization.

## 8. Batch normalization (slide 25)

Batch normalization (Ioffe and Szegedy, 2015; the slide spells the first author "Loffe") normalizes the input of a layer using statistics from the current minibatch. Given a minibatch of $B$ examples $x^{(1)}, \dots, x^{(B)} \in \mathbb{R}^{d}$, it computes the elementwise mean and variance and standardizes each example:

$$
\mu_B = \frac{1}{B} \sum_{i=1}^{B} x^{(i)}, \qquad
\sigma_B^2 = \frac{1}{B} \sum_{i=1}^{B} \big(x^{(i)} - \mu_B\big)^2, \qquad
\hat{x}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

All operations are elementwise, so $\mu_B, \sigma_B^2, \hat{x}^{(i)} \in \mathbb{R}^d$, and the small constant $\epsilon$ (PyTorch's default is $10^{-5}$) avoids division by zero. Then comes a step that the PyTorch formula on the slide includes, a **learnable scale and shift**:

$$
y^{(i)} = \gamma \odot \hat{x}^{(i)} + \beta, \qquad \gamma, \beta \in \mathbb{R}^{d}
$$

Without $\gamma$ and $\beta$, every layer's inputs would be forced to have mean 0 and variance 1, which might not be what the network needs. With them, the network can undo the normalization if that helps: setting $\gamma = \sqrt{\sigma_B^2 + \epsilon}$ and $\beta = \mu_B$ recovers the original input.

**In a CNN (BatchNorm2d).** For conv layers, the statistics are computed **per channel**, pooled over the batch and all spatial positions $(N, H, W)$. This is the blue region of the "Batch Norm" cube on the slide. Each channel gets its own $\gamma$ and $\beta$, which connects back to slide 20's question about trainable parameters:

- A BN layer on $C$ channels adds $2C$ **trainable** parameters. Adding BN after conv1 and conv2 in the running example would add $2 \cdot 6 + 2 \cdot 12 = 36$ parameters, which is negligible.
- It also keeps a running mean and a running variance per channel. These are updated during training but **not trained by gradient descent**.
- A conv layer followed by BN no longer needs its own bias, because $\beta$ does the same job. That is why ResNet-style code uses `bias=False` for those convolutions.

**Training vs. testing.** During training, BN uses the statistics of the current batch. At test time it uses the running averages instead (in PyTorch, an exponential moving average with momentum 0.1, selected by `model.eval()`). That way a test image's prediction does not depend on which other images happen to be in its batch.

**Why does it work?** As the slide says, BN works very well in practice, but there is no consensus on why. The original paper credits it with reducing *internal covariate shift*: the distribution of each layer's inputs keeps changing as earlier layers update, and BN stabilizes it. Santurkar et al. (2018) argued that this stability "has little to do with the success of BatchNorm". In their view BN mainly makes the loss surface smoother, which keeps gradients more predictable and allows larger learning rates. The slide also warns that BN can cause gradient instability. Yang et al. (2019) showed that in deep networks without skip connections, "batch normalization itself is the cause of gradient explosion" at initialization. BN's batch statistics also get noisy when batches are small.

**Group Normalization** (Wu and He, 2018; the slide spells the first author "We") avoids depending on the batch. The four cubes on the slide, taken from that paper, show which numbers share one mean and variance. $N$ is the batch axis, $C$ the channels, and $H, W$ the spatial positions, drawn as a single axis.

| Method | Statistics shared over | Depends on the batch? |
|---|---|---|
| Batch Norm | $N, H, W$: one mean and variance per channel | yes |
| Layer Norm | $C, H, W$: one per example | no |
| Instance Norm | $H, W$: one per example and channel | no |
| Group Norm | $H, W$ and a group of channels: one per example and group | no |

With $G$ groups, Group Norm becomes Layer Norm when $G = 1$ and Instance Norm when $G = C$. Its paper reports that on ResNet-50 for ImageNet with a batch size of 2, GN had 10.6% lower error than BN. That makes GN useful in tasks such as object detection, where memory allows only a few large images per GPU.

## 9. Deep learning today (slides 26–27)

**Slide 26** plots the best ImageNet *top-1* accuracy over time (a stricter measure than AlexNet's top-5: the single best guess must be right). The state of the art rose from below 70% (FireCaffe, a GoogLeNet-based model) to about 78% with ResNet-152 by 2016. ResNet variants such as ResNeXt pushed it into the mid-80s, and it passed 90% around 2020–2021 with Meta Pseudo Labels (EfficientNet-L2) and ViT-G/14. The latest point on the chart, OmniVec, is at about 92%. Two trends drive the curve: much larger training sets (extra labeled or weakly labeled images, and pseudo-labels on unlabeled ones), and since about 2020, *Vision Transformers* (ViT, CoCa), which mostly replace convolutions with attention. Transformers come later in this course.

**Slide 27** is Epoch AI's plot of parameter counts for notable ML systems (Sevilla, Villalobos, and Cerón, 2021), on a logarithmic axis where each gridline is 10 times the last. The models in this lecture fit on it: our running example has $1.8 \times 10^{7}$ parameters, AlexNet $6 \times 10^{7}$, and VGG-16 $1.4 \times 10^{8}$, in line with the vision systems of their time. The article estimates that from 2000 until about 2016–2018 parameter counts doubled every 18–24 months in all domains. Since then, language models (red squares) have grown much faster and now exceed $10^{11}$–$10^{12}$ parameters.

The questions from slide 20 (how many parameters, how much memory, how much computation) are exactly what limits these models. AlexNet was split across two GPUs because 3 GB of memory was not enough. Today's largest models are trained on thousands of accelerators for the same reason, at a much larger scale.

## 10. Summary

- Slide 18's idea is to **learn filters from data instead of designing them by hand**. Stacked conv layers learn a hierarchy (edges, then parts, then objects) because each layer's receptive field is larger than the one before.
- A **complete CNN** is a conv/ReLU/pool feature extractor, then flattening (or global average pooling), then FC layers and softmax, trained end to end with backpropagation. With shared weights, a filter's gradient is a sum over all locations.
- **Costs:** a conv layer has $(F^2 C + 1)\,K$ parameters regardless of image size. FC layers hold most of the parameters (93% in the running example, about 96% in AlexNet, 89% in VGG-16). Conv layers do most of the computation in real CNNs, and early layers dominate activation memory.
- **AlexNet** (2012) showed that deep CNNs with ReLU, trained on GPUs, work at ImageNet scale. **VGGNet** (2015) showed that stacks of $3 \times 3$ filters cover large receptive fields more cheaply, and that depth helps, up to a point.
- **Very deep plain networks are hard to train.** Gradients are products of many factors that can vanish, and plain networks show the degradation problem. ReLU, careful initialization, **residual shortcuts** ($y = \mathcal{F}(x) + x$), and **batch normalization** together made networks with 100+ layers practical.

## References

- Wang, K. (2026). *CSE/ISyE 6740: Computational Data Analysis*, Lecture 14: "Convolutional Neural Networks," Georgia Tech, Fall 2026 (10/19/2026), slides 18–27. The running example and all figures discussed are from these slides; the figures are described here, not reproduced.
- Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS 25*. [Paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- Krizhevsky, A. (2014). One weird trick for parallelizing convolutional neural networks. [arXiv:1404.5997](https://arxiv.org/abs/1404.5997)
- LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. [DOI](https://doi.org/10.1109/5.726791)
- Zeiler, M. D., and Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*. [arXiv:1311.2901](https://arxiv.org/abs/1311.2901)
- Simonyan, K., and Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- He, K., Zhang, X., Ren, S., and Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- He, K., Zhang, X., Ren, S., and Sun, J. (2016a). Deep residual learning for image recognition. *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- He, K., Zhang, X., Ren, S., and Sun, J. (2016b). Identity mappings in deep residual networks. *ECCV*. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
- Ioffe, S., and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Wu, Y., and He, K. (2018). Group normalization. *ECCV*. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
- Santurkar, S., Tsipras, D., Ilyas, A., and Madry, A. (2018). How does batch normalization help optimization? *NeurIPS*. [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
- Yang, G., Pennington, J., Rao, V., Sohl-Dickstein, J., and Schoenholz, S. S. (2019). A mean field theory of batch normalization. *ICLR*. [arXiv:1902.08129](https://arxiv.org/abs/1902.08129)
- Hendrycks, D., and Gimpel, K. (2016). Gaussian error linear units (GELUs). [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- PyTorch documentation: [torch.nn.BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- Stanford CS231n course notes: [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) (source of the VGG-16 memory figures in Section 3.2).
- Sevilla, J., Villalobos, P., and Cerón, J. F. (2021). [Parameter counts in machine learning](https://epoch.ai/blog/parameter-counts). Epoch AI.

**AI assistance:** This section was drafted with help from Claude (Anthropic), and I reviewed the full text against the lecture slides before submitting.
