---
semester: fall-2026
lecture_number: "14"
slot: optional-1
slot_order: 3
role: optional_topic
role_label: Optional topic
tab_title: Extension I
assignee: AbirHaque
issue: 40
---
## Assignment pending

# Lecture 5 Optional: CNN filters, convolution, and signal processing
Note regarding AI usage: Gemini was utilized to help reinforce my understanding of concepts from various websites and articles. The actual topics and text presented in this document is written in my own words.

## 1. What is the connection between convolutional layers, filters, and signal processing? 
Filters, also known as kernels, are matrices that are used for extracting features in classical signal processing [1]. The convolution operation is applied onto an input signal/image with a filter [2]. Convolutional layers slide the filters to compute feature maps. The convolution operation within convolutional neural networks (CNNs) is utilized to optimize the filters automatically, whereas traditional signal processing utilizes hand-engineered filters [2,3].

### References
1. https://en.wikipedia.org/wiki/Kernel_(image_processing)
2. https://en.wikipedia.org/wiki/Convolutional_neural_network
3. Lecture 14, slide 18.

## 2. Why use element-wise products followed by summation over local patches and shared filters rather than conventional matrix multiplication directly between an image and filter?

Element-wise products followed by a summation is the same thing as the dot product [1]. In particular, this operation is done in a sliding manner over local patches that are the same size as the input shared kernel. The issue with utilizing conventional matrix multiplication is that it has a significantly higher memory usage requirement [2]. This is because portions of patches end up being duplicated across the matrix. Furthermore, this requires storing not only the original input in memory, but also the transformed version of the input. This transformed input essentially stretches patches into columns that can be multiplied with the kernel.

### References
1. https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
2. https://cs231n.github.io/convolutional-networks/

## 3. Explain filtering and learned feature detectors, local receptive fields, spatial structure, weight sharing, and channel summation.

Filtering is the process of using fixed, pre-defined matrices known as kernels to extract features [1]. This entire process is hand-engineered, which is time consuming and impractical [2]. 

Learned feature detectors learns features directly from data by optimizing filters via CNNs. The local receptive field is a hyperparameter that defines the size of the input region  that is connected to a neuron [1,3,4]. This is also known as the filter size. 

Spatial structure is the actual topology of the data [3,4]. In particular, traditional neural network architectures assume that pixel data have an all-to-all relationship. This is computationally expensive to operate on, since it is more typical for neighboring points to be significantly more related. Hence, CNNs exploit this spatial local correlation in the spatial structure to only operate on neighboring pixel data.

Weight sharing is the reuse of filter weights across all of the spatial structure of the data [4,5]. This helps limit the number of total parameters within a given layer, which also helps avoid the vanishing/exploding gradients problem.

Channel summation is the operation of summing feature maps derived from convolutions applied to each channel individually [6]. The channels of input data are typically just color channels for an image, i.e., red, green, and blue. The purpose of this operation is to reduce the number of output features that have to be managed computationally.

### References
1. https://en.wikipedia.org/wiki/Kernel_(image_processing)
2. Lecture 14, slide 18.
3. https://en.wikipedia.org/wiki/Receptive_field
4. https://cs231n.github.io/convolutional-networks/
5. https://en.wikipedia.org/wiki/Convolutional_neural_network
6. https://medium.com/@w.wojtak/making-sense-of-cnns-breaking-down-the-core-concepts-5bac7a2ceaab


## 4. Clarify convolution versus the cross-correlation convention commonly used in neural-network libraries. Investigate how the same operation can be represented or implemented using appropriately structured matrix multiplication; avoid assuming convolution and matrix multiplication are inherently incompatible. Use a small example or diagram, equations where helpful, and cited references. A full CNN implementation is not required.

The actual operation utilized in CNN implementations is the cross-correlation operation, not the convolution operation [1,2,3]. In particular, it is technically a misnomer to call the operation a true convolution. The convolution operation requires the kernel to be flipped before being applied to the image, whereas cross-correlation does not. If the filter is symmetric, then the two operations are mathematically identical. Note that since filter weights are learned, cross-correlation does not affect the final results. Instead, cross-correlation is utilized to improve performance, as the extra flip operation is avoided [4].

This operation can be represented as a matrix multiplication via the im2col representation of images [5,6]. In the im2col algorithm, local patches are flattened into a column vector. All column vectors are placed next to each other to form a matrix, which we call $X$. The filters are similarly transformed into row vectors. Again, similarly to the input image, the all filter row vectors are arranged into another matrix, which we call $W$. The output feature map matrix $Y$ is computed via matrix multiplication with $W$ and $X$ as $Y=W*X$. A very small example is provided below:

$$X_{\text{image}}=\begin{bmatrix} 1,2,3\\ 4,5,6 \end{bmatrix} $$
$$W_{\text{filter}}=\begin{bmatrix} 1,1\\1,1  \end{bmatrix} $$
$$X_{\text{patch}}^1=\begin{bmatrix} 1,2\\ 4,5 \end{bmatrix} \rightarrow \begin{bmatrix}1\\2\\ 4\\5 \end{bmatrix} $$
$$X_{\text{patch}}^2=\begin{bmatrix} 2,3\\ 5,6 \end{bmatrix} \rightarrow \begin{bmatrix}2\\3\\ 5\\6 \end{bmatrix} $$
$$X_{\text{im2col}}=\begin{bmatrix}1,2\\2,3\\4,5\\5,6 \end{bmatrix}$$
$$W_{\text{im2col}}=\begin{bmatrix}1,1,1,1 \end{bmatrix}$$
$$Y=W_{\text{im2col}}*X_{\text{im2col}}$$

# References
1. https://courses.cs.washington.edu/courses/cse446/22wi/sections/08/convolutional_networks.html
2. https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5/
3. https://discuss.pytorch.org/t/why-are-pytorch-convolutions-implemented-as-cross-correlations/115010
4. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8721631
5. https://cs231n.github.io/convolutional-networks/
6. https://triton-lang.org/main/getting-started/tutorials/gluon/conv-im2col.html
