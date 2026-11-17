---
semester: fall-2026
lecture_number: "19"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: emilynwhite2003
issue: 64
---
## Background
Transformers propose an attention mechanism to compute the similarity of different tokens and were introduced in 2017 by Google Brain. Prior to transformers, RNN and LSTMs were primarily used to store and propagate information from previous tokens. A problem is that these use recurrence, where information from previous tokens is passed sequentially through the through the network, which makes it not possible to parallelize computations, increasing computation/processing time. 

Transformers address this issue with the use of attention mechanisms, which allow the model to determine how much different tokens relate to each other in a given sequence, eliminating the need for recurrence. This allows for parallelization through the elimination of recurrence and instead allowing the relationship of all tokens to be calculated at the same time with matrix operations using self-attention. 

## What is a Self-Attention Mechanism?

Self-attention allows each token in a sequence to incorporate information about other tokens in a same given sequence. It calculates how strongly each token relates to other tokens in the sequence, and it uses these relationships to determine what information is most important. It is also sometimes called intra-attention. 

For example, consider an input sequence with length L, where each token is represented by a vector embedding with d dimensions. This creates an input matrix: 

$$
X=\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_L
\end{bmatrix}\in\mathbb{R}^{L\times d}
$$

In order for the self-attention to be calculated, we need values, keys, and queries, which are calculated using weight matrices. These weight matrices are learned during training

$\text{Values: }V = XW^v \quad \in \mathbb{R}^{L \times d_v}, \quad \text{Keys: } K = XW^k \quad \in \mathbb{R}^{L \times d_k}, \quad \text{Queries: } Q = XW^q \quad \in \mathbb{R}^{L \times d_k}$

The values (V) represent the information that will be combined to generate the attention output. The keys (K) are representations of the tokens that are used to determine how relevant a token is to other tokens are in a query. The queries (Q) represent what information each token is looking for. Each token has its own value, key, and query; individually, 
- Each query $q_i$ is computed by $x_i W^q$. 
- Each key $k_i$ is computed by $x_i W^k$. 
- Each value $v_i$ is computed by $x_i W^v$. 

In the below example, each word in the following sentence represents a token, each with its own corresponding query, key, and value. 

<img width="732" height="290" alt="Image showing how each word in a given sentence is a token with a corresponding qi, ki, and, vi. It also shows the matrix multiplication of the row xi with the weight matrix Wv equaling vi." src="https://github.com/user-attachments/assets/e71f4164-6196-4c59-a0cd-ba00d4542e1b" />

## Attention Score

To estimate the similarity between each query and key, a dot product is used:

$$
similarity_{ij} = q_i \cdot k_j^{\top} \quad \in \mathbb{R}
$$

<img width="277" height="123" alt="Image showing the multiplication of row vector qi with column vector kj transpose equalling sij." src="https://github.com/user-attachments/assets/769f6f11-feef-43c9-91aa-770eda71d9a3" />


Altogether,

$$
similarity= Q \cdot K^{\top} \quad \in \mathbb{R}^{L \times L}
$$

In the resulting similarity matrix, each row corresponds to a query token and each column corresponds to a key token. Thus, each entry represents the similarity between the two tokens. A higher similarity indicates that the key token is relevant to the query token and should receive a higher attention weight, hence the name "attention score".


This is then normalized/scaled using the row-wise softmax equation below:

$$
\text{normalized similarity} = softmax(\frac{Q \cdot K^{\top}}{\sqrt{d_k}})
$$


<img width="815" height="126" alt="Image showing each row qi being multiplied with each column kj transpose equaling all the sij elements making up the similarity matrix. It then shows after normalization/ row-wise softmax each element becomes s'ij." src="https://github.com/user-attachments/assets/b9a7e71f-8cae-43da-90e0-b10e4330e6bc" />

Dividing by $\sqrt{d_k}$ prevents the dot products from becoming too large when the dimensions of the vectors increase, essentially cancelling out the effect of dimensionality. The softmax function converts the similarity scores into nonnegative weights that sum together to 1 for each row. These normalized/scale similarity scores then represent the weights that represent the importance of each token to be applied to the values.

Each row of the similarity matrix represents how similar every token is to that specific token. For example, the first row: $s'_{11}$, $s'_{12}$, ..., $s'_{1L}$ represents how similar each token (token 1 through L) is to token 1.

Finally, the attention is calculated by multiplying the normalized softmax with V to get a resulting matrix in $\mathbb{R}^{L \times d_v}$. This looks like: 

$$
Attention(Q,K,V) = softmax(\frac{QK^{\top}}{\sqrt{d_k}})V
$$

Visually,

<img width="462" height="155" alt="This image shows how the attention matrix is created. It shows all elements of the normalized softmax being multiplied by each row vector vj giving us each row of the attention matrix ai. " src="https://github.com/user-attachments/assets/5194af0c-af1d-40b7-b5fc-8a187747c542" />

The attention output for each token becomes a weighted sum of the values, where the weights represent the similarity between the query and token keys. Altogether, this process allows each token to include relevant information from other tokens in the given sequence. 

## Masking Operators

Masking can be used to restrict what attentions we use. For example, we only want to compare information between current and previous tokens, not future tokens, so we can mask the upper triangular part of the matrix (not including the digonal). This prevents future information from influencing anything. 

<img width="656" height="149" alt="This image illustrates what the calculation of the attention matrix looks like with the upper triangular part of the normalized softmax (exluding the diagonal) discarded." src="https://github.com/user-attachments/assets/c4beaeaf-c6a9-4c32-bf3c-4edc4c226e4a" />

## Summary

Overall, self-attention allows transformers to identify relationships among tokens without recurrence. Inputs are first transformed into values, queries, and keys, then similarity is calculated and then scaled and normalized using softmax, and finally the attention weights are multiplied by values to produce the final attention output. These operations allow transformers to include contextual information while allowing for parallel computing. 

## Sources

Lecture 19 slides (all images)

"Attention Is All You Need": Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin 2017 https://arxiv.org/abs/1706.03762

https://www.math.utah.edu/~bwang/mathds/Lecture14.pdf

https://towardsdatascience.com/contextual-transformer-embeddings-using-self-attention-explained-with-diagrams-and-python-code-d7a9f0f4d94e/

https://www.codecademy.com/article/transformer-architecture-self-attention-mechanism

Chatgpt was also used to verify that my explanations were accurate


