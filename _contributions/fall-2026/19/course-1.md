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
Transformers propose an attention mechanism to compute the correlation of different tokens and were introduced in 2017 by Google Brain. Prior to transformers, RNN and LSTMs were primarily used to store and propagate information from previous tokens. A problem is that these use recurrence, where information from previous tokens is passed through the through the network, which requires sequential processing and it makes it not possible to parallelize computations, making the computation take longer. 

Transformers address these issues in the use of attention mechanisms, which allow the model to determine how much different tokens relate to each other in a given sequence, eliminating the need for recurrence. This allows for parallelization through the elimination of recurrence and instead allowing the relationship of all tokens to be calculated at the same time with matrix operations using self-attention mechanisms.  

## What is a Self-Attention Mechanism?

Self-attention allows each token in a sequence to incorporate information about other tokens in a same given sequence. It calculates how strongly each token relates to other tokens in the sequence, and it uses these relationships to determine what information is most important.  

For example, consider an input sequence with length L, where each token is represented by a vector embedding with d dimensions. This creates an input matrix: 

$$
X=\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_L
\end{bmatrix}\in\mathbb{R}^{L\times d}
$$

In order for the self-attention to be calculated, we need values, keys, and queries, which are calculated using weight matrices.

$\text{Values: }V = XW^v \quad \in \mathbb{R}^{L \times d_v}, \quad \text{Keys: } = XW^k \quad \in \mathbb{R}^{L \times d_k}, \quad \text{Queries: } = XW^q \quad \in \mathbb{R}^{L \times d_k}$

The values (V) represent the information that will be combined to generate the attention output. The keys (K) determine how relevant other tokens are in a query. The queries (Q) represent what information each token is looking for. Each token has its own value, key, and query; individually, 
- Each query $q_i$ is computed by $x_i W^q$
- Each key $k_i$ is computed by $x_i W^k$
- Each value $v_i$ is computed by $x_i W^v$

In the below example, each word in the following sentence represents a token, each with its own corresponding query, key, and value. 

<img width="732" height="290" alt="image" src="https://github.com/user-attachments/assets/e71f4164-6196-4c59-a0cd-ba00d4542e1b" />

## Attention Score

To estimate the similarity between each query and key, a dot product is used:

$$
similarity_{ij} = q_i \dot k_j^{\top} \quad \in \mathbb{R}
$$

<img width="277" height="123" alt="image" src="https://github.com/user-attachments/assets/769f6f11-feef-43c9-91aa-770eda71d9a3" />


Altogether,

$$
similarity_{ij} = Q \dot K^{\top} \quad \in \mathbb{R}^{L \times L}
$$

This is then normalized using the row-wise softmax:

$$
\text{normalized_similarity} = softmax(\frac{QK^{\top}}{\sqrt{d_k}})
$$


<img width="815" height="126" alt="image" src="https://github.com/user-attachments/assets/b9a7e71f-8cae-43da-90e0-b10e4330e6bc" />


