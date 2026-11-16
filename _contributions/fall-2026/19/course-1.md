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

$\text{Values: }V = XW^v \quad \in \mathbb{R}^{L \times d_v} \quad \text{Keys: } = XW^k \quad \in \mathbb{R}^{L \times d_k} \quad \text{Queries: } = XW^q \quad \in \mathbb{R}^{L \times d_q}$


