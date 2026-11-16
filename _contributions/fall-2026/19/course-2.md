---
semester: fall-2026
lecture_number: "19"
slot: course-2
slot_order: 2
role: course_material
role_label: Course materials
tab_title: Core explanation II
assignee: erikgoeke
issue: 65
---
## Multi-head Attention Mechanisms

A single attention operation gives each token a weighted average of other tokens, which is very limiting since a token can relate to one piece of context in more ways than one. For example, if I said I learned in CSE6740 learned is connected both to "I" who did the learning, as well as "CSE6740" which is where I did the learning.

Multi-head attention mechanisms solve this by running attention operations, or heads, in parallel. This means that we will no longer have a situation where each token makes one query, key, and value, and receives one output. With multi-head attention mechanisms, the embedding is broken down and worked on in parallel.

This means each head j will have its own learned weight matrices, which in turn produce that head's queries, keys, and values:

$$
Q_j = XW_j^Q, \qquad K_j = X W_j^K, \\qquad V_j = X W_j^V
$$

Then, using these values, each head will also calculate its own self-attention:

$$
\text{attention} = \text{softmax} \left (\frac{Q K^T}{\sqrt{d_k}} \right)
$$

Following this step, all the outputs from the head are concatenated, and a final weight matrix is applied to transform the concatenated output into the desired dimensions. A diagram provided by https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/ can be seen below.

<img width="1020" height="528" alt="image" src="https://github.com/user-attachments/assets/1f021c14-53eb-459e-9623-f1af30a5b774" />


The best thing about this is that it comes at no extra cost because each head focuses on a smaller part of the embedding, so running on multiple heads costs the same as running on a full-width head.

Some helpful reading:
https://arxiv.org/abs/1706.03762
https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/

## Building the Transformer Layer

Taking the sentence "I went to Georgia Tech CSE 6740 and learned" as the example, each token embedding (or in this case, word) goes in as $$x_i$$, and then each position gets an output $$a_i$$ that has some information about the other tokens in the sentence mixed into it. Then each $$a_i$$ is passed through a Feed-Forward Network to produce $$o_i$$. In this step, each token is treated separately; attention is the step where tokens take each other into account. This step simply processes the information it finds within each token. These steps are then repeated N times, where the outputs $$o_i$$ become the $$x_i$$ of the next iteration. Doing it this way allows the model to learn more about the embedding and find more features. This is how one layer is constructed, when these layers are stacked, this is what becomes a model.

<img width="1147" height="467" alt="image" src="https://github.com/user-attachments/assets/0a3f556f-5696-4f52-a702-67836db5cd39" />

## PyTorch Implementation

Using mishig25's minGPT model as an example, the PyTorch method for the aforementioned steps can be analyzed.

The CasualSelfAttention() method splits the embedding into chunks and then calculates q, k, and v. What was once (B, T, C) is now $$(B, n_h, T, C/n_h)$$ which means each head will get its own slice of the embedding.

<img width="621" height="88" alt="image" src="https://github.com/user-attachments/assets/1d4caf6f-064c-45ca-befe-07286b8d1372" />

Next, an attention score is calculated, and normalization is performed. The transpose operation in the code gives a score matrix, which is then in turn multiplied by $$\frac{1}{\sqrt{d_k}}$$ in order to normalize.

<img width="618" height="35" alt="image" src="https://github.com/user-attachments/assets/49c02760-bb59-40b6-b01b-a49b762f842c" />

Next, entries above the diagonal line are filled with a value of negative infinity, because this way after the softmax they are 0 and tokens will not look at future tokens. The softmax is then applied, and dropout occurs. This is when, during training, the model randomly sets some numbers to zero to prevent overfitting. This works because the model can't focus on one connection always being the tell, it forces it to rely on many in case one is ever missing or not there.

<img width="622" height="49" alt="image" src="https://github.com/user-attachments/assets/159f7ce7-e4f6-493e-b42c-c062764d8119" />

The next step is to multiply by the values and then concatenate. This is the step where the weights are actually being used, and each token's output becomes a weighted average of the value vectors with the softmax weights.

<img width="622" height="33" alt="image" src="https://github.com/user-attachments/assets/f819f2e5-2110-4320-b8cf-9091f5bdc30f" />

Finally, the heads interact with one another and are added together after being multiplied by a block W, which determines how much impact each head's result should have on the overall vector.

<img width="617" height="34" alt="Screenshot 2026-11-16 at 6 07 41 PM" src="https://github.com/user-attachments/assets/702969da-da72-4b79-bd12-37a4d82b78f7" />

Also important is the transformer block, which is shown below.

<img width="804" height="118" alt="Screenshot 2026-11-16 at 6 09 20 PM" src="https://github.com/user-attachments/assets/9f946031-a4ad-4f58-a2cf-2f29d6edd5bb" />

Some things worth noting here are the residual connections, which help prevent vanishing gradients. As the number of layers increases, the gradients can become very small or vanish, which makes training unstable. Introducing residuals allows gradients to flow through identity shortcuts or skip connections, and this can reduce the vanishing gradient problem. This is what is seen in the code through x + self.attn(). The image below shows a Residual Block courtesy of https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/.

<img width="992" height="442" alt="Screenshot 2026-11-16 at 6 14 00 PM" src="https://github.com/user-attachments/assets/485de4bd-642d-4b73-b113-344d2b88bd64" />

Helpful links:
https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/
https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
https://www.youtube.com/watch?v=kCc8FmEb1nY






