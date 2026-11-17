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

This means each head $$j$$ will have its own learned weight matrices, which in turn produce that head's queries, keys, and values:

$$
Q_j = XW_j^Q, \qquad K_j = X W_j^K, \qquad V_j = X W_j^V
$$

Then, using these values, each head will also calculate its own self-attention:

$$
\text{attention} = \text{softmax} \left(\frac{Q_j K_j^T}{\sqrt{d_k}} \right) V_j
$$

Following this step, all the outputs from the head are concatenated, and a final weight matrix is applied to transform the concatenated output into the desired dimensions. A diagram provided by https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/ can be seen below.

<img width="1020" height="528" alt="An image showing the architecture of a multi-head attention mechanism." src="https://github.com/user-attachments/assets/1f021c14-53eb-459e-9623-f1af30a5b774" />


The best thing about this is that it comes at no extra cost because each head focuses on a smaller part of the embedding, so running on multiple heads costs the same as running on a full-width head.

## Building the Transformer Layer

Taking the sentence "I went to Georgia Tech CSE 6740 and learned" as the example, each token embedding (or in this case, word) goes in as $$x_i$$, and then each position gets an output $$a_i$$ that has some information about the other tokens in the sentence mixed into it. Then each $$a_i$$ is passed through a Feed-Forward Network to produce $$o_i$$. In this step, each token is treated separately; attention is the step where tokens take each other into account. This step simply processes the information it finds within each token. These steps are then repeated N times, where the outputs $$o_i$$ become the $$x_i$$ of the next iteration. Doing it this way allows the model to learn more about the embedding and find more features. This is how one layer is constructed; when these layers are stacked, this is what becomes a model.

<img width="1147" height="467" alt="an image showing the architecture of taking an input sentence, creating an output, and feeding it in again as the next input. In other words, a diagram of the architecture of one layer." src="https://github.com/user-attachments/assets/0a3f556f-5696-4f52-a702-67836db5cd39" />

*Source: Kai Wang, Lecture 19: Transformers*

## PyTorch Implementation

Using Andrej Karpathy's minGPT model as an example, the PyTorch method for the aforementioned steps can be analyzed.

The CausalSelfAttention class splits the embedding into chunks and then calculates q, k, and v. What was once (B, T, C) is now $$(B, n_h, T, C/n_h)$$ which means each head will get its own slice of the embedding.

<img width="621" height="88" alt="An image displaying the splitting and calculation of q, k, and v" src="https://github.com/user-attachments/assets/1d4caf6f-064c-45ca-befe-07286b8d1372" />

*Source: Kai Wang, Lecture 19: Transformers*

Next, an attention score is calculated, and normalization is performed. The transpose operation in the code gives a score matrix, which is then in turn multiplied by $$\frac{1}{\sqrt{d_k}}$$ in order to normalize.

<img width="618" height="35" alt="An image displaying code showing the calculation of an attention score and the application of normalization" src="https://github.com/user-attachments/assets/49c02760-bb59-40b6-b01b-a49b762f842c" />

*Source: Kai Wang, Lecture 19: Transformers*

Next, entries above the diagonal line are filled with a value of negative infinity, because this way, after the softmax, they are 0 and tokens will not look at future tokens. The softmax is then applied, and dropout occurs. This is when, during training, the model randomly sets some numbers to zero to prevent overfitting. This works because the model can't focus on one connection always being the tell; it forces it to rely on many in case one is ever missing or not there.

<img width="622" height="49" alt="An image displaying code showing the softmax step in gpt-2" src="https://github.com/user-attachments/assets/159f7ce7-e4f6-493e-b42c-c062764d8119" />

*Source: Kai Wang, Lecture 19: Transformers*

The next step is to multiply by the values and then concatenate. This is the step where the weights are actually being used, and each token's output becomes a weighted average of the value vectors with the softmax weights.

<img width="622" height="33" alt="An image displaying the concatenation step in gpt-2" src="https://github.com/user-attachments/assets/f819f2e5-2110-4320-b8cf-9091f5bdc30f" />

*Source: Kai Wang, Lecture 19: Transformers*

Finally, the heads interact with one another and are added together after being multiplied by a block W, which determines how much impact each head's result should have on the overall vector.

<img width="617" height="34" alt="An image displaying code which displays heads interacting after being multiplied by W" src="https://github.com/user-attachments/assets/702969da-da72-4b79-bd12-37a4d82b78f7" />

*Source: Kai Wang, Lecture 19: Transformers*

Also important is the transformer block, which is shown below.

<img width="804" height="118" alt="An image of code showing the transformer block" src="https://github.com/user-attachments/assets/9f946031-a4ad-4f58-a2cf-2f29d6edd5bb" />

*Source: Kai Wang, Lecture 19: Transformers*

Some things worth noting here are the residual connections, which help prevent vanishing gradients. As the number of layers increases, the gradients can become very small or vanish, which makes training unstable. Introducing residuals allows gradients to flow through identity shortcuts or skip connections, and this can reduce the vanishing gradient problem. This is what is seen in the code through x + self.attn(). The image below shows a Residual Block courtesy of https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/.

<img width="992" height="442" alt="An image showing an example of a residual block architecture" src="https://github.com/user-attachments/assets/485de4bd-642d-4b73-b113-344d2b88bd64" />

## Transformer Analysis

The GPT class in minGPT is made up of four different pieces. The first is the attention block, which makes n_layer copies of the previously created block. This is the part of the process where we repeat "N" times; this is labelled as h. Next is wte, or the word token embedding; this is when a table is used in order to make each token into a vector. Next is wpe, or word positional embedding, which marks the position; attention alone is not enough because it has no idea of position, so wpe is needed to add this context. Finally, optional caching can help speed things along; instead of recomputing each time, a new token can use the previously calculated values of k and v.

<img width="585" height="156" alt="An image of code showing the attention block, wte, and wpe" src="https://github.com/user-attachments/assets/8052d154-7642-4a08-8e37-eae50455a8d7" />

*Source: Kai Wang, Lecture 19: Transformers*

GPT-2 comes in 4 sizes: gpt2, gpt2-medium, gpt2-large, and gpt2-xl. With more parameters come higher costs for memory and compute; however, the advantage is a better model. Logically, this makes sense: storing more parameters is expensive; however, doing so also allows for better and more tailored representations.

<img width="603" height="127" alt="An image of code showing the different gpt-2 model sizes" src="https://github.com/user-attachments/assets/476a2604-d863-46ec-92d9-a4f99e020099" />

*Source: Kai Wang, Lecture 19: Transformers*

So then what do 1B parameters actually mean?

Well, "1B parameters" means that the model has learned one billion numbers, and how much memory these numbers take depends on how many bytes are used per number. This is to say, depending on the precision of the parameters used, we can have different memory sizes.
- **32-bit float (FP32):** 4 bytes per parameter, so $$4 \text{ bytes} \times 10^9 = 4 \text{ GB}$$
- **8-bit integer (INT8):** 1 byte per parameter, so $$1 \text{ byte} \times 10^9 = 1 \text{ GB}$$

This only describes how much memory is needed to run the model, known as inference. Training needs much more than this and is usually at least 3 times more than that of the inference memory. The training memory is linearly dependent on the number of parameters and the number of intermediate variables and consists of weights, gradients, and the optimizer state.

## Challenges of Transformers

Training stability is a major challenge of some transformers, and the breakdown comes to pre-norm and post-norm, which are two different methods of applying normalization. In the original "Attention is All You Need" the normalization is performed after the addition; however, later models such as GPT-2 normalize first and then run the layer and add. Post-norm is unstable because the layer normalization wraps around the shortcut for the gradient problem. This makes x not pass through cleanly and forces it to be renormalized at each layer. Since at the start all the weights are random, the model is very wrong and gradients are very large. Since backpropagation makes the gradient go through each layer, the ones at the output are very large and make it uneven. The solution is to slowly scale up the learning rate from something small; this makes it so that at the start the gradients and weights are still very wrong, but they won't make massive jumps into another region that may be totally wrong. Pre-norm also works because the gradients are not forced to go through each layer since the residual path skips the LayerNorm; the final output here is simply each layer's contribution plus the input.

What makes a transformer better than an RNN?

A transformer is faster because it computes all positions in parallel, unlike an RNN, which has to process tokens one after another. In a transformer, it is also simple to tie one token to any other, but in an RNN the information must go through many steps, and it fades with each step in the process. This also ties into memory, which in a transformer holds each token's representation. In an RNN, it only holds what's able to fit in the hidden state. This is why there is no vanishing gradient issue for the transformer: it stores all the tokens in memory, so gradients aren't vanishing. In a transformer, gradients also aren't vanishing because there are identity shortcuts that allow for skips. The advantage of RNN, however, is that it is linear in its computation cost, whereas the transformer is quadratic. This is the same for memory, where RNN is linear in terms of sequence length and a transformer is quadratic.

## Applications

The following image shows how a transformer is used. A source sentence is fed into it, and then embeddings and projections turn this sentence into a vector describing position information and the meaning of words. This is then repeated N times, with the output serving as the next input, and is normalized. Then the decoder on the right writes the output. The decoder is given all but the last word, and using the tokens and positional encodings, it is asked to predict what that word may be. It then looks at what words it has written so far and predicts the next word again and continues.

<img width="531" height="767" alt="A diagram of a standard transformer architecture" src="https://github.com/user-attachments/assets/b6822b8b-fc56-4214-a60f-dc51a3c533e6" />

*Source: Kai Wang, Lecture 19: Transformers*

The earlier explanation makes it seem like a pretrained model can only be used for generation, and while it can be, this isn't all that it is good for. It can also be used to extract features! A sequence can be fed in, such as a picture, and it can be run through the transformer and stopped at the last layer, outputting the vector instead of generating another vector with the decoder. This can be used to compare features and extract what is important. For example, on social media, banning certain language detection can come through looking at vector outputs from a pretrained model.

Transformers can also be used for multimodal training, such as text-to-image generation with CLIP. A transformer can take in some text, turn it into some vector, and then, by training another encoder on the image the caption is attached to, this encoder can make each image hold some vector of values. This can then be compared, and weights can be modified to fit it to the text encoder's vector. Essentially, compare the two encoder vectors and pull similarities together while forcing differences to be further apart.

## Vision Transformers (ViT)

This is when the transformer is applied to images to extract certain features. The idea is as follows: split the images up into multiple patches and have each patch be a token. Add positional embeddings so that each token has a known location, and then run a transformer model that looks at dependencies between patches and can classify what it sees. This is an application of computer vision, because using vectors and numbers, it allows a computer to "see" a picture and classify what it is as a collection of vectors. Further applications can even have a computer classify and label patches if it is trained to do so through the changing of its weights.

<img width="595" height="556" alt="An drawn diagram of a vision transformers architecture" src="https://github.com/user-attachments/assets/c5abe79a-2849-4192-9ee4-bbabc0966e76" />

*Source: Kai Wang, Lecture 19: Transformers*

## References

CSE 6740, Lecture 19: Transformers, slides 18–36 (Fall 2026). [Lecture 19 slide deck](/assets/semesters/fall-2026/lectures/19/lecture-19-transformers.pdf)

Vaswani et al., "Attention Is All You Need," NeurIPS 2017. https://arxiv.org/abs/1706.03762

Dive into Deep Learning, "Multi-Head Attention." https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html

GeeksforGeeks, "Multi-Head Attention Mechanism." https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/

A. Karpathy, minGPT model.py. https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

A. Karpathy, "Let's build GPT: from scratch, in code, spelled out." https://www.youtube.com/watch?v=kCc8FmEb1nY

GeeksforGeeks, "Residual Networks (ResNet) – Deep Learning." https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/

X. Xia, "RNN vs. Transformer: A Deep Dive from Fundamentals to Applications," Medium. https://medium.com/@xiaxiami/rnn-vs-transformer-a-deep-dive-from-fundamentals-to-applications-ee4d700dd152

GeeksforGeeks, "RNN vs LSTM vs GRU vs Transformers." https://www.geeksforgeeks.org/deep-learning/rnn-vs-lstm-vs-gru-vs-transformers/

Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021. https://arxiv.org/abs/2103.00020

Ramesh et al., "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL·E 2), 2022. https://arxiv.org/abs/2204.06125

Stanford HAI, "What is a Vision Transformer (ViT)?" https://hai.stanford.edu/ai-definitions/what-is-a-vit

GeeksforGeeks, "Vision Transformer (ViT) Architecture." https://www.geeksforgeeks.org/deep-learning/vision-transformer-vit-architecture/

X. Xia, "A Deep Dive into Vision Transformers (ViT): Concepts, Fundamentals, Methods, and Applications," Medium. https://medium.com/@xiaxiami/a-deep-dive-into-vision-transformers-vit-concepts-fundamentals-methods-and-applications-52efc5e8e452

I also used Claude to talk through and digest research that I did to make sure I had not misunderstood anything I read in research papers. Everything written is in my own words and comes from information from the slides and the cited references.
