---
semester: fall-2026
lecture_number: "18"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: YihuayangZ
issue: 62
---

## 1. Why Do We Need Recurrent Neural Networks? (Slides 1–5)

### 1.1 Where This Lecture Starts

Lecture 18 introduces sequence data, recurrent neural networks (RNNs), long short-term memory (LSTM), and other recurrent architectures. These notes cover slides 1–18, ending with two ways to make one prediction for an entire sequence. [1]

For an ordinary prediction task, we can feed a feature vector $$x$$ into a network and get a prediction $$\hat{y}$$. When each sample can be handled on its own, the model can process the samples in any order. What happens when the order is part of the data? **How can the network use what it has already seen to make sense of the next input?**

An RNN carries a hidden state from one position to the next. It applies the same update rule each time, so an earlier input can still affect what the model does later in the sequence. [1]

### 1.2 A Little Background

The timeline on slide 3 places RNNs and LSTMs among several approaches to machine learning. An SVM learns a decision boundary, while a hidden Markov model describes a sequence through underlying states. A CNN looks for local patterns, such as nearby pixels in an image. An RNN carries a hidden state forward through the sequence, and a Transformer uses attention to relate positions. The timeline gives context for the models discussed in class. [1]

### 1.3 What Counts as Sequence Data?

Words in a sentence, prices over time, and notes in a song all arrive in an order. We can write a sequence of length $$T$$ as

$$
(x_1,x_2,\ldots,x_T),
\qquad x_t \in \mathbb{R}^d,\quad t=1,2,\ldots,T.
$$

Here, $$t$$ is the **time step** or sequence position, $$T$$ is the **sequence length**, and $$d$$ is the **number of features at each step**. For example, suppose we record the closing price, trading volume, and intraday volatility of a stock for 30 days. Then $$T=30$$ and $$d=3$$. Some tasks have a label $$y_t$$ at each step, giving a sequence of labels $$(y_1,\ldots,y_T)$$. Other tasks have one label for the whole sequence. [1]

“I am happy” and “I am **not** happy” differ by one word, yet that word can flip the sentiment. A model that only counts which words appear can miss how they work together. An RNN gives earlier inputs a way to influence how later inputs are processed.

## 2. How Does an RNN Read a Sequence? (Slides 6–10)

### 2.1 Carrying a State Forward

Start with the single block on slide 6. The block receives the first input $$x_1$$ and the initial hidden state $$h_0$$. It computes a new state $$h_1$$, which can be used to produce the first output $$y_1$$. The initial state is often a zero vector. In PyTorch, this is the default when we omit it. [1, 2]

I think of $$h_t$$ as the model's working summary of what it has read through step $$t$$. After “I am not ...,” for example, the state can already carry information about the negation before the next word arrives. How much of the earlier input survives depends on what the model learns to keep in that state.

### 2.2 From Words to Input Vectors

The translation example on slide 7 shows “How are you” alongside “Cómo estás.” Each input to the network is a vector. For text, we split the sentence into **tokens** such as words or subwords, give each token an integer ID, and look up a vector called its **embedding**. For illustration, *How* could map to $$x_1=(0.2,0.5,-0.1)^\top$$ and *are* to $$x_2=(-0.3,0.8,0.4)^\top$$. These numbers are just made up for the example. PyTorch's `nn.Embedding` performs the lookup. [1, 3]

At the first step, the model combines $$x_1$$ and $$h_0$$ to obtain $$h_1$$. At the next step, it combines $$x_2$$ and $$h_1$$ to obtain $$h_2$$. The general update is

$$
h_t=f_W(h_{t-1},x_t).
$$

The same function $$f_W$$, with the same weights $$W$$, is used at every step. This is **parameter sharing**. The model needs only one set of recurrent weights, so it can apply the same rule to sequences of different lengths. [1, 4]

### 2.3 Four Input and Output Patterns

Slide 8 counts inputs and outputs at the level of the whole task. [1]

| Pattern | What goes in and what comes out | Example |
| --- | --- | --- |
| One-to-one | One input, one output | Classify one image. |
| Many-to-one | A sequence of inputs, one output | Predict the sentiment of an entire sentence. |
| One-to-many | One initial input, a sequence of outputs | Generate a sequence from a starting condition. |
| Many-to-many | A sequence of inputs, a sequence of outputs | Translate a sentence or tag each word. |

For sentiment analysis, we might read “The movie was surprisingly good” and predict only one label, positive or negative. In part-of-speech tagging, we want one label per token. Translation also has sequences on both sides, and their lengths can differ.

### 2.4 Reading the Unrolled Diagram

The diagram on slide shows six copies of the same `NN` block, with inputs $$x_1,\ldots,x_6$$ below and outputs $$y_1,\ldots,y_6$$ above. The horizontal arrows carry state from one step to the next. Follow the arrows from $$h_0$$ and you can see which information is available when each block runs. This view of the same cell at successive time steps is called **unrolling**. [1, 4]

## 3. The Forward Pass (Slides 11–14)

### 3.1 Updating the Hidden State

The recurrent cell on slide 11 takes the previous hidden state and the current input. In a vanilla RNN with a tanh activation, the update is

$$
h_t=\tanh\left(W_x x_t+W_h h_{t-1}+b_h\right).
$$

If the input has dimension $$d$$ and the hidden state has dimension $$m$$, then $$W_x\in\mathbb{R}^{m\times d}$$, $$W_h\in\mathbb{R}^{m\times m}$$, and $$b_h\in\mathbb{R}^m$$. The term $$W_x x_t$$ brings in the new input, $$W_h h_{t-1}$$ brings in the previous state, and $$b_h$$ is the bias. Tanh is applied to each component of the sum. [1]

In “How are you,” the state at *are* depends on both the embedding of *are* and the state left by *How*. By the time the model reads *you*, the new state can reflect information from all three words.

For a single-layer, unidirectional `nn.RNN` with `nonlinearity="tanh"`, PyTorch writes the update with two bias terms. [2]

$$
h_t=\tanh\left(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh}\right).
$$

The two biases add up to the single $$b_h$$ above. In PyTorch, `input_size` is $$d$$, `hidden_size` is $$m$$, and `num_layers` is the number of recurrent layers stacked on top of one another.

### 3.2 Turning a State into an Output

Slide 12 uses a linear output layer. Once the cell has computed $$h_t$$, the output is

$$
y_t=f_{W_y}(h_t)=W_yh_t+b_y.
$$

For a scalar output, $$W_y\in\mathbb{R}^{1\times m}$$ and $$b_y\in\mathbb{R}$$. For classification with $$K$$ classes, we can use $$W_y\in\mathbb{R}^{K\times m}$$ and $$b_y\in\mathbb{R}^K$$ to produce $$K$$ logits. A softmax can turn those logits into class probabilities. The hidden state carries information forward, while the output layer reads that state to make a task-specific prediction. [1]

$$
(x_t,h_{t-1})\longrightarrow h_t\longrightarrow y_t.
$$

### 3.3 Repeating the Same Computation

Slide 13 repeats the update and the output mapping for each new input. In the many-to-many graph shown there, each step produces an output. The first three state updates, written out, are

$$
\begin{aligned}
h_1 &= \tanh\left(W_xx_1+W_hh_0+b_h\right),\\
h_2 &= \tanh\left(W_xx_2+W_hh_1+b_h\right),\\
h_3 &= \tanh\left(W_xx_3+W_hh_2+b_h\right).
\end{aligned}
$$

For example, $$x_1$$ can affect the third output through the path $$h_1\to h_2\to h_3\to y_3$$. The carried state can compress information from the earlier inputs. [1, 4]

With `batch_first=False`, a batched, unidirectional `nn.RNN` takes an input of shape `(T, B, d)`, where `B` is the batch size. Its `output` has shape `(T, B, m)` and contains the last layer's hidden state at **every time step**. Its `h_n` has shape `(num_layers, B, m)` and contains the **final hidden state of each layer**. To obtain the lecture's predictions $$y_t$$, we still apply the task's output layer to the appropriate hidden states. [2]

### 3.4 Adding a Training Loss

In slide 14, the predictions feed into a single loss node. Using $$y_t^*$$ for the ground-truth label at step $$t$$, we can write

$$
L = \mathrm{loss}((y_{1}, \ldots, y_{T}), (y_{1}^{\ast}, \ldots, y_{T}^{\ast})).
$$

One common choice is to add the losses at each step, $$L=\sum_{t=1}^T\ell_t$$. Either way, the scalar $$L$$ measures how far the sequence predictions are from their targets and gives us an objective for training. [1]

## 4. The Computation Graph and Backpropagation Through Time (Slides 15–16)

### 4.1 Why Does a Weight Receive Gradients from Several Steps?

The trainable parameters in this RNN are $$W_x,W_h,W_y,b_h,b_y$$. Each recurrent step uses the same $$W_x,W_h,b_h$$, and each prediction in the graph on slide 15 uses the same $$W_y,b_y$$. The loss receives all the predictions, so we add the contributions along the paths from those predictions back to a parameter. This is **backpropagation through time** (BPTT). [1, 4]

Take $$w_h=\mathrm{vec}(W_h)$$, the entries of the recurrent weight matrix written as a vector. For fixed inputs, its effect on the loss runs through the hidden states:

$$
\frac{dL}{dw_h}
=\sum_{t=1}^{T}
  \frac{\partial L}{\partial y_t}
  \frac{\partial y_t}{\partial h_t}
  \frac{dh_t}{dw_h}.
$$

There are two ways for $$w_h$$ to affect $$h_t$$. It appears in the current update, and it also helped compute the previous state $$h_{t-1}$$. Let $$D_t=dh_t/dw_h$$, let $$A_t$$ be the partial derivative of $$h_t$$ with respect to $$w_h$$ with $$h_{t-1}$$ held fixed, and let $$J_t=\partial h_t/\partial h_{t-1}$$. The chain rule gives

$$
\begin{aligned}
D_t &= A_t+J_tD_{t-1}\\
    &= A_t+J_tA_{t-1}+J_tJ_{t-1}D_{t-2} \qquad (t\geq 2).
\end{aligned}
$$

The second line shows how an earlier use of the same weight contributes to the gradient at the current step. A fixed initial state gives $$D_0=0$$. For the output weight $$W_y$$, the path instead goes directly through $$y_t=W_yh_t+b_y$$, since the hidden states in this graph are computed without the output layer. [1, 4]

### 4.2 Why Can Gradients Vanish or Explode?

To see what happens over many time steps, write the pre-activation as $$a_t=W_x x_t+W_hh_{t-1}+b_h$$, so $$h_t=\tanh(a_t)$$. Because tanh acts component by component, the Jacobian is

$$
J_t=\frac{\partial h_t}{\partial h_{t-1}}
=\mathrm{diag}\left(1-h_t^2\right)W_h,
$$

where the square in $$h_t^2$$ is elementwise. The derivative from an earlier state $$h_s$$ to a later state $$h_t$$ is an ordered matrix product:

$$
\frac{\partial h_t}{\partial h_s}
=J_tJ_{t-1}\cdots J_{s+1},
\qquad 0\leq s<t.
$$

If those factors keep shrinking a signal as it travels backward, the gradient reaching an early step can be very small. This is the **vanishing gradient** problem. If the factors repeatedly amplify it along some directions, the gradient can grow quickly. This is the **exploding gradient** problem. An early word can matter to the final prediction, yet training may give the model very little signal about how to use that word. [1, 4]

## 5. Many-to-One Computation Graphs (Slides 17–18)

### 5.1 Predicting from the Last State

In slide 17, the model reads the whole sequence and makes one prediction at the last step. For example, it can read “The movie is very good” word by word and predict one sentiment for the sentence. The earlier steps still compute and pass along their states. The prediction is read only from the final state $$h_T$$. [1]

$$
\hat{y}=g(h_T),
\qquad \text{for example, } \hat{y}=W_yh_T+b_y.
$$

### 5.2 Giving the Output Direct Access to Earlier States

Slide 18 keeps the same recurrent chain and the same single output. The dashed arrows give that output direct access to the earlier hidden states. In the four-step diagram, the final output $$y_{t+3}$$ can read $$h_t$$, $$h_{t+1}$$, and $$h_{t+2}$$ in addition to the final state. For a sequence of length $$T$$, that pattern can be written as

$$
\hat{y}=g(h_1,h_2,\ldots,h_T).
$$

In slide 17, an important early word has to affect the prediction through $$h_T$$. In slide 18, its earlier hidden state has an additional path to the output. Both graphs still produce one prediction for the whole sequence. [1]

### 5.3 A Small PyTorch Example

This example uses either the last hidden state or the mean of all hidden states before applying the classifier. Taking the mean is one possible way to use the extra connections in slide 18. The example assumes an unpadded batch, with `token_ids` shaped `(T, B)` and every sequence in the batch having length `T`. [1–3]

```python
from torch import nn


class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=16, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=False)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids, use_all_states=False):
        x = self.embed(token_ids)      # (T, B, embed_dim)
        out, hidden = self.rnn(x)      # out: (T, B, hidden_dim)
        if use_all_states:
            summary = out.mean(dim=0)  # Mean over time (one option for slide 18).
        else:
            summary = hidden[-1]       # Final state of the last layer, as in slide 17.
        logits = self.fc(summary)      # (B, num_classes)
        return logits
```

## References

1. Kai Wang. [CSE/ISyE 6740, Lecture 18: Recurrent Neural Networks](https://github.com/GT-KOALA/Gatech-CSE6740/blob/main/assets/semesters/fall-2026/lectures/18/lecture-18-rnn-lstm.pdf). Georgia Institute of Technology, November 4, 2026, slides 1–18.
2. PyTorch documentation. [torch.nn.RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html).
3. PyTorch documentation. [torch.nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html).
4. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. [Deep Learning, Chapter 10: Sequence Modeling](https://www.deeplearningbook.org/contents/rnn.html). MIT Press, 2016.

I wrote the initial draft and used ChatGPT to help with Markdown formatting, formatting the equations, and checking the math and PyTorch example.
