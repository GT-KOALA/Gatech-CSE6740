---
semester: fall-2026
lecture_number: "11"
slot: course-1
slot_order: 1
role: course_material
role_label: Course materials
tab_title: Core explanation I
assignee: User12ss
issue: 33
---
## What these notes cover

This  is the note on Lecture 11, *Neural Networks*, slides 1–22. Slides 1–10 explain why we need more than linear models. Most of the notes are about the perceptron (slides 11–21): what one unit computes, how it is trained, what it cannot do, and why connecting a few units fixes that.

**Notation I use for this note.** Example $i$ is an input $x^i \in \mathbb{R}^d$ with label $y^i$; the superscript is the example index, not a power. The weights $w \in \mathbb{R}^d$ give the weighted sum $w^\top x = w_1 x_1 + \dots + w_d x_d$.

## 1. Why linear models are not enough

The lecture uses three pictures to tell why some of data are linear separable while some other are not: data that one line separates, the XOR gate (diagonally opposite corners of the unit square share a label), and speech recognition in which the data from each class is mixed up so it is hard to tell the difference. The last two cannot be split by one straight line.

Linear regression, logistic regression and the SVM all score an input with $f(x) = w^\top x$. They differ mainly in the loss: $(y - f(x))^2$, $\log(1 + e^{-y f(x)})$ and $\max(0, 1 - y f(x))$ respectively. For linear regression, $y$ can be any real number; logistic regression and the SVM use class labels $y \in \lbrace -1, +1 \rbrace$. If we wants to use them as classifiers, all three can predict from the sign of $w^\top x$, so the boundary is where $w^\top x = 0$. In 2-D, with a constant input of 1 for the intercept, that is the straight line $a x_1 + b x_2 + c = 0$. Training only picks $a, b, c$, so no loss can make the line bend.

The fix on the same slide is to transform the input first:

$$
f(x) = w^\top \phi(x)
$$

This is still linear in $w$, but not in $x$. For example, take $\phi(x) = (1, x_1, x_2, x_1 x_2)$ (I chose $w$ so that $f$ equals the label at each corner in XOR gate: $(0,0)$ gives $w_0=1$; $(0,1)$ gives $w_0+w_2=0$, so $w_2=-1$; $(1,0)$ gives $w_0+w_1=0$, so $w_1=-1$; and $(1,1)$ gives $w_0+w_1+w_2+w_3=1$, so $w_3=2$.) and $w = (1, -1, -1, 2)$, so $f(x) = 1 - x_1 - x_2 + 2 x_1 x_2$.

- On the four corners $(0,0), (0,1), (1,0), (1,1)$, this gives $1, 0, 0, 1$. These answers are exactly the lecture's XOR-style labels.
- The boundary $f(x) = 0.5$ (the threshold to decide whether predict label 1/0) is $1-x_1-x_2+2x_1x_2=0.5$ $\to$ $(2x_1 - 1)(2x_2 - 1) = 0$, i.e. the two lines $x_1 = 0.5$ and $x_2 = 0.5$, there will be 2 lines that are used to separate two classes of data.

Kernel methods fix $\phi$ in advance and only ever compute $\phi(x)^\top \phi(x^{\prime})$. A neural network instead **learns $\phi$ together with $w$**. That is the idea the rest of the lecture builds.

## 2. Motivation in brief

- **Function approximators.** Any 1-D function can be approximated by a staircase (piecewise constant) or by connected line segments (piecewise linear). On a grid with $k$ pieces per input axis, though, this needs about $k^d$ numbers. With $k = 10$ that is 100 numbers for $d = 2$ and $10^{10}$ for $d = 10$. That is the slide's "number of parameters explodes".
- **ImageNet.** This task is to classify about 1.3 million color images into 1,000 classes. The 8-layer network of Krizhevsky et al. (2012) had 5 convolutional layers, 2 fully connected layers and a final multiclass logistic regression. It reached 15.3% top-5 error, against 26.2% for second place (top-5 error is the fraction of images whose true label is not among the five top guesses).[2] The first seven layers turn the image into a feature vector $\phi(x)$, and the last layer is logistic regression on $\phi(x)$. Because all layers are trained together, $\phi$ is learned rather than fixed in advance.
- **Brains.** A neuron switches in about 0.001 s, yet recognizing a scene takes only about 0.1 s. Each step in a chain has to wait for the result of the previous one, so there is a capacity for having at most $0.1 / 0.001 = 100$ steps continuously. Recognizing a scene is far too complex to finish in 100 steps, so the brain must do this job in parallel, it has about $10^{10}$ neurons, each connected to $10^4$ to $10^5$ others, which is roughly $10^{10} \times 10^4 = 10^{14}$ to $10^{10} \times 10^5 = 10^{15}$ connections in total. My takeaway is that the brain's computation process is short (few steps in sequence) but very wide (massively parallel). This is what the perceptron in Section wants to take advantages of . Its inputs play the role of incoming signals, its weights play the role of connection strengths, and its threshold plays the role of the minimum total input a neuron needs before it passes a signal on: the unit outputs 1 once its weighted sum reaches the threshold, and 0 otherwise. A network stacks many of these simple units, and all units in a layer compute at the same time. What it borrows is this structure, not the actual biology of real neurons.

## 3. The perceptron: what one artificial neuron computes

A **perceptron** does two things in a row:

1. **Linear combiner:** add up the weighted inputs, $u = w^\top x = w_1 x_1 + w_2 x_2$.
2. **Nonlinear transform:** pass $u$ through a function $f$ and compare with a threshold $\theta$. The output $Y$ is 1 if $f(u) \ge \theta$, and 0 otherwise.

For example, using the weights from the lecture's example in page 18: $w = (0.2, 0.35)$, $f(u) = u$ on $[0, 1]$, and $\theta = 0.5$. For input $x = (1, 1)$:

$$
u = 0.2 \cdot 1 + 0.35 \cdot 1 = 0.55, \qquad f(0.55) = 0.55 \ge 0.5 \ \Rightarrow\ Y = 1
$$

**Writing the threshold as a weight.** In this example $f(u) = 0$ for $u < 0$, $f(u) = u$ for $0 \le u \le 1$, and $f(u) = 1$ for $u > 1$, so $f(u) \ge 0.5$ happens exactly when $u \ge 0.5$. The rule is therefore "output 1 when $w_1 x_1 + w_2 x_2 \ge \theta$". Subtracting $\theta$ from both sides gives $w_1 x_1 + w_2 x_2 - \theta \ge 0$. Now add an extra input $x_0$ that is always 1, with weight $w_0 = -\theta$. Then $w_0 x_0 = -\theta$, and the rule becomes

$$
w_0 x_0 + w_1 x_1 + w_2 x_2 \ge 0
$$

With the numbers above, $w_0 = -0.5$. Input $(1,1)$ gives $-0.5 + 0.55 = 0.05 \ge 0$, so $Y = 1$; input $(0,1)$ gives $-0.5 + 0.35 = -0.15 < 0$, so $Y = 0$. These are the same answers as before. The extra weight $w_0$ is usually called the **bias**. Under this format, every unit compares with 0, and $\theta$ is trained like any other weight. 

## 4. Four choices for the nonlinear transform

![Four side-by-side plots of output against the weighted sum u, for u from minus 4 to 4. Hard limiter: a jump from 0 to 1 at u equals 0. Piecewise linear: 0 for negative u, rising in a straight line to 1 at u equals 1, then flat. Sigmoid: a smooth S-curve from 0 to 1 through 0.5 at u equals 0. Hyperbolic tangent: a steeper S-curve from minus 1 to 1 through 0 at u equals 0.](course-1-images/fig2-activation-functions.png)

*Figure 1. The four transforms on the "Other Nonlinear Neurons" slide.*

| Transform | $f(u)$ | $u = -2$ | $u = 0$ | $u = 0.5$ | $u = 2$ |
|---|---|---|---|---|---|
| Hard limiter | 1 if $u \ge 0$, else 0 | 0 | 1 | 1 | 1 |
| Piecewise linear | $\min(\max(u, 0), 1)$ | 0 | 0 | 0.5 | 1 |
| Sigmoid | $\sigma(u) = 1 / (1 + e^{-u})$ | 0.119 | 0.5 | 0.622 | 0.881 |
| Hyperbolic tangent | $\tanh(u)$ | $-0.964$ | 0 | 0.462 | 0.964 |

There are two properties that are shared by all transforms:

- **All four are non-decreasing**: if $u_1 < u_2$ then $f(u_1) \le f(u_2)$. Section 7 shows that this alone forces a single unit to draw a straight boundary.
- **Only the transform with smooth curve can be trained by gradient descent.** The hard limiter's slope is 0 everywhere except at the jump, so the gradient doesn't know towards which way to move the weights. The sigmoid and tanh have a nonzero slope everywhere.

## 5. Logistic regression is a perceptron with a sigmoid

Logistic regression, with labels $y \in \lbrace 0, 1 \rbrace$, assumes

$$
p(y = 1 \mid x, w) = \sigma(w^\top x) = \frac{1}{1 + \exp(-w^\top x)}
$$

This is exactly a perceptron whose transform is the sigmoid. The slide maps the vocabulary: independent variables are the inputs, coefficients are the weights, and the dependent variable is the output.

The example is: age 34, gender 1 and stage 4, with weights 5, 4 and 8, give $u = 34 \cdot 5 + 1 \cdot 4 + 4 \cdot 8 = 170 + 4 + 32 = 206$, so the "probability of surviving" is $\sigma(206) \approx 1$. 

**Derivative of the sigmoid.** The training rule in the next section needs it. The slide only states the result, so here are the steps:

1. Write $\sigma(u) = (1 + e^{-u})^{-1}$.
2. The inner part has derivative $\frac{d}{du}(1 + e^{-u}) = -e^{-u}$.
3. By the chain rule, $\frac{d}{du}(g)^{-1} = -g^{-2} \frac{dg}{du}$. So $\frac{d\sigma}{du} = -(1 + e^{-u})^{-2} \cdot (-e^{-u}) = \frac{e^{-u}}{(1 + e^{-u})^2}$.
4. Split the fraction: $\frac{e^{-u}}{(1 + e^{-u})^2} = \frac{1}{1 + e^{-u}} \cdot \frac{e^{-u}}{1 + e^{-u}}$.
5. The first factor is $\sigma(u)$. The second is $1 - \sigma(u)$, because $1 - \frac{1}{1 + e^{-u}} = \frac{1 + e^{-u} - 1}{1 + e^{-u}} = \frac{e^{-u}}{1 + e^{-u}}$.

$$
\frac{d\sigma(u)}{du} = \sigma(u)\left(1 - \sigma(u)\right)
$$

As a check, at $u = 0$ both step 3 and the final formula give $0.25$. At $u = 4$ the slope is only $0.982 \times 0.018 \approx 0.018$: once a sigmoid is far into its flat part, changing the weights barely changes its output.

(The slide writes "$\sigma(x) = 1/(1 + \exp(-u))$"; the left side should be $\sigma(u)$.)

**Link to the logistic loss in Section 1** (this confused me at first). With labels $\pm 1$, the probability the model gives to the true label is $\sigma(z)$ with $z = y w^\top x$; for $y = -1$ this uses $\sigma(-u) = 1 - \sigma(u)$. The logistic loss is its negative log: $-\log \frac{1}{1 + e^{-z}} = \log(1 + e^{-z})$, because $-\log(1/a) = \log a$. So $1 + e^{-z}$ is the denominator of $\sigma$, not $\sigma$ itself.

## 6. Training one sigmoid unit with the square loss

- **Model:** the unit outputs $\sigma(w^\top x)$, read as $P(y = 1 \mid x, w)$.
- **Objective:** make those probabilities close to the labels. The labels may be values other than 0 or 1, for example 0.7 if 70% of annotators said yes.

$$
\min_w\ l(w) = \sum_{i=1}^{n} \left(y^i - \sigma(w^\top x^i)\right)^2
$$

- **Optimizer:** gradient descent. The slide warns that the objective is **not convex**, so gradient descent is only guaranteed to reach a local optimum.

**Why it is not convex.** A convex function satisfies, for any two points $a$ and $b$, "value at the midpoint $\le$ average of the values at $a$ and $b$". One violation is enough to show a function is not convex. Take one training example with $x = 1$, $y = 1$, so $l(w) = (1 - \sigma(w))^2$:

| $w$ | $\sigma(w)$ | $l(w)$ |
|---|---|---|
| $-4$ | 0.018 | 0.964 |
| $-2$ | 0.119 | 0.776 |
| $0$ | 0.5 | 0.250 |

The average at the two ends is $(0.964 + 0.250)/2 = 0.607$, but the value at the midpoint $w = -2$ is $0.776 > 0.607$. Not convex.

![Two side-by-side plots of the loss on one example as the weight w runs from minus 6 to 6. Left, square loss: flat near 1 for very negative w, then an S-shaped drop to 0; the region left of w equals minus ln 2 is shaded and labelled concave, so not convex. Right, logistic loss: falls along a straight line from about 6 and flattens toward 0 for positive w.](course-1-images/fig3-square-vs-logistic-loss.png)

*Figure 2. Left: the square loss $(1 - \sigma(w))^2$ from the table above. Between $w = -4$ and $w = 0$ the curve is above the straight line formed by connecting those two points, so it is not convex. Right: for comparison, the logistic loss $-\log \sigma(w)$ on the same example, which is convex.*

**Where the shaded region starts.** The dashed line in the left panel of Figure 2 marks where the curve switches from bending down to bending up, i.e. where the second derivative of $l(w) = (1 - \sigma(w))^2$ changes sign.

1. Since $1 - \frac{1}{1 + e^{-w}} = \frac{e^{-w}}{1 + e^{-w}} = \frac{1}{e^{w} + 1}$, we have $1 - \sigma(w) = \sigma(-w)$. Write $s = \sigma(-w)$, so $l(w) = s^2$.
2. By the chain rule and the sigmoid derivative from Section 5 (the inner function $-w$ has derivative $-1$), $\frac{ds}{dw} = -s(1 - s)$.
3. First derivative: $\frac{dl}{dw} = 2s \frac{ds}{dw} = -2s^2 + 2s^3$.
4. Second derivative: $\frac{d^2 l}{dw^2} = (-4s + 6s^2) \frac{ds}{dw} = (-4s + 6s^2)(-s(1 - s)) = s^2 (1 - s)(4 - 6s)$.
5. Since $0 < s < 1$, the factors $s^2$ and $1 - s$ are positive, so the second derivative has the sign of $4 - 6s$. The curve bends down exactly when $s > 2/3$.
6. $\sigma(-w) > 2/3$ means $\frac{1}{1 + e^{w}} > \frac{2}{3}$. Multiplying both sides by the positive number $3(1 + e^{w})$ gives $3 > 2 + 2e^{w}$, which is $e^{w} < \frac{1}{2}$.
7. Because $e^{w}$ increases with $w$, this holds exactly when $w < \ln \frac{1}{2} = -\ln 2 \approx -0.693$.

So the square loss is not convex (is concave) for $w < -\ln 2$ and until for $w > -\ln 2$. A convex function can never bend down, which is, for a function with a second derivative, convex means $\frac{d^2 l}{dw^2} \ge 0$ at every $w$, where 0 (a straight piece) is allowed.

The figure shows a second problem. When the unit is badly wrong ($w$ very negative), the square loss goes flat near 1, so gradient descent barely moves. The logistic loss keeps a slope near $-1$ there and is convex, which is why logistic regression is trained with it instead.

**The gradient, step by step.** Write $u^i = w^\top x^i$ (a number).

1. The derivative of a sum is the sum of the derivatives, so take one term $(y^i - \sigma(u^i))^2$.
2. Chain rule on the square: $\frac{\partial}{\partial w}(y^i - \sigma(u^i))^2 = 2(y^i - \sigma(u^i)) \cdot \frac{\partial}{\partial w}(y^i - \sigma(u^i))$.
3. The label $y^i$ does not depend on $w$, so $\frac{\partial}{\partial w}(y^i - \sigma(u^i)) = -\frac{\partial \sigma(u^i)}{\partial w}$.
4. Chain rule again, using Section 5 and $\frac{\partial (w^\top x^i)}{\partial w} = x^i$: $\frac{\partial \sigma(u^i)}{\partial w} = \sigma(u^i)(1 - \sigma(u^i)) x^i$.
5. Put steps 2–4 together and sum over $i$:

$$
\frac{\partial l(w)}{\partial w} = -\sum_{i=1}^{n} 2\left(y^i - \sigma(u^i)\right)\sigma(u^i)\left(1 - \sigma(u^i)\right) x^i
$$

Each term is a number times $x^i \in \mathbb{R}^d$, so the gradient has the same shape as $w$.

**The slide's formula maybe missing the leading minus sign(slide page 17).** The minus sign comes from step 3. Gradient descent will moves **against** the gradient: $w \leftarrow w - \eta \frac{\partial l}{\partial w}$. One step with $x = 1$, $y = 1$, starting at $w = 0$ with step size $\eta = 1$:

- $\sigma(0) = 0.5$, so the loss is $0.25$ and the gradient is $-2(1 - 0.5)(0.5)(1 - 0.5)(1) = -0.25$.
- Correct sign: $w = 0 - 1 \cdot (-0.25) = 0.25$. The loss drops to $(1 - \sigma(0.25))^2 = 0.192$.
- Slide's sign: $w = -0.25$. The loss rises to $(1 - \sigma(-0.25))^2 = 0.316$.

**How I read the formula.** Plugging the gradient into $w \leftarrow w - \eta \frac{\partial l}{\partial w}$, the two minus signs cancel out and the update becomes:

$$
w \leftarrow w + \sum_{i=1}^{n} c^i x^i, \qquad c^i = 2\eta \left(y^i - \sigma(u^i)\right) \sigma(u^i) \left(1 - \sigma(u^i)\right)
$$

Here an "example" is one training example $(x^i, y^i)$, which is one single term of the sum. Each example adds $c^i x^i$ to $w$: every weight sub-element $w_j$ changes by the number $c^i$ times that example's input $x^i_j$. So weights on large inputs move more, and weights on inputs equal to 0 do not move at all.

- **Direction.** $\eta$, 2 and $\sigma(u^i)(1 - \sigma(u^i))$ are all positive (the sigmoid's output is set strictly between 0 and 1), so $c^i$ has the same sign as the error $y^i - \sigma(u^i)$. If the output $\sigma(u^i)$ is below the label $y^i$, $c^i > 0$ and $w$ moves along $x^i$; if it is above $y^i$, $c^i < 0$ and $w$ moves against $x^i$. Its push alone changes $u^i = w^\top x^i$ by $c^i \lVert x^i \rVert^2$, where $\lVert x^i \rVert^2 = (x^i_1)^2 + \dots + (x^i_d)^2$. (Each weight $w_j$ changes by $c^i x^i_j$ and is multiplied by $x^i_j$ inside $u^i$, which gives $c^i (x^i_j)^2$.) A sum of squares is never negative, so the change has the sign of $c^i$.
- **Strength.** The size of $c^i$ is (error) x (sigmoid slope), times $2\eta$. If $u^i$ has already entered to the flat region of the sigmoid, the slope is near 0, so even a badly wrong example barely moves $w$. With $y^i = 1$ and $u^i = -4$ the output score is only $0.018$, with $c^i / \eta \approx 0.035$ ($2 \times (error) \times (sigmoid slope)$ is $c^i / \eta$, this is the part that decide strength and learning rate is manually set by us and it is fixed for all examples.), compared with $0.25$ at $u^i = 0$.

**Example.** Take one training example with $x^i = (1, 2)$, starting from $w = (0, 0)$ with $\eta = 1$. Then $u^i = 0$, the output is $\sigma(0) = 0.5$, the slope is $0.5 \times 0.5 = 0.25$, and $\lVert x^i \rVert^2 = 1^2 + 2^2 = 5$.

| Step | Label $y^i = 1$ | Label $y^i = 0$ |
|---|---|---|
| Error $y^i - \sigma(u^i)$ | $1 - 0.5 = 0.5$: output below the label | $0 - 0.5 = -0.5$: output above the label |
| $c^i = 2\eta \times \text{error} \times 0.25$ | $0.25$ | $-0.25$ |
| New $w = w + c^i x^i$ | $(0.25, 0.5)$: along $x^i$ | $(-0.25, -0.5)$: against $x^i$ |
| Change in $u^i$, $c^i \lVert x^i \rVert^2$ | $0.25 \times 5 = 1.25$ | $-0.25 \times 5 = -1.25$ |
| New output $\sigma(u^i)$ | $\sigma(1.25) = 0.777$: closer to 1 | $\sigma(-1.25) = 0.223$: closer to 0 |

As a check on the fourth row, computing $u^i$ directly with the new weights gives $0.25 \times 1 + 0.5 \times 2 = 1.25$ and $-0.25 \times 1 - 0.5 \times 2 = -1.25$.

## 7. What a single perceptron can draw: AND

The lecture's unit uses weights $0.2$ and $0.35$, the piecewise-linear transform, and threshold $\theta = 0.5$. (The slide calls the two inputs $x, y$ and the label $Z$, and I write the inputs as $x_1, x_2$ so that $y$ is not used for two things.)

| $x_1$ | $x_2$ | Label | $u = 0.2 x_1 + 0.35 x_2$ | $f(u)$(piecewise Linear) | Output ($f(u) \ge 0.5$?) |
|---|---|---|---|---|---|
| 0 | 0 | 0 | $0 + 0 = 0$ | 0 | 0 |
| 0 | 1 | 0 | $0 + 0.35 = 0.35$ | 0.35 | 0 |
| 1 | 0 | 0 | $0.2 + 0 = 0.2$ | 0.2 | 0 |
| 1 | 1 | 1 | $0.2 + 0.35 = 0.55$ | 0.55 | 1 |

All four are correct, so this unit computes AND. The slide labels the $f(u)$ column $P(y \mid x)$, but with a piecewise-linear transform it is just the unit's output, not a probability.

**Where the boundary is.** On $[0, 1]$ the transform is $f(u) = u$, so "$f(u) \ge 0.5$" means "$u \ge 0.5$", which is $0.2 x_1 + 0.35 x_2 \ge 0.5$. The boundary is the line $0.2 x_1 + 0.35 x_2 = 0.5$. It meets the $x_1$-axis at $0.5 / 0.2 = 2.5$($x_2 = 0, (x_1,x_2) = (2.5,0)$) and the $x_2$-axis at $0.5 / 0.35 \approx 1.43$($x_1 = 0, (x_1,x_2) = (0,1.43)$). Only $(1,1)$ will be above this linear boundary.

**Difficult Problem for Perceptron: Why one unit's boundary is a single straight line.**

**Claim:** If $f$ is non-decreasing (monotone) (all four transforms on the "Other Nonlinear Neurons" slide are), then the inputs on which unit outputs 1 are exactly the points on one side of a single straight line.

**Proof:**

1. **The boundary is made of straight lines.** The output depends on $x$ only through $u = w^\top x$. Different points $(x_1, x_2)$ can have the same value of $u$, and all points with $u = k$ lie on the straight line $w^\top x = k$, so they all get the same output. The output can therefore only change when we move from one of these parallel lines to another. This step uses nothing about $f$.
2. **There is only one such line.** Because $f$ is non-decreasing, if $f(u_1) \ge \theta$ and $u_2 > u_1$, then $f(u_2) \ge f(u_1) \ge \theta$. So once the output is 1, it stays 1 as $u$ grows: it switches from 0 to 1 at a single cutoff $c$, and the output is 1 exactly on one side of the line $w^\top x = c$. In the AND gate example, $c = 0.5$ and the line is $0.2 x_1 + 0.35 x_2 = 0.5$. With a sigmoid and $\theta = 0.6$ instead, $\sigma(u) \ge 0.6$ means $e^{-u} \le 2/3$, so $c = \ln(3/2) \approx 0.405$. The line moves to $0.2 x_1 + 0.35 x_2 = 0.405$, but it is still a single straight line, and $(1,1)$ (with $u = 0.55$) is still the only corner above it.

**Why the condition (f is monotone) matters.** Step 1 holds for any $f$, but step 2 needs $f$ to be non-decreasing. With the non-monotone $f(u) = u^2$ and $\theta = 1$, the output is 1 when $u \le -1$ or $u \ge 1$. So the boundary is two parallel lines, $w^\top x = -1$ and $w^\top x = 1$. This does not contradict the claim, because $u^2$ is not non-decreasing.

**The relation between this proof and the slides.** The "Difficult Problem for Perceptron" slide says no weights work for XOR or XNOR with any monotone transform and any threshold, and this proof is the first half of the reason: a single unit with a monotone transform always draws one straight line. Section 8 below gives the second half: no single straight line separates these points. So changing the transform or the threshold cannot help, and the next slide, "Connecting Perceptrons", uses more units instead. 

- **Back to Section 1.** The "Learning Non-linear Decision Boundary" slide says a linear model $w^\top x$ can only draw a straight boundary. Step 1 shows that putting a nonlinear transform $f$ on top of $u = w^\top x$ does not change this, so a single perceptron is still a linear classifier.
- **The AND gate example on slide.** On the "Decision Boundary of a Perceptron" slide, the line that separates $(1,1)$ from the other three points is straight because of step 1. It is exactly the line $0.2 x_1 + 0.35 x_2 = 0.5$ because of step 2, with cutoff $c = 0.5$.
- **The XOR slide.** The "Difficult Problem for Perceptron" slide says no weights work for XOR with any **monotone** function and any threshold, but it does not say why. The claim would like to support: no matter which monotone $f$ and threshold we pick, a unit draws one straight line. Section 8 then shows that no single straight line separates the XOR points.
- **Connecting perceptrons.** So, within the slide's rule that $f$ is monotone, changing $f$ or $\theta$ cannot help. The only way out is to use more units: each hidden unit draws its own line, and the output unit combines them. Therefore, the XOR problem can be solved!
## 8. What a single perceptron cannot do: XOR

The "Difficult Problem for Perceptron" slide asks for weights that produce this table, with **any** monotone transform and **any** threshold:

| $x_1$ | $x_2$ | Label |
|---|---|---|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

This table represents a XNOR gate which is the complement of XOR gate. Using either gate is able to show that if there is only one unit with any monotone transform and any threshold, then it is impossible to distinguish two outputs from XOR or XNOR gate with this single unit. 

**Proof that no weights work.** By Section 7, the unit outputs 1 exactly when $u \ge c$ for some cutoff $c$ (the case "$u > c$" works the same way). The cutoff $c$ already plays the role of the threshold, so I use the slide's unit with no bias: $u = w_1 x_1 + w_2 x_2$. The four corners give:

| Point | $u$ | Needed |
|---|---|---|
| $(0,0)$ | $0$ | $\ge c$ |
| $(1,1)$ | $w_1 + w_2$ | $\ge c$ |
| $(0,1)$ | $w_2$ | $< c$ |
| $(1,0)$ | $w_1$ | $< c$ |

- Adding the two label-1 rows gives $w_1 + w_2 \ge 2c$.
- Adding the two label-0 rows gives $w_1 + w_2 < 2c$.

The left-hand sides are the **same number**, so both cannot hold $w_1 + w_2$ can't not be greater than or equal to 2c and less than 2c at the same time. No choice of $w_1, w_2$, cutoff or monotone transform works. 

![Two plots of the four corners of the unit square, label-1 points as blue circles and label-0 points as orange squares. Left, AND: only (1,1) is blue, and a straight line separates it from the other three. Right, XNOR: (0,0) and (1,1) are blue, (0,1) and (1,0) are orange, and the two dotted diagonals joining each pair cross at the shared midpoint (0.5, 0.5).](course-1-images/fig4-and-vs-xnor.png)

*Figure 3. Left: the AND unit's boundary. Right: in the XNOR table, the two same-label pairs share a midpoint. Original figure made for these notes.*

## 9. Connecting perceptrons solves it

The "Connecting Perceptrons" slide connects two hidden units into one output: $w_1, w_2$ with $x_1, x_2$ enter into hidden unit 1, $w_3, w_4$ with $x_1, x_2$ enter into hidden unit 2, and $w_5, w_6$ with the hidden outputs $h_1, h_2$ enter to the output unit. With the slide's $(w_1, \dots, w_6) = (0.6, -0.6, -0.7, 0.8, 1, 1)$:

$$
u_1 = 0.6 x_1 - 0.6 x_2, \qquad u_2 = -0.7 x_1 + 0.8 x_2, \qquad u_{\text{out}} = h_1 + h_2
$$

The slide does not say which transform or thresholds to use, and the result depends on that. The choice I found that works is hard-limiter hidden units ($h = 1$ if $u \ge 0$) and an output that is 1 when $h_1 + h_2 \ge 1.5$:

| $x_1$ | $x_2$ | Label | $u_1$ | $h_1$ | $u_2$ | $h_2$ | $h_1 + h_2$ | Output |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | $0$ | 1 | $0$ | 1 | 2 | 1 |
| 0 | 1 | 0 | $-0.6$ | 0 | $0.8$ | 1 | 1 | 0 |
| 1 | 0 | 0 | $0.6$ | 1 | $-0.7$ | 0 | 1 | 0 |
| 1 | 1 | 1 | $0$ | 1 | $0.1$ | 1 | 2 | 1 |

**What each unit is doing.**

- Hidden unit 1 outputs 1 when $0.6 x_1 - 0.6 x_2 \ge 0$, i.e. $x_1 \ge x_2$.
- Hidden unit 2 outputs 1 when $-0.7 x_1 + 0.8 x_2 \ge 0$, i.e. $x_2 \ge 0.875 x_1$.
- The output needs $h_1 + h_2 \ge 1.5$. With $h_1, h_2 \in \lbrace 0, 1 \rbrace$ that means both must be 1, so the output is AND of the two hidden units.
- Together, the output is 1 only in the thin wedge $0.875 x_1 \le x_2 \le x_1$, which contains $(0,0)$ and $(1,1)$ and neither of the other corners.

![Two plots. Left, input space: the four XNOR points with a solid line x2 equals x1 and a dashed line x2 equals 0.875 times x1 from the origin, and the thin wedge between them shaded; the blue points (0,0) and (1,1) lie on its edge, and the orange points lie outside. Right, hidden space with axes h1 and h2: (0,0) and (1,1) both land at (1,1), (0,1) lands at (0,1), (1,0) lands at (1,0), and the line h1 plus h2 equals 1.5 separates blue from orange.](course-1-images/fig5-xnor-two-layer.png)

*Figure 4. Left: the line each hidden unit draws, and the wedge where both output 1. Right: the same four inputs in the coordinates $(h_1, h_2)$ produced by the hidden layer. Original figure made for these notes.*

**The part I think matters most** is the right panel. In the coordinates $(h_1, h_2)$, both label-1 points land at $(1,1)$ and the label-0 points at $(0,1)$ and $(1,0)$, so one line, $h_1 + h_2 = 1.5$, separates them. The output unit is still an ordinary perceptron, just working on features computed by the hidden layer. **The hidden layer is a learned $\phi(x)$**: the "learn $\phi$ and $w$ together" idea from Section 1.

In practice nobody sets these weights by hand. They are learned by gradient descent, and computing the gradient for hidden units (backpropagation) is the topic after the multilayer perceptron section.

## References

[1] Kai Wang. Neural Networks, CSE/ISyE 6740 Lecture 11, Georgia Tech, Fall 2026 (lecture slides).

[2] A. Krizhevsky, I. Sutskever, G. E. Hinton. ImageNet classification with deep convolutional neural networks. *NeurIPS 25*, 2012. [PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

## Declaration
AI is used to help me form this note. And here is how I use AI to assist me crafting the note:
- generating the plot to explain concepts geometrically
- explaining concepts and exploring ideas
- polishing my expression
- formatting MathJax-supported LaTeX 