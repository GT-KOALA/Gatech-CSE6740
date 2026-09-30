---
title: Neural networks
number: "11"
date: 2026-09-30
summary: Composing linear transformations and nonlinearities into trainable models.
status: published
---
## Learning goals

- Describe the structure of a feedforward network.
- Explain forward propagation and backpropagation.
- Recognize activation and initialization choices.

## Materials

- [Backpropagation Lab — interactive visualization]({{ '/demos/backpropagation/' | relative_url }})

Trace a two-hidden-layer network forward and backward, reveal the forward pass one layer at a time, and click a weight to inspect the symbolic gradients in its chain rule. Change inputs, weights, biases, or activations, then apply gradient descent and watch the loss change.

- [Lecture 11 slides (PDF, 39 pages)]({{ '/assets/semesters/fall-2026/lectures/11/lecture-11-neural-networks.pdf' | relative_url }})

## Lecture notes

The lab uses sigmoid outputs and squared loss, matching the lecture’s backpropagation setup. Its guided walkthrough highlights how gradients multiply along a path and add across branches. Try the saturated-sigmoid and inactive-ReLU examples to see why gradients can become small or zero.

