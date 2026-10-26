---
title: Diffusion models
number: "16-diffusion"
date: 2026-10-26
summary: Generating data through diffusion and learned denoising.
status: published
---
## Learning goals

- Describe the forward process that gradually adds noise to data.
- Explain how a learned denoiser reverses this process to generate samples.
- Relate DDPM training to noise prediction.

## Materials

- [Lecture 16 slides (PDF, 27 pages)]({{ '/assets/semesters/fall-2026/lectures/16/lecture-16-diffusion-models.pdf' | relative_url }})

## Lecture notes

The forward diffusion process gradually corrupts a data sample with Gaussian noise. A model learns to predict the noise at a given timestep. During generation, repeated denoising steps transform an initial noise sample into a data sample.

The slides cover the diffusion process, denoising process, DDPM training loss, and applications.
