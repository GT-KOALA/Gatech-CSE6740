---
title: LLM training and finetuning
number: "20"
date: 2026-11-11
summary: Training and adapting large language models, including parameter-efficient finetuning with LoRA.
status: published
---
## Learning goals

- Distinguish language-model pretraining from finetuning.
- Explain how LoRA adapts a frozen backbone using low-rank updates.
- Compare trainable parameter counts, memory, and compute costs.
- Examine recorded model outputs before and after adaptation.

## Materials

- [Lecture 20 slides (PDF)]({{ '/assets/semesters/fall-2026/lectures/20/lecture-20-llm-training-and-finetuning.pdf' | relative_url }})
- [LoRA Lab — interactive explanation]({{ '/demos/lora/' | relative_url }}) — Explore the low-rank update, forward and backward paths, Transformer parameters, and training costs.
- [Recorded LoRA demos]({{ '/demos/lora/#results' | relative_url }}) — Compare saved backbone and adapter outputs with word-paced replay, scores, and reference context.
