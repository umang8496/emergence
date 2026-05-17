# Deep Learning — From First Principles

This directory contains a structured deep learning learning track focused on first-principles understanding and practical implementation.

It is designed to help you move from intuition to mathematics to production-style PyTorch training workflows.

## What You Will Learn

- Core deep learning concepts: perceptron, neuron, neural networks, forward pass, backpropagation, and gradient descent.
- Architecture intuition: ANN, CNN, RNN/LSTM, Transformer, GAN, Autoencoder, VAE, GNN, and diffusion models.
- Practical PyTorch workflows: dataset handling, dataloaders, model design, training loops, evaluation, GPU usage, and checkpointing.
- Generalization and model behavior: underfitting, overfitting, bias-variance tradeoff, and regularization techniques.

## Directory Contents

| File                                                                         | Focus Area                                                            | Best For                                                  |
|------------------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------|
| [DEEP_LEARNING_BASICS.md](DEEP_LEARNING_BASICS.md)                           | Conceptual foundations + end-to-end worked example                    | Building deep intuition from scratch                      |
| [ANN_REFERENCE_GUIDE.md](ANN_REFERENCE_GUIDE.md)                             | Scale-ready PyTorch ANN implementation patterns                       | Fast implementation and architecture/training decisions   |
| [REGULARIZATION_AND_TRAINING_GUIDE.md](REGULARIZATION_AND_TRAINING_GUIDE.md) | Overfitting/underfitting diagnosis, regularization, and stabilization | Improving model generalization and training reliability   |

## Recommended Learning Path

1. Start with [DEEP_LEARNING_BASICS.md](DEEP_LEARNING_BASICS.md) to build conceptual and mathematical intuition.
2. Move to [ANN_REFERENCE_GUIDE.md](ANN_REFERENCE_GUIDE.md) to translate concepts into reusable PyTorch code.
3. Finish with [REGULARIZATION_AND_TRAINING_GUIDE.md](REGULARIZATION_AND_TRAINING_GUIDE.md) to diagnose and fix real training issues.

## How To Use This Folder

- Use the basics guide when you want to understand why something works.
- Use the ANN reference guide as a practical blueprint while coding.
- Use the regularization guide when training curves look wrong or performance does not generalize.

## Quick Decision Guide

- Need first-principles explanation: [DEEP_LEARNING_BASICS.md](DEEP_LEARNING_BASICS.md)
- Need PyTorch implementation template: [ANN_REFERENCE_GUIDE.md](ANN_REFERENCE_GUIDE.md)
- Need to reduce overfitting or stabilize training: [REGULARIZATION_AND_TRAINING_GUIDE.md](REGULARIZATION_AND_TRAINING_GUIDE.md)

## Scope Notes

- This folder is documentation-first and focuses on education and implementation patterns.
- Code snippets are illustrative and ready to adapt into notebooks or scripts.
- Content is framework-focused on PyTorch for ANN workflows.

---
