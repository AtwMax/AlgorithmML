# Optimization Methods for Machine Learning

This project implements several optimization techniques commonly used in machine learning: **Stochastic Gradient Descent (SGD)**, **Adam**, **Momentum**, and **Nesterov Accelerated Gradient (NAG)**. It includes code to test these optimizers on a benchmark cost function, the **Rosenbrock function**.

## Table of Contents
- [Overview](#overview)
- [Optimization Methods](#optimization-methods)

## Overview

This project aims to compare different optimization techniques in terms of their convergence behavior on the **Rosenbrock function**, a well-known optimization test function. The optimizers implemented include:

1. **Stochastic Gradient Descent (SGD)**
2. **Adam**
3. **Momentum**
4. **Nesterov Accelerated Gradient (NAG)**

Each optimizer is evaluated over 100,000 iterations, and the convergence (cost values) is recorded for analysis.

## Optimization Methods

The following optimization algorithms are implemented:

### 1. Stochastic Gradient Descent (SGD)
SGD is a simple and efficient optimization method that updates parameters using the gradient of the cost function with respect to the parameters. 

### 2. Adam Optimizer
Adam (Adaptive Moment Estimation) is a popular method that adapts the learning rate based on the first and second moments of the gradients.

### 3. Momentum Optimizer
Momentum accelerates SGD by adding a fraction of the previous update to the current one, which helps speed up convergence.

### 4. Nesterov Accelerated Gradient (NAG)
NAG is a variation of momentum where the gradient is calculated after the momentum step, leading to faster convergence.

