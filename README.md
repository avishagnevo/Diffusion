# 1D Diffusion Model Implementation

This repository contains a custom implementation of a 1D Diffusion Model inspired by the **Denoising Diffusion Probabilistic Models (DDPM)** paper. The code was developed from scratch, drawing insights from annotated examples and visualization techniques encountered during the research process. This implementation successfully learns the target distributions and provides a simplified yet effective approach to DDPM.

---

## Overview

The implementation follows the key principles outlined in the DDPM paper, including the forward and reverse diffusion processes. The model is trained using the variational bound and its simplified variant to optimize the diffusion process.

### Key Equations

#### Equation 3: Variational Bound on Negative Log-Likelihood
Training optimizes the variational bound on negative log-likelihood:

$$
\mathcal{L}(\theta) = \mathbb{E}_q \left[ -\log p_\theta(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]
$$

---

#### Equation 12: Simplified Loss Function
The model is also trained using a simplified variant of the variational bound:

$$
\mathcal{L}'(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[
\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta \left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t \right) \right\|^2
\right]
$$

---

## Implementation Details

### Forward Process
- **Total Time Steps**: $T = 1000$, consistent with prior work.
- **Variance Schedule**: $\beta_t = \beta = 0.02$, fixed over time.
- **Alpha Schedule**:
  - Linear: $\alpha_t = \alpha = 1 - \beta$
  - Cumulative Product: $\bar{\alpha}_t = \prod_{s=1}^t \alpha$

### Reverse Process
- Parameterized using neural networks trained on:
  - Full variational bound: $\mathcal{L}(\theta)$
  - Simplified variant: $\mathcal{L}'(\theta)$
- **Model Architecture**:
  - A lightweight feedforward neural network.
  - Naive positional encoding for time steps.
  - Basic scaling $[1, 0]$.
- **Design Choice**: A simpler architecture was chosen over the original U-Net and attention-heavy models due to the lightweight nature of the dataset.

---

## Visualization and Results
The implementation includes visualizations that demonstrate the learning process and the model's success in approximating target distributions.

---

## References
- DDPM Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Annotated Examples: Refer to the documentation and community examples explored during the development process.

---

Feel free to explore the code and extend the implementation for more complex datasets or applications. Contributions and feedback are welcome!
