# Oblique Split Algorithm

## 1. General Framework and Notation

### 1.1 Data and Parameter Definitions
- $X \in \mathbb{R}^{N \times d}$: Feature matrix
- $\mathbf{y} \in \{0,1,\ldots,K-1\}^N$ or $\mathbb{R}^N$: Target variable (categorical for classification, continuous for regression)
- $\mathbf{w} \in \mathbb{R}^d$: Optimization parameters defining the split hyperplane
- $\boldsymbol{\omega} \in \mathbb{R}^N$: Sample weights
- $\gamma$: Scaling parameter for the sigmoid function (controls the sharpness of the split)
- $\epsilon$: Small constant for numerical stability

### 1.2 Basic Computations
For each sample $i$:

$$
z_i = \mathbf{x}_i^\top\mathbf{w}
$$

- Computes the projection of each sample onto the direction vector $\mathbf{w}$

$$
p_i = \sigma(\gamma z_i) = \frac{1}{1 + e^{-\gamma z_i}}
$$

- Transforms the projection into a probability of belonging to the left child node
- Smoothed version of a hard split, allowing gradient-based optimization

### 1.3 Optimal Threshold Selection
After obtaining optimal weights $\mathbf{w}$, we select the threshold $t$ that minimizes the impurity measure:

$$
t^* = \arg\min_t \text{Impurity}(\{z_i \leq t\})
$$

where Impurity is either Gini impurity for classification or MSE for regression.
In implementation, this threshold is chosen by an exact scan over the sorted projected values $\{z_i\}_{i=1}^N$ after the weight optimization step.

## 2. Binary Classification

### 2.1 Soft Decision Tree Approach

| Left Node | Right Node |
|-----------|------------|
| $S_L = \sum_{i=1}^N \omega_i p_i + \epsilon$ | $S_R = \sum_{i=1}^N \omega_i(1-p_i) + \epsilon$ |
| $M_L = \sum_{i=1}^N \omega_i p_i y_i$ | $M_R = \sum_{i=1}^N \omega_i(1-p_i)y_i$ |
| $P_{L1} = \frac{M_L}{S_L}$ | $P_{R1} = \frac{M_R}{S_R}$ |

These statistics track the weighted distribution of samples and their labels in each child node.

The optimization objective is to minimize:

$$
\mathcal{L}_{\text{Gini}}(\mathbf{w}) = S_L P_{L1}(1-P_{L1}) + S_R P_{R1}(1-P_{R1})
$$

The optimization requires computing several gradients:

1. Basic probability gradient:

$$
\frac{\partial p_i}{\partial z_i} = \gamma p_i(1-p_i)
$$

2. Node sum gradients:

$$
\frac{\partial S_L}{\partial \mathbf{w}} = \sum_{i=1}^N \omega_i\frac{\partial p_i}{\partial z_i}\mathbf{x}_i
$$

$$
\frac{\partial M_L}{\partial \mathbf{w}} = \sum_{i=1}^N \omega_i y_i\frac{\partial p_i}{\partial z_i}\mathbf{x}_i
$$

3. Class probability gradient:

$$
\frac{\partial P_{L1}}{\partial \mathbf{w}} = \frac{S_L\frac{\partial M_L}{\partial \mathbf{w}} - M_L\frac{\partial S_L}{\partial \mathbf{w}}}{S_L^2}
$$

4. Final gradient:

$$
\frac{\partial \mathcal{L}_{\text{Gini}}}{\partial \mathbf{w}} = \frac{\partial S_L}{\partial \mathbf{w}}P_{L1}(1-P_{L1}) + S_L\frac{\partial P_{L1}}{\partial \mathbf{w}}(1-2P_{L1}) + \frac{\partial S_R}{\partial \mathbf{w}}P_{R1}(1-P_{R1}) + S_R\frac{\partial P_{R1}}{\partial \mathbf{w}}(1-2P_{R1})
$$

### 2.2 Linear Classification Approach
This approach treats the split as a linear classification problem.

$$
\mathcal{L}_{\text{BCE}}(\mathbf{w}) = -\frac{1}{\sum_{i=1}^N \omega_i}\sum_{i=1}^N \omega_i[y_i\log(\sigma(\mathbf{x}_i^\top\mathbf{w})) + (1-y_i)\log(1-\sigma(\mathbf{x}_i^\top\mathbf{w}))]
$$

$$
\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{w}} = \frac{1}{\sum_{i=1}^N \omega_i}\mathbf{X}^\top(\boldsymbol{\omega} \odot (\sigma(\mathbf{X}\mathbf{w}) - \mathbf{y}))
$$

## 3. Multiclass Classification

### 3.1 Soft Decision Tree Approach

| Left Node | Right Node |
|-----------|------------|
| $C_{L,k} = \sum_{i:y_i=k} \omega_i p_i$ | $C_{R,k} = \sum_{i:y_i=k} \omega_i(1-p_i)$ |
| $S_L = \sum_{k=0}^{K-1} C_{L,k} + \epsilon$ | $S_R = \sum_{k=0}^{K-1} C_{R,k} + \epsilon$ |
| $P_{L,k} = \frac{C_{L,k}}{S_L}$ | $P_{R,k} = \frac{C_{R,k}}{S_R}$ |

These statistics track the weighted distribution of samples for each class in the child nodes.

The optimization objective is to minimize:

$$
\mathcal{L}_{\text{MultiGini}}(\mathbf{w}) =
S_L \sum_{k=0}^{K-1} P_{L,k}(1-P_{L,k}) +
S_R \sum_{k=0}^{K-1} P_{R,k}(1-P_{R,k})
$$

The optimization requires computing several gradients:

1. Basic probability gradient:

$$
\frac{\partial p_i}{\partial z_i} = \gamma p_i(1-p_i)
$$

2. Total soft mass gradient:

$$
\frac{\partial S_L}{\partial \mathbf{w}} = \sum_{i=1}^N \omega_i\frac{\partial p_i}{\partial z_i}\mathbf{x}_i
$$

3. Final gradient.

Define

$$
v_i = 2\,\omega_i\frac{\partial p_i}{\partial z_i}\left(P_{R,y_i} - P_{L,y_i}\right)
$$

Then

$$
\frac{\partial \mathcal{L}_{\text{MultiGini}}}{\partial \mathbf{w}} =
\mathbf{X}^\top \mathbf{v} +
\frac{\partial S_L}{\partial \mathbf{w}}
\left(
\sum_{k=0}^{K-1} P_{R,k}(1-P_{R,k}) -
\sum_{k=0}^{K-1} P_{L,k}(1-P_{L,k})
\right)
$$

### 3.2 One-vs-Rest Approach
For $K$ classes, we train $K$ binary classifiers where for each classifier $k$:
- Class $k$ samples are labeled as 1
- All other class samples are labeled as 0

The final decision is made by selecting the class whose model gives the minimum loss:

$$
k^* = \arg\min_k \mathcal{L}_k(\mathbf{w}_k)
$$

## 4. Regression

### 4.1 Soft Decision Tree Approach

| Left Node | Right Node |
|-----------|------------|
| $S_L = \sum_{i=1}^N \omega_i p_i + \epsilon$ | $S_R = \sum_{i=1}^N \omega_i(1-p_i) + \epsilon$ |
| $M_L = \sum_{i=1}^N \omega_i p_i y_i$ | $M_R = \sum_{i=1}^N \omega_i(1-p_i)y_i$ |
| $m_L = \frac{M_L}{S_L}$ | $m_R = \frac{M_R}{S_R}$ |

These statistics track the weighted sum and mean of target values in each child node.

The optimization objective is to minimize:

$$
\mathcal{L}_{\text{MSE}}(\mathbf{w}) = \frac{1}{\sum_{i=1}^N \omega_i}\sum_{i=1}^N \omega_i[p_i(y_i-m_L)^2 + (1-p_i)(y_i-m_R)^2]
$$

The optimization requires computing several gradients:

1. Node mean gradients:

$$
\frac{\partial m_L}{\partial \mathbf{w}} = \frac{S_L\frac{\partial M_L}{\partial \mathbf{w}} - M_L\frac{\partial S_L}{\partial \mathbf{w}}}{S_L^2}
$$

$$
\frac{\partial m_R}{\partial \mathbf{w}} = \frac{S_R\frac{\partial M_R}{\partial \mathbf{w}} - M_R\frac{\partial S_R}{\partial \mathbf{w}}}{S_R^2}
$$

2. Final gradient:

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \mathbf{w}} = 
    & \frac{1}{\sum_{i=1}^N \omega_i}\sum_{i=1}^N \omega_i\left[\frac{\partial p_i}{\partial \mathbf{w}}(y_i-m_L)^2 - \frac{\partial p_i}{\partial \mathbf{w}}(y_i-m_R)^2 + \right. \\
    & \left. p_i\frac{\partial m_L}{\partial \mathbf{w}}(-2)(y_i-m_L) + (1-p_i)\frac{\partial m_R}{\partial \mathbf{w}}(-2)(y_i-m_R)\right]
\end{aligned}
$$

### 4.2 Linear Regression Approach

$$
\mathcal{L}_{\text{LinReg}}(\mathbf{w}) = \frac{1}{2\sum_{i=1}^N \omega_i}\sum_{i=1}^N \omega_i(y_i - \mathbf{x}_i^\top\mathbf{w})^2
$$

$$
\frac{\partial \mathcal{L}_{\text{LinReg}}}{\partial \mathbf{w}} = -\frac{1}{\sum_{i=1}^N \omega_i}\mathbf{X}^\top(\boldsymbol{\omega} \odot (\mathbf{y} - \mathbf{X}\mathbf{w}))
$$
## 5. L-BFGS Optimization

The optimization process uses a custom unconstrained L-BFGS implementation to solve:

$$
\min_{\mathbf{w}} \mathcal{L}(\mathbf{w})
$$

Key elements of the optimization process:

1. Objective Function:
   - Minimizes the loss function (Gini impurity or MSE)

2. Stopping Criteria:
   - Stops when maximum iterations (`maxiter`) is reached
   - Stops when the infinity norm of the gradient falls below a small tolerance
   - Stops when relative improvement falls below threshold:

     $$
     \frac{\mathcal{L}(\mathbf{w}_{t-1}) - \mathcal{L}(\mathbf{w}_t)}{\mathcal{L}(\mathbf{w}_{t-1})} \leq \text{relative\_change}
     $$

3. Solution Normalization:
   - Final weights are normalized:
   
     $$
     \mathbf{w} \leftarrow \frac{\mathbf{w}}{\max(|\mathbf{w}|)}
     $$
