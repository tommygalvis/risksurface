# Portfolio Risk Surface

Visualizes the risk of a two-asset portfolio with a nonlinear payoff, using its gradient (deltas), Hessian (curvature) and Taylor approximations.

## Model

```
V(S₁, S₂) = w₁·S₁ + w₂·S₂ + k·sin(S₁/20)·cos(S₂/20)
```

The defaults are `w₁ = 100` shares, `w₂ = 80` shares and `k = 50`, with the reference point at `S₁ = S₂ = 100`. The gradient and Hessian are computed analytically.

## Dashboard

| | | |
|---|---|---|
| **Surface & gradient**: value heatmap with the gradient arrow at the reference point | **Cross-sections**: V along each price axis through the reference point | **Delta surface**: ∂V/∂S₁ across the price grid |
| **Directional derivatives**: rate of change of V in every direction (polar plot) | **Principal curvature directions**: Hessian eigenvectors | **Taylor approximations**: linear and quadratic approximations vs the true V along the diagonal |

## Usage

```bash
pip install -r requirements.txt
python risksurface.py
```
