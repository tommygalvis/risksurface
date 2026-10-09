"""
Portfolio Risk Surface - Final Comprehensive Visualization

This script:
- Defines a 2-asset nonlinear portfolio V(S1, S2)
- Computes gradient and Hessian at a reference point
- Builds a 2x3 dashboard:
    (1,1) Surface heatmap + gradient at reference
    (1,2) Cross-sections through reference
    (1,3) Delta (∂V/∂S1) surface
    (2,1) Directional derivatives (polar plot)
    (2,2) Principal curvature directions (Hessian eigenvectors)
    (2,3) Taylor approximations vs true V along the diagonal
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

# Color palette
COLORS = {
    'primary': '#2ecc71',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'purple': '#9b59b6',
}

# Portfolio parameters
W1 = 100        # Position in asset 1 (shares)
W2 = 80         # Position in asset 2 (shares)
K = 50          # Nonlinear interaction strength

# Reference point
S1_STAR = 100
S2_STAR = 100


def portfolio_value(s1, s2, w1=W1, w2=W2, k=K):
    """Portfolio value: V = w1*S1 + w2*S2 + k*sin(S1/20)*cos(S2/20)."""
    linear_part = w1 * s1 + w2 * s2
    phi = k * np.sin(s1 / 20) * np.cos(s2 / 20)
    return linear_part + phi


def compute_gradient(s1, s2, w1=W1, w2=W2, k=K):
    """Gradient ∇V = (∂V/∂S1, ∂V/∂S2)."""
    dV_dS1 = w1 + (k / 20) * np.cos(s1 / 20) * np.cos(s2 / 20)
    dV_dS2 = w2 - (k / 20) * np.sin(s1 / 20) * np.sin(s2 / 20)
    return np.array([dV_dS1, dV_dS2])


def directional_derivative(s1, s2, direction):
    """Directional derivative D_u V at (s1, s2) in direction (d1, d2)."""
    d1, d2 = direction
    magnitude = np.sqrt(d1**2 + d2**2)
    u1, u2 = d1 / magnitude, d2 / magnitude
    grad_local = compute_gradient(s1, s2)
    return np.dot(grad_local, [u1, u2])


def compute_hessian(s1, s2, k=K):
    """Hessian H of V at (s1, s2)."""
    h11 = -(k / 400) * np.sin(s1 / 20) * np.cos(s2 / 20)
    h22 = -(k / 400) * np.sin(s1 / 20) * np.cos(s2 / 20)
    h12 = -(k / 400) * np.cos(s1 / 20) * np.sin(s2 / 20)
    return np.array([[h11, h12], [h12, h22]])


def taylor_approximation(s1, s2, order=2, s1_star=S1_STAR, s2_star=S2_STAR):
    """
    Taylor approximation of V around (s1_star, s2_star).

    order = 1 → linear
    order = 2 → quadratic
    """
    V_star_local = portfolio_value(s1_star, s2_star)
    grad_local = compute_gradient(s1_star, s2_star)
    delta = np.array([s1 - s1_star, s2 - s2_star])
    if order == 1:
        return V_star_local + np.dot(grad_local, delta)
    H_local = compute_hessian(s1_star, s2_star)
    return V_star_local + np.dot(grad_local, delta) + 0.5 * np.dot(delta, H_local @ delta)


# Precompute key quantities at reference point
V_star = portfolio_value(S1_STAR, S2_STAR)
grad = compute_gradient(S1_STAR, S2_STAR)
grad_magnitude = np.linalg.norm(grad)
grad_direction = grad / grad_magnitude
H = compute_hessian(S1_STAR, S2_STAR)
eigenvalues, eigenvectors = np.linalg.eigh(H)
idx = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


def main():
    print("Running final comprehensive visualization...")

    # Grid and surface
    s1_range = np.linspace(60, 140, 100)
    s2_range = np.linspace(60, 140, 100)
    S1, S2 = np.meshgrid(s1_range, s2_range)
    V = portfolio_value(S1, S2)

    # Delta grid (∂V/∂S1)
    dV_dS1_grid = W1 + (K / 20) * np.cos(S1 / 20) * np.cos(S2 / 20)

    # Directional derivatives around the reference point
    theta = np.linspace(0, 2 * np.pi, 360)
    D_u_values = [
        directional_derivative(S1_STAR, S2_STAR, (np.cos(t), np.sin(t)))
        for t in theta
    ]
    grad_angle = np.arctan2(grad[1], grad[0])

    # ---- Final comprehensive visualization ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Surface views
    # 1.1: Heatmap with gradient
    im1 = axes[0, 0].contourf(S1, S2, V, levels=20, cmap='viridis')
    axes[0, 0].contour(S1, S2, V, levels=10, colors='white',
                       alpha=0.3, linewidths=0.5)
    axes[0, 0].quiver(
        S1_STAR, S2_STAR,
        grad[0]*0.3, grad[1]*0.3,
        color='red', scale=50, width=0.015
    )
    axes[0, 0].scatter(
        [S1_STAR], [S2_STAR],
        color='red', s=100, zorder=5, edgecolors='white'
    )
    axes[0, 0].set_xlabel('$S_1$')
    axes[0, 0].set_ylabel('$S_2$')
    axes[0, 0].set_title('Surface & Gradient', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], label='V($)')

    # 1.2: Cross-sections
    axes[0, 1].plot(
        s1_range,
        portfolio_value(s1_range, S2_STAR),
        color=COLORS['secondary'], linewidth=2,
        label=f'V($S_1$, {S2_STAR})'
    )
    axes[0, 1].plot(
        s1_range,
        portfolio_value(S1_STAR, s1_range),
        color=COLORS['primary'], linewidth=2,
        label=f'V({S1_STAR}, $S_2$)'
    )
    axes[0, 1].axvline(S1_STAR, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].scatter([S1_STAR], [V_star], color='red', s=100, zorder=5)
    axes[0, 1].set_xlabel('Price')
    axes[0, 1].set_ylabel('V($)')
    axes[0, 1].set_title('Cross-Sections', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 1.3: Delta surface (∂V/∂S₁)
    im3 = axes[0, 2].contourf(S1, S2, dV_dS1_grid, levels=20, cmap='RdYlBu_r')
    axes[0, 2].scatter(
        [S1_STAR], [S2_STAR],
        color='black', s=100, zorder=5, edgecolors='white'
    )
    axes[0, 2].set_xlabel('$S_1$')
    axes[0, 2].set_ylabel('$S_2$')
    axes[0, 2].set_title('∂V/∂$S_1$ (Delta to Asset 1)', fontweight='bold')
    plt.colorbar(im3, ax=axes[0, 2])

    # Row 2: Analysis
    # 2.1: Directional derivatives (polar)
    axes[1, 0].remove()  # replaced by the polar axes below
    ax_polar = plt.subplot(2, 3, 4, projection='polar')
    ax_polar.plot(theta, D_u_values, color=COLORS['secondary'], linewidth=2)
    ax_polar.fill(theta, D_u_values, alpha=0.3, color=COLORS['secondary'])
    ax_polar.scatter(
        [grad_angle], [grad_magnitude],
        color=COLORS['primary'], s=80, zorder=5
    )
    ax_polar.set_title('Directional Derivatives', fontweight='bold', pad=20)

    # 2.2: Principal directions
    axes[1, 1].contour(S1, S2, V, levels=15, colors='gray', alpha=0.5)
    scale = 15
    axes[1, 1].arrow(
        S1_STAR, S2_STAR,
        eigenvectors[0, 0]*scale, eigenvectors[1, 0]*scale,
        head_width=2, head_length=1,
        fc=COLORS['accent'], ec=COLORS['accent'],
        linewidth=2
    )
    axes[1, 1].arrow(
        S1_STAR, S2_STAR,
        eigenvectors[0, 1]*scale, eigenvectors[1, 1]*scale,
        head_width=2, head_length=1,
        fc=COLORS['primary'], ec=COLORS['primary'],
        linewidth=2
    )
    axes[1, 1].scatter(
        [S1_STAR], [S2_STAR],
        color='red', s=100, zorder=5, edgecolors='white'
    )
    axes[1, 1].set_xlabel('$S_1$')
    axes[1, 1].set_ylabel('$S_2$')
    axes[1, 1].set_title('Principal Curvature Directions', fontweight='bold')
    axes[1, 1].set_xlim(70, 130)
    axes[1, 1].set_ylim(70, 130)

    # 2.3: Taylor approximation accuracy along diagonal
    t = np.linspace(-15, 15, 100)
    V_true_diag = portfolio_value(S1_STAR + t, S2_STAR + t)
    V_lin_diag = np.array([
        taylor_approximation(S1_STAR + ti, S2_STAR + ti, order=1)
        for ti in t
    ])
    V_quad_diag = np.array([
        taylor_approximation(S1_STAR + ti, S2_STAR + ti, order=2)
        for ti in t
    ])

    axes[1, 2].plot(
        t, V_true_diag,
        color=COLORS['secondary'], linewidth=2.5, label='True V'
    )
    axes[1, 2].plot(
        t, V_lin_diag,
        color=COLORS['accent'], linewidth=2, linestyle='--', label='Linear'
    )
    axes[1, 2].plot(
        t, V_quad_diag,
        color=COLORS['primary'], linewidth=2, linestyle=':', label='Quadratic'
    )
    axes[1, 2].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 2].scatter([0], [V_star], color='red', s=100, zorder=5)
    axes[1, 2].set_xlabel('Distance along diagonal')
    axes[1, 2].set_ylabel('V($)')
    axes[1, 2].set_title('Taylor Approximations', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
