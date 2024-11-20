import matplotlib.pyplot as plt
import numpy as np

# Plotting the cavity
plt.figure(figsize=(6, 6))
plt.quiver([0.5], [1.0], [0.5], [0.0], angles="xy", scale_units="xy", scale=1, color="r", label="Lid velocity")
plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, fill=None, edgecolor="black"))
plt.text(0.5, 1.05, "u = U, v = 0", ha="center", color="red")
plt.text(0.05, 0.5, "u = 0, v = 0", va="center", rotation=90)
plt.text(0.5, -0.05, "u = 0, v = 0", ha="center")
plt.text(1.05, 0.5, "u = 0, v = 0", va="center", rotation=90)
plt.axis("scaled")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title("Lid-Driven Cavity: Boundary Conditions")
plt.legend()
plt.show()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# Poisson solver using Jacobi iteration
@jax.jit
def solve_poisson_jacobi(omega, psi, h, max_iter=500, tol=1e-6):
    h2 = h * h
    for _ in range(max_iter):
        psi_new = (psi[1:-1, 2:] + psi[1:-1, :-2] + psi[2:, 1:-1] + psi[:-2, 1:-1] + omega[1:-1, 1:-1] * h2) / 4.0
        psi = psi.at[1:-1, 1:-1].set(psi_new)

        # Compute the residual for convergence check (optional)
        residual = jnp.linalg.norm(psi_new - psi[1:-1, 1:-1])
        # if residual < tol:
        #     break

    return psi


# Lid-driven cavity solver
def cavity(Re, N, T):
    # Parameters
    Uwall = 1.0
    L = 1.0
    nu = Uwall * L / Re
    s = jnp.linspace(0.0, 1.0, N + 1)
    h = s[1] - s[0]
    h2 = h * h
    Y, X = jnp.meshgrid(s, s)

    # Initialize variables
    O = jnp.zeros((N + 1, N + 1))  # Vorticity
    P = jnp.zeros((N + 1, N + 1))  # Streamfunction
    Q = jnp.zeros((N + 1, N + 1))  # Temporary array for updates

    # Time stepping
    dt = min(0.25 * h2 / nu, 4.0 * nu / Uwall**2)
    Nt = int(jnp.ceil(T / dt))

    # Time loop
    for n in range(Nt):
        # Poisson solve for streamfunction
        P = solve_poisson_jacobi(O, P, h)

        # Vorticity boundary conditions
        O = O.at[1:N, 0].set(-2.0 * P[1:N, 1] / h2)  # Bottom
        O = O.at[1:N, N].set(-2.0 * P[1:N, N - 1] / h2 - 2.0 / h * Uwall)  # Top
        O = O.at[0, 1:N].set(-2.0 * P[1, 1:N] / h2)  # Left
        O = O.at[N, 1:N].set(-2.0 * P[N - 1, 1:N] / h2)  # Right

        # Vorticity update (interior nodes)
        Px = (P[1:-1, 2:] - P[1:-1, :-2]) / (2 * h)  # dP/dx
        Py = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2 * h)  # dP/dy
        Ox = (O[1:-1, 2:] - O[1:-1, :-2]) / (2 * h)  # dO/dx
        Oy = (O[2:, 1:-1] - O[:-2, 1:-1]) / (2 * h)  # dO/dy

        convection = -0.25 * (Px * Oy - Py * Ox)
        diffusion = nu * (O[1:-1, 2:] + O[1:-1, :-2] + O[2:, 1:-1] + O[:-2, 1:-1] - 4.0 * O[1:-1, 1:-1]) / h2

        Q = convection + diffusion
        # Update time step
        U = jnp.abs(P[:, 1 : N + 1] - P[:, 0:N]) / h
        V = jnp.abs(P[1 : N + 1, :] - P[0:N, :]) / h
        vmax = max(jnp.max(U) + jnp.max(V), Uwall)
        dt = min(0.25 * h2 / nu, 4.0 * nu / vmax**2)
        # print(jnp.linalg.norm(Q))

        # Forward Euler update
        O = O.at[1:-1, 1:-1].add(dt * Q)

    return X, Y, O, P


# Run the simulation
Re = 100
N = 32
T = 10.0
X, Y, O, P = cavity(Re, N, T)

plt.contourf(X, Y, P, levels=10, cmap="viridis")
plt.colorbar(label="Streamfunction")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.contourf(X, Y, O, levels=10, cmap="viridis")
plt.colorbar(label="Vorticity")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


@jax.jit
def compute_velocity(psi, U):
    u = (psi[:, 2:] - psi[:, :-2]) / (2 * dx)  # u = dpsi/dy
    v = -(psi[2:, :] - psi[:-2, :]) / (2 * dx)  # v = -dpsi/dx
    # Pad velocity components
    u = jnp.pad(u, ((0, 0), (1, 1)), mode="constant")  # Pad in x-direction
    v = jnp.pad(v, ((1, 1), (0, 0)), mode="constant")  # Pad in y-direction

    # Apply boundary conditions for u and v
    # Top wall (lid): u = Uwall, v = 0
    u = u.at[:, -1].set(U)  # Set u to Uwall on the top boundary
    v = v.at[-1, :].set(0)  # Set v to 0 on the top boundary

    # Other walls (no-slip): u = 0, v = 0
    u = u.at[0, :].set(0)  # Bottom boundary
    u = u.at[:, 0].set(0)  # Left boundary
    u = u.at[-1, :].set(0)  # Right boundary
    v = v.at[0, :].set(0)  # Bottom boundary
    v = v.at[:, 0].set(0)  # Left boundary
    v = v.at[:, -1].set(0)  # Right boundary
    return u, v


dx = L / N
Uwall = 1.0
u, v = compute_velocity(P, Uwall)

# Create a quiver plot
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, u, v, scale=20, color="blue", pivot="middle", alpha=0.7)
plt.title("Velocity Field (Quiver Plot)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("scaled")
plt.show()
