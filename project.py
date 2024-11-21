import numpy as np
import matplotlib.pyplot as plt

def generate_shishkin_mesh(N, epsilon, alpha=1):
    tau = min(0.5, alpha * epsilon * np.log(N))
    x = np.zeros(N + 1)
    for i in range(N + 1):
        t = i / N
        if t <= 0.5:
            x[i] = tau * np.log(1 + 2 * t * (np.exp(1 / tau) - 1))
        else:
            x[i] = 1 - 2 * (1 - tau) * (1 - t)
    return x

def generate_adaptive_mesh(N, monitor_function):
    x = np.linspace(0, 1, N + 1)
    for _ in range(10):
        M = monitor_function(x)
        cumulative_M = np.cumsum(M[:-1] * np.diff(x))
        cumulative_M /= cumulative_M[-1]
        x_new = np.interp(np.linspace(0, 1, N + 1), np.hstack(([0], cumulative_M)), x)
        if np.allclose(x, x_new, atol=1e-5):
            break
        x = x_new
    return x

def monitor_function_example(u_prime):
    return lambda x: 1 + np.abs(u_prime(x))**0.5

def solve_bvp(N, epsilon, example, mesh_type="shishkin", scheme="hybrid"):
    if mesh_type == "shishkin":
        x = generate_shishkin_mesh(N, epsilon)
    elif mesh_type == "adaptive":
        u_prime = lambda x: -2 * np.exp(-x / epsilon) / epsilon
        x = generate_adaptive_mesh(N, monitor_function_example(u_prime))
    
    U = np.zeros(N + 1)
    lamb = 0.5
    for _ in range(100):
        A = np.zeros((N + 1, N + 1))
        B = np.zeros(N + 1)
        for i in range(1, N):
            hi = x[i] - x[i - 1]
            sigma = 0.5 if scheme == "hybrid" else 1.0
            U_mid = sigma * U[i] + (1 - sigma) * U[i - 1]
            A[i, i - 1] = -epsilon / hi
            A[i, i] = epsilon / hi + 2
            B[i] = -example(x[i], U_mid, lamb)
        A[0, 0] = A[-1, -1] = 1
        B[0], B[-1] = 0, 1
        delta_U = np.linalg.solve(A, B)
        U += delta_U
        lamb += np.mean(delta_U)
        if np.linalg.norm(delta_U, ord=np.inf) < 1e-5:
            break
    return x, U

def example_6_1(x, u, lamb):
    return 2 * u - np.exp(-u) + lamb

def example_6_2(x, u, lamb):
    return 2 * u - np.exp(-u) + x * lamb + x**2

def compute_error(U, U_exact):
    return np.abs(U - U_exact)

N = 64
epsilon = 0.01

x1_shishkin, U1_shishkin = solve_bvp(N, epsilon, example_6_1, mesh_type="shishkin", scheme="hybrid")
x1_adaptive, U1_adaptive = solve_bvp(N, epsilon, example_6_1, mesh_type="adaptive", scheme="hybrid")

x2_shishkin, U2_shishkin = solve_bvp(N, epsilon, example_6_2, mesh_type="shishkin", scheme="hybrid")
x2_adaptive, U2_adaptive = solve_bvp(N, epsilon, example_6_2, mesh_type="adaptive", scheme="hybrid")

errors_shishkin = compute_error(U1_shishkin[:-1], U1_shishkin[-2])
errors_adaptive = compute_error(U1_adaptive[:-1], U1_adaptive[-2])

plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
plt.plot(x1_shishkin, U1_shishkin, label="Shishkin Mesh")
plt.plot(x1_adaptive, U1_adaptive, label="Adaptive Mesh")
plt.title("Example 6.1 Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x2_shishkin, U2_shishkin, label="Shishkin Mesh")
plt.plot(x2_adaptive, U2_adaptive, label="Adaptive Mesh")
plt.title("Example 6.2 Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()

plt.subplot(2, 2, 3)
plt.loglog(x1_shishkin[:-1], errors_shishkin, label="Shishkin Mesh")
plt.loglog(x1_adaptive[:-1], errors_adaptive, label="Adaptive Mesh")
plt.title("Error Convergence (Example 6.1)")
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x1_adaptive[:-1], np.diff(x1_adaptive), label="Adaptive Mesh Step Sizes")
plt.title("Adaptive Mesh Step Sizes")
plt.xlabel("x")
plt.ylabel("Step Size")
plt.legend()

plt.tight_layout()
plt.show()

print("Example 6.1 (Shishkin Mesh):")
print("x =", x1_shishkin)
print("U =", U1_shishkin)
print("\nExample 6.1 (Adaptive Mesh):")
print("x =", x1_adaptive)
print("U =", U1_adaptive)

print("\nExample 6.2 (Shishkin Mesh):")
print("x =", x2_shishkin)
print("U =", U2_shishkin)
print("\nExample 6.2 (Adaptive Mesh):")
print("x =", x2_adaptive)
print("U =", U2_adaptive)
