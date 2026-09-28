import numpy as np
import matplotlib.pyplot as plt

# ----- 1. Ham so va dao ham -----
def f(x):
    return x**2 - 4*x + 5

def f_prime(x):
    return 2*x - 4

print("Ham so: f(x) = x^2 - 4x + 5")
print("Dao ham: f'(x) = 2x - 4\n")

# ----- 2 & 3. Gradient Descent: 4 buoc cap nhat -----
x = 5.0        # diem khoi tao x(0)
eta = 0.2      # learning rate
n_steps = 4    # so buoc cap nhat

x_history = [x]
f_history = [f(x)]

print(f"{'Buoc':<6}{'x(t)':<12}{'f_prime(x(t))':<16}{'x(t+1)':<12}{'f(x(t))':<10}")
for t in range(n_steps):
    grad = f_prime(x)
    x_new = x - eta * grad
    print(f"{t:<6}{x:<12.4f}{grad:<16.4f}{x_new:<12.4f}{f(x):<10.4f}")
    x = x_new
    x_history.append(x)
    f_history.append(f(x))

print(f"\nSau {n_steps} buoc: x = {x:.4f}, f(x) = {f(x):.4f}")
print("Diem toi uu ly thuyet: x* = 2, f(x*) = 1")

# ----- 4. Ve bieu do minh hoa su hoi tu -----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Duong cong f(x) va cac buoc GD di qua
x_curve = np.linspace(-1, 6, 400)
axes[0].plot(x_curve, f(x_curve), label='f(x) = x^2 - 4x + 5', color='steelblue')
axes[0].plot(x_history, f_history, 'o--', color='red', label='Gradient Descent path')
for i, (xi, fi) in enumerate(zip(x_history, f_history)):
    axes[0].annotate(f'x{i}', (xi, fi), textcoords="offset points", xytext=(5, 8))
axes[0].scatter([2], [f(2)], color='green', zorder=5, label='Diem toi uu x*=2')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Gradient Descent tren duong cong f(x)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) f(x) giam dan qua tung buoc
axes[1].plot(range(len(f_history)), f_history, 'o-', color='darkorange')
for i, fi in enumerate(f_history):
    axes[1].annotate(f'{fi:.3f}', (i, fi), textcoords="offset points", xytext=(0, 8), ha='center')
axes[1].set_xlabel('Buoc (t)')
axes[1].set_ylabel('f(x(t))')
axes[1].set_title('Gia tri f(x) giam dan qua cac buoc')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gd_326_plot.png', dpi=150)   # luu file anh
plt.show()                              # hien thi bieu do khi chay tren may co GUI