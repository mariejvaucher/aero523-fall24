#!/usr/bin/env python
# coding: utf-8

# # Homework 1 - Solution
#
# This first part of Homework 1 focuses on the root finding methods.
#

# #### #1 Bisection
#
# In this question you were asked to plot the function $f(x) = exp(-x^{2})-2x+3$ and then determine the input required for the bisection method. Remember that the goal of the bisection method is to find the root of the function. Therefore you need to have a previous guess for the two inputs $a$ and $b$ that bracket root $x_0$.

# In[1]:


import numpy as np

# Import matplotlib and niceplots for plotting
import matplotlib.pyplot as plt


# Let's first define the function $f$.

# In[2]:


def func(x):
    return np.exp(-(x**2)) - 2 * x + 3


# Then in order to have a good guess for $a$ and $b$ we plot $f$.

# In[3]:


x = np.linspace(0.5, 2.5, 100)
plt.plot(x, func(x))
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.axvline(1, color="green", linestyle="--")
plt.axvline(2, color="green", linestyle="--")
plt.axhline(0, color="red", linestyle="--")
plt.xlim(x.min(), x.max())


# Apparently $[1,2]$ could be a good initial guess.

# In[4]:


a = 1
b = 2


# Now we define the bisection method as given in the Week $1$.

# In[5]:


def bisection(f, a, b, tol=1e-6, max_iterations=100):
    # First, check that a and b do actually bracket a root
    fa = f(a)
    fb = f(b)
    f_history = []

    if fa * fb > 0:
        raise ValueError("Initial interval does not bracket a root.")

    for i in range(max_iterations):
        c = (a + b) / 2
        fc = f(c)
        f_history.append(abs(float(fc)))

        print(f"Iteration {i:02d}: x = {c: 7.6e}, f(x) = {fc: 7.6e}")

        if np.abs(fc) < tol:
            return c, f_history

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (
        None,
        f_history,
    )  # Return None as the root if the root was not found within max_iterations


# And then we run the method for the function we have defined and the initial guess for the bracket. We print the value of $x$ and the residuals $f(x)$ for each iteration.

# In[6]:


root, bisect_history = bisection(func, a, b)
if root is not None:
    print("Root found:", root)
else:
    print("Root not found within max_iterations")


# Let's plot the convergence history.

# In[7]:


fig, ax = plt.subplots()
ax.set_xlabel("Iteration")
ax.set_ylabel("abs(f(x))", rotation="horizontal", ha="right")

# Convergence plots like this look better when the y-axis is logarithmic
ax.set_yscale("log")
ax.plot(bisect_history)


# The residuals converge to zero. The convergence is not regular due to the bisection method: one iteration can be closer to the root that the next one. However it is globally converging by a factor of 3 on average.

# #### #2 Projection Newton-Raphson

# In this problem your are asked to project a point $(x_0,y_0)$ to the curve of the $exp$ function.

# We first define the function.

# In[8]:


def func(x):
    return np.exp(x)


# Then we define coordinated of the point to be projected.

# In[9]:


x0 = 1.5
y0 = 1


# We plot the function and the point.

# In[10]:


x = np.linspace(-1, 2, 100)
plt.plot(x, func(x))
plt.plot(x0, y0, "x")
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.xlim(x.min(), x.max())
plt.ylim(0, 3)


# In order to obtain the projection we define the distance between a point located on the curve of wich coordinates are $(x,f(x))$ and point $(x_0,y_0)$ : $l^{2}(x)=(x-x_0)^{2}+(f(x)-y_0)^{2}$
#

# The idea is to find  a value $x_{min}$ that minimizes the distance (i.e. the value of $l^{2}$) to point $(x_0,y_0)$. Therefore $x_{min}$ is the solution to the following equation : $$\frac{\partial l^{2}}{\partial x}=0$$
# $$2(x-x_0)+2f^{'}(x)(f(x)-y_0)=0$$
# $$2(x-x_0)+2e^{x}(e^{x}-y_0)=0$$
# To solve this equation, we use Newton Raphson method. The method requires an initial guess as an input that should not be too far away from the root. According to the above plot $x_{init}=1.0$ seems to be a good guess. The method also needs the function from which we search the root and its derivatives that we provide in the code below.
#

# In[11]:


def norm(x, x0, y0):
    return (x - x0) ** 2 + (np.exp(x) - y0) ** 2


def norm_prime(x, x0, y0):
    return 2 * (x - x0) + 2 * np.exp(x) * (np.exp(x) - y0)


def norme_double_prime(x, x0, y0):
    return 2 + 2 * (np.exp(x) * (np.exp(x) - y0) + np.exp(x) ** 2)


# We define the Newton-Raphson method exactly as it was done in the code provided in Week $1$ except that we add the parameters $x_0$ and $y_0$ to the function inside the code.

# In[12]:


def newton_raphson(f, xinit, tol=1e-6, max_iterations=100):
    x = xinit
    f_history = []
    for i in range(max_iterations):
        f_val = f(x, x0, y0)
        f_history.append(abs(float(f_val)))
        print(f"Iteration {i:02d}: x = {x: 7.3e}, f(x) = {f_val: 7.3e}")

        # If f(x) is close enough to zero, we are done
        if np.abs(f_val) < tol:
            return x_new, f_history

        f_prime_val = norme_double_prime(x, x0, y0)

        # Otherwise, take a Newton-Raphson step
        x_new = x - f_val / f_prime_val
        x = x_new

    return (
        None,
        f_history,
    )  # Return None if the root was not found within max_iterations


# In[13]:


# Initial guess
xinit = 0.5

root, newton_history = newton_raphson(norm_prime, xinit)
if root is not None:
    print("Root found:", root)
else:
    print("Root not found within max_iterations")


# In order to check that the root find makes sense, we plot $(x_0,y_0)$ and the root found.

# In[14]:


x = np.linspace(-1, 2, 100)
plt.plot(x, func(x))
plt.plot(x0, y0, "x")
plt.plot(root, func(root), "x")
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.xlim(x.min(), x.max())
plt.ylim(0, 3)


# We plot the convergence history as we did for the bisection method.

# In[15]:


fig, ax = plt.subplots()
ax.set_xlabel("Iteration")
ax.set_ylabel("abs(f(x))", rotation="horizontal", ha="right")

# Convergence plots like this look better when the y-axis is logarithmic
ax.set_yscale("log")
ax.plot(newton_history)


# The residuals converge to zero and the tolerance is reached within 2 iterations. As expected the rate of convergence is higher for Newton-Raphson method than for the bisection method. The rate of convergence is not regular through the process, it increases as we get closer to the solution. Newton methods as a quadratic rate of convergence.

# #### #3 Newton-Raphson with JAX

# For the JAX version of the problem we do not compute the function symbolically anymore. We also replace all the numpy instances by JAX numpy. We then need to import the library and ass the following new header.

# In[16]:


import jax.numpy as jnp
import jax


# The norm is defined with JAX numpy.

# In[17]:


def norm(x, x0, y0):
    return (x - x0) ** 2 + (jnp.exp(x) - y0) ** 2


# We derive the norm with the jax.grad function.

# In[18]:


norm_prime = jax.grad(norm)
norme_double_prime = jax.grad(norm_prime)


# Only one change is needed in the Newton function : the condition where the function value is within the tolerance must use jnp.

# In[19]:


def newton_raphson(f, xinit, tol=1e-6, max_iterations=100):
    x = xinit
    f_history = []
    for i in range(max_iterations):
        f_val = f(x, x0, y0)
        f_history.append(abs(float(f_val)))
        print(f"Iteration {i:02d}: x = {x: 7.3e}, f(x) = {f_val: 7.3e}")

        # If f(x) is close enough to zero, we are done
        if jnp.abs(f_val) < tol:
            return x_new, f_history

        f_prime_val = norme_double_prime(x, x0, y0)

        # Otherwise, take a Newton-Raphson step
        x_new = x - f_val / f_prime_val
        x = x_new

    return (
        None,
        f_history,
    )  # Return None if the root was not found within max_iterations


# In[20]:


# Initial guess
x0 = 1.0

root, newton_history = newton_raphson(norm_prime, x0)
if root is not None:
    print("Root found:", root)
else:
    print("Root not found within max_iterations")


# The fact that both methods found the same root is a good indication that derivatives from first method were correctly computed.
