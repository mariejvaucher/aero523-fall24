#!/usr/bin/env python
# coding: utf-8

# # Homework 2 code solution

# ### Finite-difference formulas with JAX

# In this question you are asked to evaluate the coefficients for the first derivative $\frac{\partial u}{\partial x}$ using the Lagrange interpolating polynomials. For this purpose we use the code provided in Week $2$.

# In[1]:


import numpy as np
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("pdf", "svg")

import niceplots

plt.style.use(niceplots.get_style())
colors = niceplots.get_colors_list()


# In[2]:


def get_lagrange_func(xPoints, i):
    """Create a function that computes the ith Lagrange polynomial for a given set of points.

    Parameters
    ----------
    xPoints : list/array of floats
        X coordinate values of the points to be interpolated
    i : int
        The index of the lagrange polynomial to create (must be between 0 and len(x)-1)

    Returns
    -------
    function
        A function that computes the ith Lagrange polynomial for a given x value, e.g. L_i(x)
    """

    def lagrange_poly(x):
        result = 1.0
        N = len(xPoints)
        for j in range(N):
            if j != i:
                result *= (x - xPoints[j]) / (xPoints[i] - xPoints[j])
        return result

    return lagrange_poly


# To obtain a finite difference scheme from the interpolating polynomial, we can simply take the derivative of the polynomial and evaluate it at the point of interest. e.g. for the first derivative:
#
# $$\frac{dp}{dx} = \sum_{i=0}^{N-1} f_i \frac{dL_i}{dx}$$
#
# $$\left.\frac{df}{dx}\right|_{x^*} \approx \left.\frac{dp}{dx}\right|_{x^*} = \left.\frac{dL_0}{dx}\right|_{x^*} f_0 + \left.\frac{dL_1}{dx}\right|_{x^*} f_1 +...$$
#

# First we define the position of the points for which we want to define the Lagrange polynomials.

# In[3]:


h = 1
xPoints = [0, h, 2 * h, 3 * h]


# Then we compute the Lagrange polynomial at each point. Note that we do not need to know $f_0$,$f_1$, etc.

# In[4]:


pol_0 = get_lagrange_func(xPoints, 0)
pol_1 = get_lagrange_func(xPoints, 1)
pol_2 = get_lagrange_func(xPoints, 2)
pol_3 = get_lagrange_func(xPoints, 3)


# Now we want to compute the coefficients for the first derivative so we simply derive each polynomial with JAX.

# In[5]:


coef_0 = jax.grad(pol_0)
coef_1 = jax.grad(pol_1)
coef_2 = jax.grad(pol_2)
coef_3 = jax.grad(pol_3)


# Now we evaluate these coefficient at the point of interest. In our case we want the derivative at $x=0$.

# In[6]:


print(coef_0(0.0))
print(coef_1(0.0))
print(coef_2(0.0))
print(coef_3(0.0))


# In the first problem you should have find the following coefficients: $$a_0=-\frac{11}{6h}, a_1=\frac{3}{h}, a_2=-\frac{3}{2h}, a_3=\frac{1}{3h}$$
# If we take $h=1$ we obtain the values above.

# Now in the second question you are asked to do the same for the second derivative. The process is exactly the same except that now we take the second derivative for each Lagrange polynomial.

# In[7]:


double_coef_0 = jax.grad(jax.grad(pol_0))
double_coef_1 = jax.grad(jax.grad(pol_1))
double_coef_2 = jax.grad(jax.grad(pol_2))
double_coef_3 = jax.grad(jax.grad(pol_3))

print(double_coef_0(0.0))
print(double_coef_1(0.0))
print(double_coef_2(0.0))
print(double_coef_3(0.0))
