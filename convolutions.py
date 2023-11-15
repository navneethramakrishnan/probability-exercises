# How do you add two random variables following uniform distributions? Or more generally, two random variables with arbitrary distribution? 
# Inspired by https://www.hackerrank.com/challenges/random-number-generator/problem

import numpy as np
import matplotlib.pyplot as plt

def convolution(f, g, x_range, delta):
    result = np.convolve(f(x_range), g(x_range), mode = 'full')*delta
    return result

# Define uniform distribution for some a > 0. This part can be adapted to arbitrary distributions
def uniform_dist(x, a):
    return np.where((x >= 0) & (x <= a), 1/a, 0)

# Set the range of x values, y values and constants
steps = 100001 #Numerical precision depends on this. The extra step is to keep clean numbers in x_range and y_range
x_lim_low = -5
x_lim_upp = 5
a1 = 1
a2 = 1
x_range = np.linspace(x_lim_low, x_lim_upp, steps)
y_range = np.linspace(2*x_lim_low, 2*x_lim_upp, 2*steps - 1) 
delta = (x_range[-1] - x_range[0])/steps


# Perform convolution
convolution_pdf = convolution(lambda x: uniform_dist(x, a1), lambda x: uniform_dist(x, a2), x_range, delta)

# Plot the original functions and the convolution result
plt.plot(x_range, uniform_dist(x_range, a1), label='Uniform from 0 to' + str(a1))
plt.plot(x_range, uniform_dist(x_range, a2), label='Uniform from 0 to' + str(a2))
plt.plot(y_range, convolution_pdf, label='Convolution')
plt.title('Convolution of Continuous Distributions')
plt.legend()
plt.show()

# Get some useful stuff out of the output distribution
convolution_cdf = np.cumsum(convolution_pdf)*delta
plt.plot(y_range, convolution_cdf, label='Convolution CDF')
plt.show()

convolution_mean = np.sum(convolution_pdf*y_range)*delta
median_index = np.searchsorted(convolution_cdf, 0.5)
convolution_median = y_range[median_index]
print(f'Mean = {convolution_mean:.4f}, Median = {convolution_median:.4f}')
