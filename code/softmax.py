"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
def calculate(x):
    x = np.exp(x)/(np.sum(np.exp(x)))
    return x
def softmax(x):
    x=np.asarray(x)
    """Compute softmax values for each sets of scores in x."""
    if x.size == len(x):
        x = calculate(x)
    else:
        x = x.T
        for row in range(x.shape[0]):
            x[row]= calculate(x[row])
        x = x.T
    return x  # TODO: Compute and return softmax(x)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
