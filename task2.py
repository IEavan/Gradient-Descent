import torch
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Task 2
def cost_function(X, labels, w, epsilon=1e-8):
    """ Compute the average cross entropy loss """
    scores = torch.mv(X, w).sigmoid()
    pos = - torch.log(scores + epsilon)
    neg = - torch.log(1 - scores + epsilon)
    return (labels * pos + (1 - labels) * neg).mean()


def gradient_descent(x_start, loss_func, eta):
    """ Standard gradient descent yielding updates as a generator """
    x = x_start
    while True:
        x.requires_grad = True
        loss = loss_func(x)
        loss.backward()

        with torch.no_grad():
            x = x - eta * x.grad

        yield x


data = np.loadtxt("newiris.csv", delimiter=",")
X = np.c_[data[:, 2:-1], np.ones(data.shape[0])]
labels = data[:, -1]
loss_function = functools.partial(cost_function,
                                  torch.Tensor(X), torch.Tensor(labels))
w_start = torch.zeros(3)
ws = gradient_descent(w_start, loss_function, 0.5)
trace = []
print("Finding decision boundary for iris data")
for i, w in enumerate(ws):
    if i % 5 == 0:
        trace.append(w.numpy())
    if (i + 1) % 2000 == 0:
        print(f"Iteration {i + 1} | Loss: {loss_function(w)}")
        print(f"Current W = {w}")
    if i > 20000:
        print("\n---- Final Result -----")
        print(f"{trace[-1]}")
        break

fig = plt.figure()
ax = plt.axes(ylim=(0.5, 3))
ax.scatter(X[:50, 0], X[:50, 1], c='g')
ax.scatter(X[50:, 0], X[50:, 1], c='b')
line, = ax.plot([], [])


def plot_frame(w):
    y_data = [-1, 3]  # From axis size
    x_data = [(-w[1] * y - w[2])/w[0] for y in y_data]
    line.set_data(x_data, y_data)
    return line,


ani = animation.FuncAnimation(fig, plot_frame, trace, blit=True, interval=10)
plt.show()
