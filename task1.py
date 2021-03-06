import torch
import itertools


def gd_back(x_start, loss_func, alpha=0.5, beta=0.5, s=1):
    """ Compute minimum using gradient descent with backtracking """
    x = x_start
    while True:
        x.requires_grad = True
        loss = loss_func(x)
        loss.backward()

        # Calculate eta
        with torch.no_grad():
            eta = s
            while True:
                gain = loss_func(x) - loss_func(x - eta * x.grad)
                if gain >= alpha * eta * torch.dot(x.grad, x.grad):
                    break
                else:
                    eta *= beta

            grad = torch.zeros_like(x.grad)
            grad.copy_(x.grad)
            x = x - eta * x.grad

        yield x, grad


def optim(xs, limit=0.1):
    """ Read sequence of updates until gradient norm is less than limit """
    for i, (x, grad) in enumerate(xs):
        grad_norm = torch.sqrt(torch.dot(grad, grad))
        if grad_norm < limit:
            x_opt = x
            break

    print("-----Result-----")
    print(x_opt)
    print(f"{i} iterations")
    print(f"Gradient Norm {grad_norm}\n\n")


# Task 1.1
A = torch.zeros(size=(5, 5))
for i, j in itertools.product(range(5), repeat=2):
    A[i, j] = 1 / (i + j + 1)

print("With alpha=0.5, beta=0.5, s=1")
xs = gd_back(torch.Tensor([1, 2, 3, 4, 5]),
             lambda x: torch.dot(x, torch.mv(A, x)))
optim(xs)

print("With alpha=0.1, beta=0.1, s=1")
xs = gd_back(torch.Tensor([1, 2, 3, 4, 5]),
             lambda x: torch.dot(x, torch.mv(A, x)), alpha=0.1, beta=0.1)
optim(xs)

# Task 1.2
print("Min x^2 + 2y^2")
xs = gd_back(torch.Tensor([2, 1]),
             lambda x: x[0].pow(2) + 2 * x[1].pow(2),
             alpha=0.25, beta=0.5, s=2)
optim(xs, limit=10 ** -5)

print("Min x^2 + y^2 / 100")
xs = gd_back(torch.Tensor([0.01, 1]),
             lambda x: x[0].pow(2) + 0.01 * x[1].pow(2),
             alpha=0.25, beta=0.5, s=2)
optim(xs, limit=10 ** -5)
