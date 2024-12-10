import torch
import numpy as np

x = torch.rand(2,3)
print(x)

x = torch.zeros(2,3)
print(x)

x = torch.ones(2,3)
print(x)

x = torch.ones(2,3, dtype=torch.float16)
print(x)

x = torch.ones(2,3, dtype=torch.double)
print(x)

x = torch.ones(2,3, dtype=torch.int)
print(x)
print(x.size())

x = torch.tensor([[2.5, 0.1, 0.3],[1.9, 0.7, 0.2]])
print(x)

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)

# add
print("")
z = x + y
print(z)
z = torch.add(x, y)
print(z)
y.add_(x) #in place operation
print(y)

#substraction
print("")
z = y - x
z = torch.sub(y, x)
y.sub_(x)
print(z)
print(y)

# multiply
print("")
z = x * y
z = torch.mul(x, y)
y.mul_(x)
print(z)
print(y)

# divide
print("")
z = y/x
z = torch.div(y, x)
y.div_(x)
print(z)
print(y)


x = torch.rand(5, 3)
print(x)
print(x[:,0])
print(x[0,:])
print(x[0, 0].item()) # only if there is one value

x = torch.rand(4,4)
y = x.view(-1,8)
print(x)
print(y)
print(y.size())


# work with numpy
a = torch.ones(5)
b = a.numpy()
print(a)
print(b)
print(type(b))
# if the tensor and the numpy matrix are both on the CPU, they are are going to share the same memory location
# The change in a will change b
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
a += 1
print(a)
print(b)


if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    y = torch.ones(5)
    y = y.to(device)

    z = x + y
    print(z)

    z = z.to("cpu")
    print(z)

x = torch.ones(5, requires_grad = True)
print(x)

