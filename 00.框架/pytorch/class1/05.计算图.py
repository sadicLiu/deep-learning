import torch

# 叶子节点的梯度在反向传播时被保留,非叶子节点的梯度为了节省内存,默认不保留
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

# 若要保留非叶子节点的梯度,使用retain_grad()方法
a = torch.add(w, x)  # retain_grad()
a.retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
