import torch

torch.manual_seed(10)

# ====================================== retain_graph ==============================================
# notes: tensor.backward(retain_graph=True)指明反向传播后不要释放计算图,否则不能多次调用backward函数
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)
    print(w.grad)
    y.backward()
    print(w.grad)

# ====================================== grad_tensors ==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)  # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)
    y1 = torch.add(a, b)  # y1 = (x+w) + (w+1)    dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)  # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)  # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad)

# ====================================== autograd.gard ==============================================
# flag = True
flag = False
if flag:
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)  # y = x**2

    # 这里设置create_graph=True是为了下一步计算梯度,这里不释放计算图
    grad_1 = torch.autograd.grad(y, x, create_graph=True)  # grad_1 = dy/dx = 2x = 2 * 3 = 6
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)  # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)

# ====================================== tips: 1 ==============================================
# notes: 梯度不会自动清空,每次更新都要调用tensor.grad.zero_(),zero_()是inplace操作
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        w.grad.zero_()

# ====================================== tips: 2 ==============================================
# notes: 非叶子节点的requires_grad属性默认都是True,因为求叶子节点的梯度要用到
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)

# ====================================== tips: 3 ==============================================
# flag = True
flag = False
if flag:
    a = torch.ones((1,))
    print(id(a), a)

    # a = a + torch.ones((1, ))
    # print(id(a), a)

    a += torch.ones((1,))
    print(id(a), a)

# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)

    y.backward()
