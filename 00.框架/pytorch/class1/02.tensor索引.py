import torch

torch.manual_seed(1)

# ======================================= example 1 =======================================
# torch.cat,只拼接不增加维度

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))

    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t, t], dim=1)

    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# ======================================= example 2 =======================================
# torch.stack,增加维度

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))

    t_stack = torch.stack([t, t, t], dim=0)

    print("\nt_stack:{} shape:{}".format(t_stack, t_stack.shape))

# ======================================= example 3 =======================================
# torch.chunk,将一个tensor划分成若干个,若维度不能整除,最后一个tensor形状将被缩减

# flag = True
flag = False

if flag:
    a = torch.ones((2, 7))  # 7
    list_of_tensors = torch.chunk(a, dim=1, chunks=3)  # 3

    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

# ======================================= example 4 =======================================
# torch.split,将一个tensor划分成若干个,不同的是自己指定每个tensor的维度

# flag = True
flag = False

if flag:
    t = torch.ones((2, 5))

    list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

# ======================================= example 5 =======================================
# torch.index_select

# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    idx = torch.tensor([0, 2], dtype=torch.long)  # float
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))

# ======================================= example 6 =======================================
# torch.masked_select

# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.le(5)  # ge is mean greater than or equal/   gt: greater than  le  lt
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))

