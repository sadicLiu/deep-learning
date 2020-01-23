# PyTorch Cookbook

## 1. 基础配置

### 检查PyTorch版本

```
torch.__version__               # PyTorch version
torch.version.cuda              # Corresponding CUDA version
torch.backends.cudnn.version()  # Corresponding cuDNN version
torch.cuda.get_device_name(0)   # GPU type
```

### 指定程序运行在特定GPU卡上

在命令行指定环境变量
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

或在代码中指定
```
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

### 判断是否有CUDA支持

```
torch.cuda.is_available()
```

### 设置为cuDNN benchmark模式

Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。
```
torch.backends.cudnn.benchmark = True
```

如果想要避免这种结果波动，设置
```
torch.backends.cudnn.deterministic = True
```

### 清除GPU存储

有时Control-C中止运行后GPU存储没有及时释放，需要手动清空。在PyTorch内部可以
```
torch.cuda.empty_cache()
```

或在命令行可以先使用ps找到程序的PID，再使用kill结束该进程
```
ps aux | grep python
kill -9 [pid]
```
或
```
fuser -v /dev/nvidia*
kill 掉所有连号的进程
```

或者直接重置没有被清空的GPU
```
nvidia-smi --gpu-reset -i [gpu_id]
```

## 2. 张量处理

### 张量基本信息

```
tensor.type()   # Data type
tensor.size()   # Shape of the tensor. It is a subclass of Python tuple
tensor.dim()    # Number of dimensions.
```

### 数据类型转换

```
# Set default tensor type. Float in PyTorch is much faster than double.
torch.set_default_tensor_type(torch.FloatTensor)

# Type convertions.
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()
```























