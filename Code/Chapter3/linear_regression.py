import random
import torch
from d2l import torch as d2l


# Generating the Dataset
def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成X
    y = torch.matmul(X, w) + b  # 生成y
    y += torch.normal(0, 0.01, y.shape)  # 加入噪声
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])  # 真实的w
true_b = 4.2  # 真实的b
features, labels = synthetic_data(true_w, true_b, 1000)  # 生成数据集
print('features:', features[0], '\nlabel:', labels[0])  # 打印第一个样本

# 生成第二个特征和标签
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)  # 画散点图


# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 样本数
    indices = list(range(num_examples))  # 生成索引
    random.shuffle(indices)  # 打乱索引
    for i in range(0, num_examples, batch_size):  # 每次取batch_size个样本
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])  # 取出索引
        yield features[batch_indices], labels[batch_indices]  # 返回数据


batch_size = 10  # 批量大小
for X, y in data_iter(batch_size, features, labels):  # 读取数据
    print(X, '\n', y)  # 打印数据
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 初始化w
b = torch.zeros(1, requires_grad=True)  # 初始化b


# 定义模型
def linreg(X, w, b):  # 线性回归模型
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):  # 平方损失
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):  # 小批量随机梯度下降
    with torch.no_grad():  # 不计算梯度
        for param in params:  # 遍历参数
            param -= lr * param.grad / batch_size  # 更新参数
            param.grad.zero_()  # 梯度清零


# 训练模型
lr = 0.03  # 学习率
num_epochs = 3  # 迭代次数
net = linreg  # 线性回归模型
loss = squared_loss  # 平方损失

for epoch in range(num_epochs):  # 训练模型
    for X, y in data_iter(batch_size, features, labels):  # 读取数据
        batch_loss = loss(net(X, w, b), y)  # 计算损失
        batch_loss.sum().backward()  # 反向传播
        sgd([w, b], lr, batch_size)  # 更新参数
    with torch.no_grad():  # 不计算梯度
        train_l = loss(net(features, w, b), labels)  # 计算损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印损失

print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')  # 打印w误差
print(f'error in estimating b: {true_b - b}')  # 打印b误差
