import torch
import torch.nn as nn
import numpy as np
import sys
def fc_net_pytorch(input_dim, layers, out_layers=None, activation=nn.ReLU(), weight_decay=1e-3, weights_initializer=None):
    """
    构建一个前馈神经网络，类似于 TensorFlow 中的 fc_net 函数。

    参数：
        input_dim (int): 输入特征的维度。
        layers (list): 隐藏层的尺寸列表。
        out_layers (list): 输出层的列表，每个元素是 (out_dim, activation) 的元组。
        activation (nn.Module): 隐藏层使用的激活函数。
        weight_decay (float): L2 正则化系数（在此代码中未直接使用）。
        weights_initializer (callable): 权重初始化函数。

    返回：
        如果 out_layers 为空或为 None，返回包含隐藏层的 nn.Sequential 模型。
        如果指定了 out_layers，返回一个列表，包含针对每个输出层的 nn.Sequential 模型。
    """
    # 构建隐藏层
    modules = []
    prev_dim = input_dim

    for idx, layer_size in enumerate(layers):
        linear_layer = nn.Linear(prev_dim, layer_size)
        if weights_initializer is not None:
            weights_initializer(linear_layer.weight)
        modules.append(linear_layer)
        modules.append(activation)
        prev_dim = layer_size

    # 如果未指定输出层，返回隐藏层模型
    if not out_layers:
        return nn.Sequential(*modules)

    # 构建输出层模型列表
    outputs = []
    for i, (out_dim, out_activation) in enumerate(out_layers):
        layers_out = modules.copy()
        linear_layer = nn.Linear(prev_dim, out_dim)
        if weights_initializer is not None:
            weights_initializer(linear_layer.weight)
        layers_out.append(linear_layer)
        if out_activation is not None:
            layers_out.append(out_activation)
        outputs.append(nn.Sequential(*layers_out))

    # 如果只有一个输出层，直接返回该模型
    return outputs if len(outputs) > 1 else outputs[0]

def get_y0_y1_pytorch(model,y, f0, f1, shape=(), L=1, verbose=True):
    """
    使用模型计算潜在的 y0 和 y1 值。

    参数：
        model (callable): 用于生成 y 的函数，通常是一个 PyTorch 模型或方法。
        f0 (dict): 包含 t=0 时输入数据的字典。
        f1 (dict): 包含 t=1 时输入数据的字典。
        shape (tuple): 输出数组的形状。
        L (int): 采样次数，用于平均。
        verbose (bool): 是否显示进度。

    返回：
        y0 (numpy.ndarray): t=0 时的潜在结果。
        y1 (numpy.ndarray): t=1 时的潜在结果。
    """
    y0 = np.zeros(shape, dtype=np.float32)
    y1 = np.zeros(shape, dtype=np.float32)
    ymean = y.mean()

    for l in range(L):
        if L > 1 and verbose:
            sys.stdout.write('\rSample {}/{}'.format(l + 1, L))
            sys.stdout.flush()

        with torch.no_grad():
            # 使用模型生成 y0 和 y1
            y0_pred = model(ymean,**f0)
            y1_pred = model(ymean,**f1)
            y0 += y0_pred.gpu().numpy() / L
            y1 += y1_pred.gpu().numpy() / L

    if L > 1 and verbose:
        print()
    return y0, y1
def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    ymean = y.mean()
    for l in range(L):
        if L > 1 and verbose:
            sys.stdout.write('\r Sample {}/{}'.format(l + 1, L))
            sys.stdout.flush()
        y0 += sess.run(ymean, feed_dict=f0) / L
        y1 += sess.run(ymean, feed_dict=f1) / L

    if L > 1 and verbose:
        print()
    return y0, y1

