# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Bernoulli
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from datasets import IHDP
from evaluation import Evaluator
import numpy as np
from argparse import ArgumentParser
from utilstorch import get_y0_y1_pytorch, fc_net_pytorch
from evaluation import Evaluator
# 设置参数解析器
parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=10)
parser.add_argument('-earl', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-print_every', type=int, default=10)
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(3407)
np.random.seed(3407)

# 加载数据集
dataset = IHDP(replications=args.reps)
dimx = 25
scores = np.zeros((args.reps, 3))
scores_test = np.zeros((args.reps, 3))

M = None  # batch size during training
d = 200    # latent dimension
lamba = 1e-4  # weight decay
nh, h = 3, 200  # number and size of hidden layers

# 定义模型的神经网络组件
# 定义模型的神经网络组件
class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, t_dim, z_dim):
        super(Encoder, self).__init__()
        # 共享隐藏层
        self.fc1 = fc_net_pytorch(x_dim, [h] * (nh - 1), activation=nn.ELU())
        # q(t|x)
        self.fc_qt = fc_net_pytorch(h, [d], out_layers=[(1, None)], activation=nn.ELU())
        # q(y|x,t)
        self.fc_qy_t0 = fc_net_pytorch(h, [h], out_layers=[(1, None)], activation=nn.ELU())
        self.fc_qy_t1 = fc_net_pytorch(h, [h], out_layers=[(1, None)], activation=nn.ELU())
        # q(z|x,t,y)
        self.fc_qz = fc_net_pytorch(x_dim + y_dim, [h] * (nh - 1), activation=nn.ELU())
        self.fc_muq_t0 = fc_net_pytorch(h, [h], out_layers=[(d, None)], activation=nn.ELU())
        self.fc_muq_t1 = fc_net_pytorch(h, [h], out_layers=[(d, None)], activation=nn.ELU())
        self.fc_sigmaq_t0 = fc_net_pytorch(h, [h], out_layers=[(d, nn.Softplus())], activation=nn.ELU())
        self.fc_sigmaq_t1 = fc_net_pytorch(h, [h], out_layers=[(d, nn.Softplus())], activation=nn.ELU())

    def forward(self, x, y):
        # 计算共享隐藏层输出
        h = self.fc1(x)
        # q(t|x)
        logits_t = self.fc_qt(h)
        qt = Bernoulli(logits=logits_t)
        t_sample = qt.sample()
        # q(y|x,t)
        mu_qy_t0 = self.fc_qy_t0(h)
        mu_qy_t1 = self.fc_qy_t1(h)
        qy = Normal(loc=t_sample * mu_qy_t1 + (1 - t_sample) * mu_qy_t0, scale=1.0)
        # q(z|x,t,y)
        y = y.squeeze(2)
        inpt2 = torch.cat([x, y], dim=1)
        hqz = self.fc_qz(inpt2)
        muq_t0 = self.fc_muq_t0(hqz)
        sigmaq_t0 = self.fc_sigmaq_t0(hqz)
        muq_t1 = self.fc_muq_t1(hqz)
        sigmaq_t1 = self.fc_sigmaq_t1(hqz)
        muq = t_sample * muq_t1 + (1 - t_sample) * muq_t0
        sigmaq = t_sample * sigmaq_t1 + (1 - t_sample) * sigmaq_t0
        qz = Normal(loc=muq, scale=sigmaq)
        return qz, qt, qy


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim_bin, x_dim_cont):
        super(Decoder, self).__init__()
        # 共享隐藏层
        self.fc_px_z = fc_net_pytorch(z_dim, [h] * (nh - 1), activation=nn.ELU())
        # p(x|z)
        self.fc_px_z_bin = fc_net_pytorch(h, [h], out_layers=[(x_dim_bin, None)], activation=nn.ELU())
        self.fc_px_z_cont_mu = fc_net_pytorch(h, [h], out_layers=[(x_dim_cont, None)], activation=nn.ELU())
        self.fc_px_z_cont_sigma = fc_net_pytorch(h, [h], out_layers=[(x_dim_cont, nn.Softplus())], activation=nn.ELU())
        # p(t|z)
        self.fc_pt_z = fc_net_pytorch(z_dim, [h], out_layers=[(1, None)], activation=nn.ELU())
        # p(y|t,z)
        self.fc_py_t0z = fc_net_pytorch(z_dim, [h] * nh, out_layers=[(1, None)], activation=nn.ELU())
        self.fc_py_t1z = fc_net_pytorch(z_dim, [h] * nh, out_layers=[(1, None)], activation=nn.ELU())

    def forward(self, z, t):
        hx = self.fc_px_z(z)
        # p(x|z)
        logits_x_bin = self.fc_px_z_bin(hx)
        x_bin_dist = Bernoulli(logits=logits_x_bin)
        mu_x_cont = self.fc_px_z_cont_mu(hx)
        sigma_x_cont = self.fc_px_z_cont_sigma(hx)
        x_cont_dist = Normal(loc=mu_x_cont, scale=sigma_x_cont)
        # p(t|z)
        logits_t = self.fc_pt_z(z)
        t_dist = Bernoulli(logits=logits_t)
        # p(y|t,z)
        mu_y_t0 = self.fc_py_t0z(z)
        mu_y_t1 = self.fc_py_t1z(z)
        y_dist = Normal(loc=t * mu_y_t1 + (1 - t) * mu_y_t0, scale=1.0)
        return x_bin_dist, x_cont_dist, t_dist, y_dist



# 定义SVI推断过程
def model(x_bin, x_cont, t, y):
    pyro.module("decoder", decoder)
    with pyro.plate("data", x_bin.size(0)):
        z = pyro.sample("z", dist.Normal(torch.zeros([x_bin.size(0), d]), torch.ones([x_bin.size(0), d])).to_event(1))
        x_bin_probs, x_cont_dists, t_probs, y_dists = decoder(z, t)

        # 确保 t 的形状为 [batch_size]
        pyro.sample("x_bin", dist.Bernoulli(logits=x_bin_probs.logits).to_event(1), obs=x_bin)
        pyro.sample("x_cont", dist.Normal(loc=x_cont_dists.mean, scale=x_cont_dists.stddev).to_event(1), obs=x_cont)
        # 如果 t 是二元变量，确保其形状正确
        pyro.sample("t", dist.Bernoulli(logits=t_probs.logits).to_event(1), obs=t.squeeze())
        pyro.sample("y", dist.Normal(loc=y_dists.mean.squeeze(-1), scale=y_dists.stddev.squeeze(-1)).to_event(1), obs=y.squeeze())

def guide(x_bin, x_cont, t, y):
    pyro.module("encoder", encoder)
    qz, qt, qy = encoder(torch.cat([x_bin, x_cont], dim=1), y)
    with pyro.plate("data", x_bin.size(0)):
        z = pyro.sample("z", dist.Normal(loc=qz.mean, scale=qz.stddev).to_event(1))


# 开始训练
for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication {}/{}'.format(i + 1, args.reps))
    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)
    # 定义模型和优化器
    encoder = Encoder(dimx, 1, 1, d)
    decoder = Decoder(d, len(binfeats), len(contfeats))
    optimizer = Adam({"lr": args.lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # 重排序特征
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

    xalltr = np.concatenate([xtr, xva], axis=0)
    talltr = np.concatenate([ttr, tva], axis=0)
    yalltr = np.concatenate([ytr, yva], axis=0)
    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))

    # 标准化 y
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

    # 转换为张量
    xtr_tensor = torch.tensor(xtr, dtype=torch.float32)
    ttr_tensor = torch.tensor(ttr, dtype=torch.float32)
    ytr_tensor = torch.tensor(ytr, dtype=torch.float32)
    xva_tensor = torch.tensor(xva, dtype=torch.float32)
    tva_tensor = torch.tensor(tva, dtype=torch.float32)
    yva_tensor = torch.tensor(yva, dtype=torch.float32)

    n_epoch = args.epochs
    batch_size = 100
    n_iter_per_epoch = int(np.ceil(xtr.shape[0] / batch_size))

    for epoch in range(n_epoch):
        # 训练模式
        encoder.train()
        decoder.train()
        total_loss = 0.0
        perm = np.random.permutation(xtr.shape[0])

        for i in range(n_iter_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            x_batch = xtr_tensor[idx]
            x_bin_batch = x_batch[:, :len(binfeats)]
            x_cont_batch = x_batch[:, len(binfeats):]
            t_batch = ttr_tensor[idx].unsqueeze(1)
            y_batch = ytr_tensor[idx].unsqueeze(1)
             # dictionaries needed for evaluation
            tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
            tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
            f1 = {x_bin_batch: xalltr[:, 0:len(binfeats)], x_cont_batch: xalltr[:, len(binfeats):], t_batch: tr1}
            f0 = {x_bin_batch: xalltr[:, 0:len(binfeats)], x_cont_batch: xalltr[:, len(binfeats):], t_batch: tr0}
            f1t = {x_bin_batch: xte[:, 0:len(binfeats)], x_cont_batch: xte[:, len(binfeats):], t_batch: tr1t}
            f0t = {x_bin_batch: xte[:, 0:len(binfeats)], x_cont_batch: xte[:, len(binfeats):], t_batch: tr0t}

            loss = svi.step(x_bin_batch, x_cont_batch, t_batch, y_batch)
            total_loss += loss
            

        avg_loss = total_loss / xtr.shape[0]

        if epoch % args.print_every == 0:
            print("Epoch {}/{} - Loss: {:.4f}".format(epoch + 1, n_epoch, avg_loss))
            
            y0, y1 = get_y0_y1_pytorch(model, y_batch, f0, f1, shape=yalltr.shape, L=1)
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score_train = evaluator_train.calc_stats(y1, y0)
            rmses_train = evaluator_train.y_errors(y0, y1)

            y0, y1 = get_y0_y1_pytorch(model, y_batch, f0t, f1t, shape=yte.shape, L=1)
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score_test = evaluator_test.calc_stats(y1, y0)


    # 评估模型（省略评估代码，需根据具体实现补充）
    # ...

print('训练完成')
