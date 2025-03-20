import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# set seed
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1000)  # 为节省时间，这里用 1000 而非 10000
parser.add_argument('--lr', type=float, default=0.01) 
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100, 
                      start=0.,
                      stop=1,  
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # 生成时间戳
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # 生成顺时针和逆时针螺旋
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('/workspace/ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('/workspace/ground_truth.png'))

    # 对每条螺旋采样观测点
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # 不从开始或末尾附近采样 t0
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # 随机选择旋转方向
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.val = None
        self.avg = 0
    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -0.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - 0.5
    return kl


if __name__ == '__main__':
    # 固定参数
    nsample = 100
    nspiral = 1000
    ntotal = 1000
    start = 0.0
    stop = 6 * np.pi
    a = 0.0
    b = 0.3
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    niters = args.niters
    lr = args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 变化参数：不同的观测噪声水平
    noise_stds = [0.1, 0.3, 0.5, 1.0]

    # 用于保存结果
    results = {
        'noise_std': [],
        'reconstruction_error': [],
        'extrapolation_error': []
    }

    for noise_std in noise_stds:
        print(f"Running experiment with noise_std={noise_std}")

        # 生成数据
        orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
            nspiral=nspiral,
            ntotal=ntotal,
            nsample=nsample,
            start=start,
            stop=stop,
            noise_std=noise_std,
            a=a, b=b,
            savefig=False
        )
        orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
        samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
        samp_ts = torch.from_numpy(samp_ts).float().to(device)

        # 初始化模型
        func = LatentODEfunc(latent_dim, nhidden).to(device)
        rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
        dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
        params = list(func.parameters()) + list(dec.parameters()) + list(rec.parameters())
        optimizer = optim.Adam(params, lr=lr)
        loss_meter = RunningAverageMeter()

        # 训练模型
        for itr in range(1, niters + 1):
            optimizer.zero_grad()
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean

            # 前向求解 ODE
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # 计算损失
            noise_std_tensor = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_tensor).to(device)
            logpx = log_normal_pdf(samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            if itr % 100 == 0:
                print(f"Iter: {itr}, Loss: {loss_meter.avg:.4f}")

        # 生成预测和外推
        with torch.no_grad():
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            # 选择第一条轨迹，并增加一个 batch 维度
            z0 = z0[0].unsqueeze(0)  # 现在 z0 的形状为 (1, latent_dim)

            # 预测轨迹：使用 2000 个点，在 [0, 2pi] 上重构
            pred_ts = torch.from_numpy(np.linspace(0, 2 * np.pi, 2000)).float().to(device)
            # 现在 odeint 返回 (T, 1, latent_dim)，permute 调整为 (1, T, latent_dim)
            pred_z = odeint(func, z0, pred_ts).permute(1, 0, 2)
            pred_x = dec(pred_z).cpu().numpy()

            # 外推轨迹：使用 2000 个点，在 [-pi, 0] 上外推
            extrap_ts = torch.from_numpy(np.linspace(-np.pi, 0, 2000)[::-1].copy()).float().to(device)
            extrap_z = odeint(func, z0, extrap_ts).permute(1, 0, 2)
            extrap_x = dec(extrap_z).cpu().numpy()

        # 为了与原始轨迹比较，截取预测轨迹与外推轨迹前 ntotal (1000) 个点
        orig_traj = orig_trajs[0].cpu().numpy()  # 形状 (1000, 2)
        pred_x_truncated = pred_x[0, :ntotal, :]  # 形状 (1000, 2)
        extrap_x_truncated = extrap_x[0, :ntotal, :]  # 形状 (1000, 2)

        reconstruction_error = np.mean((pred_x_truncated - orig_traj) ** 2)
        extrapolation_error = np.mean((extrap_x_truncated - orig_traj) ** 2)

        results['noise_std'].append(noise_std)
        results['reconstruction_error'].append(reconstruction_error)
        results['extrapolation_error'].append(extrapolation_error)

        print(f"noise_std={noise_std}, Reconstruction Error: {reconstruction_error:.4f}, Extrapolation Error: {extrapolation_error:.4f}")

    # 绘制误差曲线
    plt.figure()
    plt.plot(results['noise_std'], results['reconstruction_error'], 'o-', label='Reconstruction Error')
    plt.plot(results['noise_std'], results['extrapolation_error'], 's-', label='Extrapolation Error')
    plt.xlabel('Noise Std')
    plt.ylabel('MSE')
    plt.title('Spiral reconstructions using a latent ODE with a variable noise_std')
    plt.legend()
    plt.savefig('/workspace/error_curve.png', dpi=500)
    plt.show()
    print('Saved error curve at /workspace/error_curve.png')

