import numpy as np
import torch

class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0))
    
    def ate(self, ypred1, ypred0):
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        y1 = np.mean(self.y[idx1])
        y0= np.mean(self.y[idx0])
        yi1 = np.mean(ypred1)
        yi0 = np.mean(ypred0)
        return yi1 - yi0

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = 12122
        ate = self.ate(ypred1, ypred0)
        pehe = 12211
        return ite, ate, pehe


def get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, x_train, t_train, L=1):
    y_infer = q_y_xt_dist(x_train.float(), t_train.float())
    # use inferred y
    xy = torch.cat((x_train.float(), y_infer.mean), 1)  # TODO take mean?
    z_infer = q_z_tyx_dist(xy=xy, t=t_train.float())
    # Manually input zeros and ones
    y0 = p_y_zt_dist(z_infer.mean, torch.zeros(z_infer.mean.shape).cuda()).mean  # TODO take mean?
    y1 = p_y_zt_dist(z_infer.mean, torch.ones(z_infer.mean.shape).cuda()).mean  # TODO take mean?

    return y0.cpu().detach().numpy(), y1.cpu().detach().numpy()
