from torch.utils.data import DataLoader

import datasets


def adjust_learning_rate(optimizer, epoch, dropLR, dropMag):
    if epoch % dropLR == 0:
        lrfac = dropMag
    else:
        lrfac = 1
    for i, param_group in enumerate(optimizer.param_groups):
        if lrfac != 1:
            print(
                "Reducing learning rate of group %d from %f to %f" % (i, param_group['lr'], param_group['lr'] * lrfac))
        param_group['lr'] *= lrfac


class AverageMeter:
    """
    计算并存储平均值和当前值
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 0 if self.count == 0 else self.sum / self.count
