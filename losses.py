import torch.nn.functional as F


class Loss:
    """损失函数"""
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config

    def StackedHourGlass(self, output, target, meta=None):
        meta = 1 if self.config.dataset == 'MPII' else meta
        loss = 0
        for i in range(self.config.nStack):
            loss += F.mse_loss(input=output[i] * meta, target=target * meta)
        return loss

    def PoseAttention(self, output, target, meta=None):
        meta = 1 if self.config.dataset == 'MPII' else meta
        loss = 0
        for i in range(self.config.nStack):
            loss += F.mse_loss(output[i] * meta, target * meta)
        return loss

    def PyraNet(self, output, target, meta=None):
        meta = 1 if self.config.dataset == 'MPII' else meta
        loss = 0
        for i in range(self.config.nStack):
            loss += F.mse_loss(output[i] * meta, target * meta)
        return loss

    def ChainedPredictions(self, output, target, meta=None):
        meta = 1 if self.config.dataset == 'MPII' else meta
        return F.mse_loss(output * meta, target * meta)

    def DeepPose(self, output, target, meta=None):
        meta = (target > -0.5 + 1e-8).float().reshape(-1, self.config.nJoints,
                                                      2) if self.config.dataset == 'MPII' else meta[:, :, :, 0]
        return F.mse_loss(output.reshape(-1, self.config.nJoints, 2) * meta,
                          target.reshape(-1, self.config.nJoints, 2) * meta)
