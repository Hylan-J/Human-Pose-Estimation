import torch
from torch.utils.data import DataLoader

import datasets
import losses
import metrics
import models


class Builder:
    def __init__(self, opts):
        super(Builder, self).__init__()
        self.config = opts

        if opts.loadModel is not None:
            self.states = torch.load(opts.loadModel)
        else:
            self.states = None

    def get_model(self):
        model = None
        # 获取一个默认的Model对象
        Model = getattr(models, self.config.model)
        # 如果是 StackedHourGlass 对象
        if self.config.model == 'StackedHourGlass':
            # 给定参数进行初始化
            model = Model(self.config.nChannels,
                          self.config.nStack,
                          self.config.nModules,
                          self.config.nReductions,
                          self.config.nJoints)
        # 如果是 ChainedPredictions 对象
        elif self.config.model == 'ChainedPredictions':
            model = Model(self.config.modelName,
                          self.config.hhKernel,
                          self.config.ohKernel,
                          self.config.nJoints)
        # 如果是 DeepPose 对象
        elif self.config.model == 'DeepPose':
            model = Model(self.config.nJoints,
                          self.config.baseName)
        # 如果是 PyraNet 对象
        elif self.config.model == 'PyraNet':
            model = Model(self.config.nChannels,
                          self.config.nStack,
                          self.config.nModules,
                          self.config.nReductions,
                          self.config.baseWidth,
                          self.config.cardinality,
                          self.config.nJoints,
                          self.config.inputRes)
        # 如果是 PoseAttention 对象
        elif self.config.model == 'PoseAttention':
            model = Model(self.config.nChannels,
                          self.config.nStack,
                          self.config.nModules,
                          self.config.nReductions,
                          self.config.nJoints,
                          self.config.LRNSize,
                          self.config.IterSize)
        else:
            assert 'Not Implemented Yet!!!'
        if self.states is not None:
            model.load_state_dict(self.states['model_state'])
        return model

    def get_loss(self):
        """
        获取loss函数
        """
        # 实例化Loss对象obj
        obj = losses.Loss(self.config)
        # 获取对应的loss函数fun
        fun = getattr(obj, self.config.model)
        return fun

    def get_metric(self):
        PCKhinstance = metrics.PCKh(self.config)
        PCKinstance = metrics.PCK(self.config)

        metric = dict()
        # 如果是 MPII 数据集，那么有 PCK 和 PCKh 两种指标
        if self.config.dataset == 'MPII':
            metric = {'PCK': getattr(PCKinstance, self.config.model),
                      'PCKh': getattr(PCKhinstance, self.config.model)}

        elif self.config.dataset == 'COCO':
            metric = {'PCK': getattr(PCKinstance, self.config.model)}
        return metric

    def get_optimizer(self, Model):
        TrainableParams = filter(lambda p: p.requires_grad, Model.parameters())
        optimizer = getattr(torch.optim, self.config.optimizer_type)(TrainableParams,
                                                                     lr=self.config.LR,
                                                                     alpha=0.99,
                                                                     eps=1e-8)
        if self.states is not None and self.config.loadOptim:
            optimizer.load_state_dict(self.states['optimizer_state'])
            if self.config.dropPreLoaded:
                for i, _ in enumerate(optimizer.param_groups):
                    optimizer.param_groups[i]['lr'] /= self.config.dropMagPreLoaded
        return optimizer

    def get_dataloaders(self):
        train_dataloader = DataLoader(
            dataset=getattr(datasets, self.config.dataset)(self.config, 'train'),
            batch_size=self.config.data_loader_size,
            shuffle=self.config.shuffle,
            pin_memory=not self.config.dont_pin_memory
        )

        val_dataloader = DataLoader(
            dataset=getattr(datasets, self.config.dataset)(self.config, 'val'),
            batch_size=self.config.data_loader_size,
            shuffle=False,
            pin_memory=not self.config.dont_pin_memory
        )
        return train_dataloader, val_dataloader

    def get_current_epoch(self):
        epoch = 1
        if self.states is not None and self.config.loadEpoch:
            epoch = self.states['epoch']
        return epoch
