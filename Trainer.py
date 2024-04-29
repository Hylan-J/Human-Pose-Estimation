import os

import torch
from progress.bar import Bar

from utils import AverageMeter, adjust_learning_rate


class Trainer:
    def __init__(self, model, optimizer, loss, metrics, file, config):
        super(Trainer, self).__init__()
        self.loss = None
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss
        self.metrics = metrics
        self.file = file
        self.config = config
        self.gpu = config.gpuid
        self.model = self.model.to('cuda')

    def train(self, dataloader, start_epoch, end_epoch):
        """
        训练模型
        """
        # 模型训练模式
        self.model.train()

        for epoch in range(start_epoch, end_epoch + 1):
            self.loss = AverageMeter()
            self.loss.reset()
            for key, value in self.metrics.items():
                setattr(self, key, AverageMeter())
            for key, value in self.metrics.items():
                getattr(self, key).reset()

            nIters = len(dataloader)
            bar = Bar('==>', max=nIters)

            for batch_idx, (data, target, meta1, meta2) in enumerate(dataloader):

                data = data.to('cuda', non_blocking=True).float()
                target = target.to('cuda', non_blocking=True).float()
                output = self.model(data)

                loss = self.loss_fun(output, target, meta1.to('cuda', non_blocking=True).float().unsqueeze(-1))
                self.loss.update(loss.item(), data.shape[0])

                self.eval_metrics(output, target, meta1, meta2)
                loss.backward()
                if (batch_idx + 1) % self.config.mini_batch_count == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                Bar.suffix = 'Train: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss: {loss.avg:.6f} ({loss.val:.6f})'.format(
                    epoch, batch_idx + 1, nIters, total=bar.elapsed_td, eta=bar.eta_td,
                    loss=self.loss) + self.print_metrics()
                bar.next()
            bar.finish()

            with open(self.file, 'a', encoding='UTF8') as file:
                message = '平均 loss: {:8f}\n'.format(self.loss.avg) + ''.join(
                    ['指标 {:10s}: {:4f}\n'.format(key, getattr(self, key).avg) for key, _ in self.metrics.items()])
                file.write(message)
                file.close()

            if epoch % self.config.saveInterval == 0:
                state = {
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                }
                # 保存参数
                torch.save(state, os.path.join(self.config.saveDir, 'model_{}.pth'.format(epoch)))
            adjust_learning_rate(self.optimizer, epoch, self.config.dropLR, self.config.dropMag)

    def evaluate(self, dataloader):
        """
        评估模型
        """
        self.model.eval()

        with torch.no_grad():
            self.memory = AverageMeter()
            self.memory.reset()
            for key, value in self.metrics.items():
                setattr(self, key, AverageMeter())
            for key, value in self.metrics.items():
                getattr(self, key).reset()

            nIters = len(dataloader)
            bar = Bar('==>', max=nIters)

            for batch_idx, (data, target, meta1, meta2) in enumerate(dataloader):
                data = data.to('cuda', non_blocking=True).float()
                target = target.to('cuda', non_blocking=True).float()
                output = self.model(data)

                loss = self.loss_fun(output, target, meta1.to('cuda', non_blocking=True).float().unsqueeze(-1))
                self.loss_fun.update(loss, data.shape[0])

                self.eval_metrics(output, target, meta1, meta2)

                Bar.suffix = 'Eval: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss: {loss.avg:.6f} ({loss.val:.6f})'.format(
                    -1, batch_idx + 1, nIters, total=bar.elapsed_td, eta=bar.eta_td,
                    loss=self.loss_fun) + self.print_metrics()
                bar.next()
            bar.finish()

    def eval_metrics(self, output, target, meta1, meta2):
        for key, value in self.metrics.items():
            value, count = value(output, target, meta1, meta2)
            getattr(self, key).update(value, count)

    def print_metrics(self):
        return ''.join(
            [('| {0}: {metric.avg:.3f} ({metric.val:.3f}) '.format(key, metric=getattr(self, key))) for key, _ in
             self.metrics.items()])
