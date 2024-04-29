import torch
from Builder import *
from Trainer import *

import os
import time
import argparse
from opts import opts

opts = opts().parse()
torch.set_default_tensor_type('torch.DoubleTensor' if opts.usedouble else 'torch.FloatTensor')

########################################################################################################################
Builder = Builder(opts)
model = Builder.get_model()
optimizer = Builder.get_optimizer(model)
loss = Builder.get_loss()
metrics = Builder.get_metric()
train_dataloader, val_dataloader = Builder.get_dataloaders()
epoch = Builder.get_current_epoch()
file = os.path.join(opts.saveDir, 'log.txt')
########################################################################################################################


model = model.to('cuda')

Trainer = Trainer(model, optimizer, loss, metrics, file, opts)

if opts.test:
    Trainer.evaluate(val_dataloader)
    exit()

Trainer.train(train_dataloader, epoch, opts.nEpochs)
