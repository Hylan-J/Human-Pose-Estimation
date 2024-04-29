import torch.nn as nn
import torchvision


class DeepPose(nn.Module):
    def __init__(self, nJoints, modelName='resnet50'):
        super(DeepPose, self).__init__()
        self.nJoints = nJoints
        self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
        self.resnet = getattr(torchvision.models, modelName)()

        if modelName == 'resnet34':
            num_features = 512
        else:
            num_features = 512 * (4 if self.block == 'BottleNeck' else 1)
        self.resnet.fc = nn.Linear(num_features, self.nJoints * 2)

    def forward(self, x):
        output = self.resnet(x)
        output = output.view(-1, self.nJoints, 2)  # Reshape output to match the expected shape
        return output
