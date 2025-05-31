import torch
from torch.nn import functional as F
from model.attention import AttentionBlock
from model.bigcn import BiGCN

class RumorSleuthNet(torch.nn.Module):
    def __init__(self, num_classes:int=4):
        super(RumorSleuthNet, self).__init__()
        self.BiGCN = BiGCN()
        self.CLIP = AttentionBlock()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear(64 * 2, num_classes)

    def forward(self, data):
        x = torch.cat(
            (
                self.BiGCN(data),
                self.CLIP(data.tokens),
            ), dim=1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x