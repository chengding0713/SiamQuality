import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class NegativeCosineSimLoss(nn.Module):
    def __init__(self):
        super(NegativeCosineSimLoss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)
        
    def forward(self,p1, p2, z1, z2):
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss
    
class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss
        
    
class MeanSquaredError(nn.Module):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        return loss
    
class NtXentLoss(nn.Module):
    def __init__(self):
        super(NtXentLoss, self).__init__()

    def forward(self, out_1, out_2, temperature = 0.5, eps=1e-6):
        out_1 = F.normalize(out_1, dim=-1, p=2)
        out_2 = F.normalize(out_2, dim=-1, p=2)
        
        out_1_dist = out_1
        out_2_dist = out_2
        
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss
    
class ByolLoss(nn.Module):
    def __init__(self):
        super(ByolLoss, self).__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        return loss