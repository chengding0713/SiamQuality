import torch
import torch.nn as nn
import torch.nn.functional as F 
from .. import resnet
import losses 

class Simclr(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=2048, predictor=True, encoder = 'ResNet50'):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Simclr, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = getattr(resnet, encoder, None)(num_classes=dim)
        self.pred_dim = dim // 2

        if predictor:
        # build a 2-layer projection head

            self.predictor = nn.Sequential(nn.Linear(dim, self.pred_dim, bias=False),
                                            nn.BatchNorm1d(self.pred_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(self.pred_dim, dim)) # output layer



    def forward(self, PPG_G, PPG_B):
        """
        Input:
            PPG_G: first views of images
            PPG_B: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        # compute features for one view
        h1, _ = self.encoder(PPG_G) # NxC
        h2, _ = self.encoder(PPG_B) # NxC

        z1 = self.predictor(h1) # NxC
        z2 = self.predictor(h2) # NxC

        criterion = losses.NtXentLoss()
        loss = criterion(z1, z2)

        return loss