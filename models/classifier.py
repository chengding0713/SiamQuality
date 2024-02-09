import torch
import torch.nn as nn
import torch.nn.init as init

from .simsiam import SimSiam
from .baselines.byol import BYOL
from .baselines.simclr import Simclr
from .baselines.swav import SwAV
from .baselines.moco import MoCo

class Classifier(nn.Module):
    def __init__(self, baseline, embedding_size, batch_size, weights = False, num_classes = 2, encoder = 'ResNet50'):
        super(Classifier, self).__init__()
        self.model = self.get_pretrained_model(baseline, embedding_size, batch_size, weights, encoder)
        self.num_classes = num_classes
        self.baseline = baseline
        self.classification_head = nn.Linear(embedding_size, num_classes)
        self._initialize_weights()
    
    def forward_baseline(self, x, baseline):
        if baseline == 'simsiam':
            z, _ = self.model.encoder(x)
        elif baseline == 'byol':
            z = self.model.online_encoder(x)
        elif baseline == 'simclr':
            z, _ = self.model.encoder(x)
        elif baseline == 'moco':
            z, _ = self.model.encoder_q(x)
        elif baseline == 'swav':
            z, _ = self.model.model.forward(x)
        
        x = self.classification_head(z)
        return x
    
    
    def forward(self, x):
        x = self.forward_baseline(x, self.baseline)
        return x
    
    def get_pretrained_model(self, baseline, embedding_size, batch_size, weights, encoder):
        if baseline == 'simsiam':
            model = SimSiam(dim = embedding_size, encoder = encoder, projector = True)
        elif baseline == 'byol':
            model = BYOL(image_size=1200, projection_size=embedding_size, encoder = encoder)
        elif baseline == 'simclr':
            model = Simclr(dim = embedding_size, encoder = encoder)
        elif baseline == 'moco':
            model = MoCo(dim = embedding_size, encoder = encoder)   
        elif baseline == 'swav':
            model = SwAV(output_dim = embedding_size, nmb_prototypes = 100, hidden_mlp = 128, queue_size = batch_size * 2, encoder = encoder)
        try:
            if weights:
                state_dict = torch.load(weights)['model_state_dict']
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
        except:
            raise ValueError("Weights are not supported")
        
        return model
    
    def _initialize_weights(self):
        init.kaiming_normal_(self.classification_head.weight, mode='fan_out')
        init.constant_(self.classification_head.bias, 0)