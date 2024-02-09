import torch
import torch.nn.functional as F
import numpy as np
from .. import resnet

class SwAV(torch.nn.Module):
    def __init__(self, output_dim, nmb_prototypes, hidden_mlp, queue_size, epsilon=0.05, temperature=0.1, sinkhorn_iterations=3, nmb_crops=[2, 1], crops_for_assign=[0], encoder = 'ResNetSwav50'):
        super(SwAV, self).__init__()

        self.model = getattr(resnet, encoder, None)(output_dim=output_dim, nmb_prototypes=nmb_prototypes, hidden_mlp=hidden_mlp)
        self.temperature = temperature
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        
        self.nmb_crops = nmb_crops
        self.crops_for_assign = crops_for_assign
        
        # Initialization of the queue
        self.model_device = next(self.model.parameters()).device
        self.queue = torch.zeros(len(nmb_crops), queue_size, output_dim).to(self.model_device)
        self.use_the_queue = False

    def forward(self, *inputs):
        bs = inputs[0].size(0)

        # Normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 1. Compute embeddings and output
        embeddings, output = zip(*[self.model(view) for view in inputs])

      # 2. Use the queue mechanism
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[crop_id][bs * crop_id: bs * (crop_id + 1)]
                
                if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                    self.use_the_queue = True
                    out = torch.cat((self.queue[i].to(self.model_device) @ self.model.prototypes.weight.t().to(self.model_device), out.to(self.model_device)))
                    
                # Fill the queue
                self.queue[i, bs:] = self.queue[i, :-bs].clone()
                self.queue[i, :bs] = embeddings[crop_id][crop_id*bs : (crop_id+1)*bs]
                
                # Compute assignments
                q = torch.exp(out / self.epsilon).t()
                q = SwAV.sinkhorn(q, self.sinkhorn_iterations)[-bs:]
            
            # Compute the loss
            subloss = 0
            for v in np.delete(np.arange(sum(self.nmb_crops)-1), crop_id):
                p = F.softmax(output[v] / self.temperature, dim=1)
                subloss -= torch.mean(torch.sum(q.to(self.model_device) * torch.log(p.to(self.model_device)), dim=1))
            loss += subloss / (sum(self.nmb_crops) - 1)
        
        return loss / len(self.crops_for_assign)

    @staticmethod
    def sinkhorn(Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape
            device = Q.device

            u = torch.zeros(K, device=device)
            r = torch.ones(K, device=device) / K
            c = torch.ones(B, device=device) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
    
    # def update_queue(self, z):
    #     # Compute the number of samples and ensure it fits within the queue
    #     batch_size = z.size(0)
    #     ptr = self.queue_ptr
    #     if ptr + batch_size <= self.queue_size:  # If z fits within the remaining queue
    #         self.queue[ptr: ptr + batch_size] = z
    #     else:  # If z does not fit, split the update
    #         split_idx = self.queue_size - ptr
    #         self.queue[ptr:] = z[:split_idx]  # Fill up to the end of the queue
    #         self.queue[:batch_size - split_idx] = z[split_idx:]  # Fill the remaining from the start

    #     # Update the pointer
    #     self.queue_ptr = (ptr + batch_size) % self.queue_size




