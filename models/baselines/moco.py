import torch
import torch.nn.functional as F
from .. import resnet
import losses

class MoCo(torch.nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, use_mlp=True, encoder = 'ResNet50'):
        super(MoCo, self).__init__()
        
        self.K = K
        self.T = T
        self.m = m
        self.use_mlp = use_mlp

        # create the encoders
        self.encoder_q = getattr(resnet, encoder, None)(num_classes=dim)
        self.encoder_k = getattr(resnet, encoder, None)(num_classes=dim)
        
        if self.use_mlp:
            self.mlp_head_q = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
            self.mlp_head_k = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, x_q, x_k):
        q, _ = self.encoder_q(x_q)
        with torch.no_grad():
            k, _ = self.encoder_k(x_k)

        if self.use_mlp:
            q = self.mlp_head_q(q)
            k = self.mlp_head_k(k)

        k = k.detach()

        # compute logits
        N, C = q.shape
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # Nx1
        l_neg = torch.mm(q, self.queue)  # NxK
        logits = torch.cat([l_pos, l_neg], dim=1)  # Nx(1+K)
        logits = logits / self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.queue.device)

        criterion = losses.CrossEntropy()
        loss = criterion(logits, labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return loss

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Get the device of self.queue
        device = self.queue.device
        
        # Clone the tensor and ensure it's on the same device
        new_queue = self.queue.clone().to(device)
        
        # Modify the clone (also ensuring keys are on the same device)
        new_queue[:, ptr:ptr+batch_size] = keys.T.to(device)
        
        # Replace the original tensor with the modified clone
        self.queue = new_queue
        
        # Update the pointer (ensuring it's also on the same device)
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr




#### Train ###

# import torch.optim as optim

# def train_moco(model, train_loader, epochs=100, lr=0.03):
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
#     criterion = torch.nn.CrossEntropyLoss()

#     model = model.cuda()

#     for epoch in range(epochs):
#         total_loss = 0.0
#         for idx, (images, _) in enumerate(train_loader):  # we don't need labels for MoCo
#             x1, x2 = images  # assuming the dataset returns pairs of augmented images
#             x1, x2 = x1.cuda(), x2.cuda()

#             optimizer.zero_grad()
#             logits, labels = model(x1, x2)
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()

#             # momentum update for key encoder
#             model._momentum_update_key_encoder()

#             total_loss += loss.item()

#             if idx % 20 == 0:
#                 print(f"Epoch [{epoch + 1}/{epochs}], Step [{idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch [{epoch +_momentum_update_key_encoder 1}/{epochs}], Average Loss: {avg_loss:.4f}")

#     print("Training completed!")

# # Usage example
# # Assuming you have defined your data loader 'train_loader' and the base_encoder

# base_encoder = ...  # Define your base encoder here
# moco_model = MoCo(base_encoder, use_mlp=True)  # Set use_mlp=True if you want the MLP head
# train_moco(moco_model, train_loader)
