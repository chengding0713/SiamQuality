import torch, math, time, os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import *
import losses
import datasets.data_generator as data_generator

from models.baselines import byol, moco, simclr, swav
from datasets.ppg import PPGDataset
from arg_parser import parse_args

import sys

args = parse_args()

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(int(time.time()),args.baseline, args.model, args.loss, args.sz_embedding, args.alpha, 
                                                                                            args.mrg, args.optimizer, args.lr, args.sz_batch, args.remark)


if torch.cuda.is_available() and args.gpu_id != -1:
    device = 'cuda'
    torch.cuda.set_device(args.gpu_id)
else:
    device = 'cpu'

print(f"device is {device}")
# Baseline Model
baseline = args.baseline

# encoder
encoder = args.encoder

# input file path
input_x = args.input_pair_path

print(f"Training Started for {baseline} baseline")

if baseline == 'byol':
    model = byol.BYOL(image_size=1200, projection_size=args.sz_embedding, encoder = encoder)
    dataset = data_generator.load_npy(path_x = input_x)
    dataset_val = data_generator.load_npy(path_x = '../data/ppg_data_val.npy')
elif baseline == 'simclr':
    model = simclr.Simclr(dim = args.sz_embedding, encoder = encoder)
    dataset = data_generator.get_augmented_data(path_x = input_x)
    dataset_val = data_generator.get_augmented_data(path_x = '../data/ppg_data_val.npy')
elif baseline == 'moco':
    model = moco.MoCo(dim = args.sz_embedding, encoder = encoder)
    dataset = data_generator.get_augmented_data(path_x = input_x)
    dataset_val = data_generator.get_augmented_data(path_x = '../data/ppg_data_val.npy')
elif baseline == 'swav':
    model = swav.SwAV(output_dim = args.sz_embedding, nmb_prototypes = 100, hidden_mlp = 128, queue_size = args.sz_batch * 2, encoder = encoder)
    dataset = data_generator.get_augmented_data(path_x = input_x, swav = True)
    dataset_val = data_generator.get_augmented_data(path_x = '../data/ppg_data_val.npy', swav = True)
else:
    sys.exit("No proper baseline defined. Please define the correct baseline in the arg_parser.py")

# if args.gpu_id == -1:
#     model = nn.DataParallel(model)

model = model.to(device)

sample_size = args.sample_size
signal_length = args.signal_length
 
# dataset = data_generator.generate_save(sample_size, signal_length, 2)

dataloader = DataLoader(dataset, batch_size=args.sz_batch, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=args.sz_batch, shuffle=True)

param_groups = [
    {'params': list(set(model.parameters()))}
]

if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

losses_list = []

torch.autograd.set_detect_anomaly(True)

for epoch in range(0, args.nb_epochs):
    model.train()
    losses_per_epoch = []
    
    pbar = tqdm(enumerate(dataloader))
    
    for batch_idx, batch in pbar:
        if baseline == 'swav':
            views = [view.to(device, dtype=torch.float32).unsqueeze(dim=1) for view in batch]
            loss= model(*views)
        else:
            x1, x2 = batch
            x1, x2 = x1.to(device, dtype=torch.float32), x2.to(device, dtype=torch.float32)                     
            loss = model(x1.unsqueeze(dim=1), x2.unsqueeze(dim=1))
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        losses_per_epoch.append(loss.cpu().data.numpy())
        opt.step()

        if baseline == 'moco':
            # momentum update for key encoder
            model._momentum_update_key_encoder()


        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dataloader),
                100. * batch_idx / len(dataloader),
                loss.item()))

    losses_list.append(np.mean(losses_per_epoch))
    print("losses_list: ", losses_list)
    # wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()

     Validation
    model_is_training = model.training
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        losses_val = []
        pbar = tqdm(enumerate(dataloader_val))
        for batch_idx, batch in pbar:
            if baseline == 'swav':
                views = [view.to(device, dtype=torch.float32).unsqueeze(dim=1) for view in batch]
                loss= model(*views)
            else:
                x1, x2 = batch
                x1, x2 = x1.to(device, dtype=torch.float32), x2.to(device, dtype=torch.float32)
                loss = model(x1.unsqueeze(dim=1), x2.unsqueeze(dim=1))
            losses_val.append(loss.cpu().data.numpy())

        print(f"Validation loss {np.mean(losses_val)}")

    model.train()
    model.train(model_is_training)
    
    if losses_list[-1] == min(losses_list):
        print("Model is going to save")
        print(f"last loss: {losses_list[-1]} | min loss: {min(losses_list)}")
        if not os.path.exists('{}'.format(LOG_DIR)):
            os.makedirs('{}'.format(LOG_DIR))
        # torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_{}_{}_best.pth'.format(LOG_DIR, args.baseline, args.model, args.sz_embedding, args.loss))
        torch.save({'model_state_dict': model.state_dict()},
                   '{}/{}_{}_{}_{}_{}.pth'.format(LOG_DIR, args.baseline, args.model, args.sz_embedding, args.loss, epoch))

    