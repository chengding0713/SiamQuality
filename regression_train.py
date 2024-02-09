import torch, time, os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import losses
import datasets.data_generator as data_generator
from models.regressor import Regressor
from arg_parser import parse_args



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

print(f"Regression Started using {baseline} baseline and {encoder} encoder")

# Backbone Model
model = Regressor(baseline = args.baseline, embedding_size=args.sz_embedding, batch_size=args.sz_batch, weights = args.pretrained_weights, encoder = encoder)

model = model.to(device)

sample_size = args.sample_size
signal_length = args.signal_length

task = 'respiration_rate'
# dataset_tr = data_generator.generate_data(sample_size, signal_length, 1, flag = 'r')
dataset_tr = data_generator.load_npy("X_train.npy", "y_train.npy",flag=True)
dataset_val = data_generator.load_npy("X_test.npy", "y_test.npy",flag=True)

dataloader_val = DataLoader(dataset_val, batch_size=args.sz_batch, shuffle=True)
dataloader_tr = DataLoader(dataset_tr, batch_size=args.sz_batch, shuffle=True)

param_groups = [
    {'params': list(set(model.parameters()))}
]

if args.loss == 'MseLoss':
    criterion = losses.MeanSquaredError().to(device)
else:
    criterion = losses.MeanSquaredError().to(device)

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



for epoch in range(0, args.nb_epochs):
    model.train()
    losses_per_epoch = []
    
    # unfreeze_model_param = list(model.predictor.parameters())
    
    # if epoch == 0:
    #     for param in list(model.parameters()):
    #         param.requires_grad = False
        # for param in list(set(unfreeze_model_param)):
        #     param.requires_grad = True
    
    pbar = tqdm(enumerate(dataloader_tr))
    
    for batch_idx, (data, label) in pbar:
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32)                     
        x = model(data)
        loss = criterion(x, label.unsqueeze(1))
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.cpu().data.numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dataloader_tr),
                100. * batch_idx / len(dataloader_tr),
                loss.item()))


    # wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()
    
    # Validation
    model_is_training = model.training
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []
    with torch.no_grad():  # Disable gradient computation for validation
        pbar = tqdm(enumerate(dataloader_val))
        for batch_idx, (data, label) in pbar:  # Iterate through your validation DataLoader
            data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32)                     
            outputs = model(data)
            true_labels.extend(label.cpu().data.numpy())
            predicted_labels.extend(outputs.cpu().data.numpy())

    mae = mean_absolute_error(true_labels, predicted_labels)
    mse = mean_squared_error(true_labels, predicted_labels)
    rmse = mean_squared_error(true_labels, predicted_labels, squared=False)
    r2 = r2_score(true_labels, predicted_labels)
    losses_list.append(mae)
    print("losses_list: ", losses_list)
    print(f'Epoch {epoch + 1}/{args.nb_epochs}, Validation MAE: {mae:.4f}, '
          f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}')

    model.train()
    model.train(model_is_training)
    
    if losses_list[-1] == min(losses_list):
        print("Model is going to save")
        print(f"last loss: {losses_list[-1]} | min loss: {min(losses_list)}")
        if not os.path.exists('{}'.format(LOG_DIR)):
            os.makedirs('{}'.format(LOG_DIR))
        torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_{}_{}_{}_best.pth'.format(LOG_DIR, args.baseline, args.model, args.sz_embedding, args.loss,task))