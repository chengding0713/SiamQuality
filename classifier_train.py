import torch, time, os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

import losses
import datasets.data_generator as data_generator
from models.classifier import Classifier
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

# Backbone Model
model = Classifier(baseline = args.baseline, embedding_size=args.sz_embedding, batch_size = args.sz_batch, weights = args.weight_file, num_classes=args.num_classes, encoder = args.encoder)

# if args.gpu_id == -1:
#     model = nn.DataParallel(model)

model = model.to(device)

dataset_tr = data_generator.load_npy("Path_X_train.npy", "Path_Y_train.npy", flag=True)
dataset_val = data_generator.load_npy("Path_X_test.npy", "Path_Y_test.npy", flag=True)



dataloader_val = DataLoader(dataset_val, batch_size=args.sz_batch, shuffle=True)
dataloader_tr = DataLoader(dataset_tr, batch_size=args.sz_batch, shuffle=True)

param_groups = [
    {'params': list(set(model.parameters()))} # only training the classification head -> model.model.classification_head.parameters()
]

if args.loss == 'CrossEntropy':
    criterion = losses.CrossEntropy().to(device)
else:
    criterion = losses.CrossEntropy().to(device)

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
    
    unfreeze_model_param = list(model.parameters()) # model.model.classification_head.parameters()
    
    if epoch == 0:
        # for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
        #     param.requires_grad = False
        for param in list(set(unfreeze_model_param)):
            param.requires_grad = True
    
    pbar = tqdm(enumerate(dataloader_tr))
    
    for batch_idx, (data, label) in pbar:
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.long)                     
        x = model(data)
        loss = criterion(x, label)
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10) # model.model.classification_head.parameters()

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
            data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.long)                     
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(label.cpu().data.numpy())
            predicted_labels.extend(predicted.cpu().data.numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(true_labels, predicted_labels)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    Score = (TP + TN) / (TP + TN + FP + 5 * FN)
    losses_list.append(f1)
    print("losses_list: ", losses_list)
    print(f'Epoch {epoch}, Validation Accuracy: {accuracy * 100:.2f}%, '
          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Score: {Score:.4f}')

    model.train()
    model.train(model_is_training)
    
    if losses_list[-1] == max(losses_list):
        print("Model is going to save")
        print(f"last loss: {losses_list[-1]} | min loss: {max(losses_list)}")
        if not os.path.exists('{}'.format(LOG_DIR)):
            os.makedirs('{}'.format(LOG_DIR))
        torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_{}_{}_{}.pth'.format(LOG_DIR, args.baseline, args.model, args.sz_embedding, args.loss,'best_AF'))
        
    
    