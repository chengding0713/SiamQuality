import torch, time, os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import *
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datasets.data_generator as data_generator
from models.regressor import Regressor
from arg_parser import parse_args

args = parse_args()


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# device = 'cpu'
model = Regressor(baseline = args.baseline, embedding_size=args.sz_embedding, batch_size=args.sz_batch, encoder = args.encoder)
model.load_state_dict(torch.load(args.finetuned_weights)['model_state_dict'])
# weight_file = r'G:\contrastive_152\logs\AAA_simsiam_vital_dbp_resnet152\backbone__embedding256_alpha32_mrg0.1_adam_lr0.0003_batch256\simsiam_backbone_256__bp_vital_best.pth'

data_folder = ''

dataset_test = data_generator.load_npy(data_folder+"X_test.npy", data_folder+"y_test.npy",flag=True)
dataloader_test = DataLoader(dataset_test, batch_size=96, shuffle=False)


model.eval().to(device)  # Set the model to evaluation mode
true_labels = []
predicted_labels = []
with torch.no_grad():  # Disable gradient computation for validation
    pbar = tqdm(enumerate(dataloader_test))
    for batch_idx, (data, label) in pbar:  # Iterate through your validation DataLoader
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.float32)                     
        outputs = model(data)
        true_labels.extend(label.cpu().data.numpy())
        predicted_labels.extend(outputs.cpu().data.numpy())

mae = mean_absolute_error(true_labels, predicted_labels)
mse = mean_squared_error(true_labels, predicted_labels)
rmse = mean_squared_error(true_labels, predicted_labels, squared=False)
r2 = r2_score(true_labels, predicted_labels)
predicted_labels = np.array(predicted_labels)
np.save(data_folder+'predicted_labels.npy', predicted_labels)
print(f'Test MAE: {mae:.4f}, '
      f'MSE: {mse:.4f}, '
      f'RMSE: {rmse:.4f}, '
      f'R^2: {r2:.4f}')