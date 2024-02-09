import torch, time, os
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.classifier import Classifier
import datasets.data_generator as data_generator
from arg_parser import parse_args

args = parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = Classifier(baseline = args.baseline, embedding_size=args.sz_embedding, batch_size = args.sz_batch, num_classes=args.num_classes, encoder = args.encoder)
# model.load_state_dict(torch.load(args.finetuned_weights)['model_state_dict'])
weight_file = r''
weight_file = weight_file.replace('\\', '/')
state_dict = torch.load(weight_file)['model_state_dict']
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

dataset_test = data_generator.load_npy("test.npy", "G:/Downstream_task/AF/stanford_test_label.npy", flag=True)
dataloader_test = DataLoader(dataset_test, batch_size=96, shuffle=False)


model.eval().to(device)  # Set the model to evaluation mode
true_labels = []
predicted_labels = []
with torch.no_grad():  # Disable gradient computation for inference
    pbar = tqdm(enumerate(dataloader_test))
    for batch_idx, (data, label) in pbar:  # Iterate through your validation DataLoader
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.long)                     
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(label.cpu().data.numpy())
        predicted_labels.extend(predicted.cpu().data.numpy())

test_accuracy = accuracy_score(true_labels, predicted_labels)
test_precision = precision_score(true_labels, predicted_labels)
test_recall = recall_score(true_labels, predicted_labels)
test_f1 = f1_score(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
np.save('predicted_labels2.npy', predicted_labels)
Score = (TP + TN) / (TP + TN + FP + 5*FN)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%, '
      f'Precision: {test_precision:.4f}, '
      f'Recall: {test_recall:.4f}, '
      f'F1-score: {test_f1:.4f},'
      f'Score: {Score:.4f}')