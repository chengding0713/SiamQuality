import numpy as np
from scipy.signal import resample
import torch
from datasets.ppg import PPGDataset, PPGDataClassifier
from datasets.augmentation import AugmentedDataset

def generate_data(N = 1000, signal_length = 1200, num_channels = 1, flag = 'c'):
    # Generate random PPG data and labels
    for i in ['train', 'val', 'test']:
        ppg_data = np.random.rand(N, num_channels, signal_length)
        
        if (flag == 'c'):
            num_zeros = N // 2
            num_ones = N - num_zeros

            labels = np.concatenate([np.zeros(num_zeros), np.ones(num_ones)])
            np.random.shuffle(labels)
        
        elif (flag == 'r'):
            labels = np.random.randint(1, 51, size=N)
            
        np.save(f'../data/{flag}_{i}_x.npy', ppg_data)
        np.save(f'../data/{flag}_{i}_y.npy', labels)
        
        if (i == 'train'):
            ppg_tensor = torch.tensor(ppg_data)
            labels = torch.tensor(labels)
            ppg_dataset = PPGDataClassifier(ppg_tensor, labels)
    
    return ppg_dataset

def generate_save(N = 1000, signal_length = 1200, num_channels = 2):
    # Generating random PPG signal data
    ppg_data = np.random.rand(N, num_channels, signal_length)
    
    # Saving the data as a .npy file
    np.save("../data/ppg_data.npy", ppg_data)
    
    ppg_tensor = torch.tensor(ppg_data, dtype=torch.float32)
    ppg_dataset = PPGDataset(ppg_tensor)
    
    return ppg_dataset

def load_npy(path_x, path_y = False, flag = False):
    if flag:
        ppg_data = resample_data(np.load(path_x))
        labels = np.load(path_y)
        ppg_tensor = torch.tensor(ppg_data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        ppg_dataset = PPGDataClassifier(ppg_tensor, labels)
    else:
        # loaded_data = resample_data(np.load(path_x)[:, None, :])
        loaded_data = np.load(path_x)
        ppg_tensor = torch.tensor(loaded_data, dtype=torch.float32)
        ppg_dataset = PPGDataset(ppg_tensor)
    
    return ppg_dataset

def get_augmented_data(path_x, augmentation_type = 'random', specified_augmentations = [], swav = False):
    ppg_data = np.load(path_x)
    reshaped_data = ppg_data.reshape(-1, 1200)
    ppg_dataset = AugmentedDataset(reshaped_data, augmentation_type=augmentation_type, specified_augmentations=specified_augmentations, swav = swav)
    return ppg_dataset

def resample_data(data):
    if len(data.shape) == 2:
        N, M = data.shape
        resampled_data = np.zeros((N, 1, 1200))

        # Loop through each signal and resample
        for i in range(N):
            resampled_data[i, 0, :] = resample(data[i, :], 1200)
    
    elif len(data.shape) == 3:
        N, _, M = data.shape
        resampled_data = np.zeros((N, 1, 1200))

        for i in range(N):
            resampled_data[i, 0, :] = resample(data[i, 0, :], 1200)

    else:
        raise ValueError("The data doesn't have an expected 2D or 3D shape.")
    
    return resampled_data

    