from torch.utils.data import Dataset

class PPGDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x1 = self.data[idx][0]
        x2 = self.data[idx][1]
        return x1, x2
    
class PPGDataClassifier(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return data, labels