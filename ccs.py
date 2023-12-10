import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

EPS = 1e-5

def process_data(saved_names, verbose=False):
    """
    Utility function to load data for CCS
    1. Loads data
    2. Split into train/test split across professions dim
    Output shape:
        hs: (prompts x layers x dim)
        y: (prompts)
    """ 

    path = os.path.join(os.getcwd(), "saved")
    total_neg = []
    total_pos = []
    total_y = []
    # Load all results
    for saved_name in saved_names:
        root = os.path.join(path, saved_name)
        neg = np.load(os.path.join(root, "fem-hs.npy"))
        pos = np.load(os.path.join(root, "male-hs.npy"))
        y = np.load(os.path.join(root, "y.npy"))
        total_neg.append(neg)
        total_pos.append(pos)
        total_y.append(y)

    # Concatenate all filters
    neg_hs_layers = np.concatenate(total_neg, axis=0)
    pos_hs_layers = np.concatenate(total_pos, axis=0)
    y = np.concatenate(total_y, axis=0)

    # Train test split
    # We want to maintain our label proportions
    neg_hs_train, neg_hs_test, pos_hs_train, pos_hs_test, y_train, y_test = train_test_split(neg_hs_layers, 
                                                                                             pos_hs_layers, 
                                                                                             y, 
                                                                                             test_size=0.2, 
                                                                                             random_state=42,
                                                                                             shuffle=True, 
                                                                                             stratify=y)

    if verbose:
        print(neg_hs_train.shape, pos_hs_train.shape, y_train.shape, 
              neg_hs_test.shape, pos_hs_test.shape, y_test.shape)

    return (neg_hs_train, pos_hs_train, y_train), (neg_hs_test, pos_hs_test, y_test)


class Scaler():
    def __init__(self, eps=EPS):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)

    def normalize(self, x):
        """
        Normalizes the data x (of shape (n, d))
        """
        normalized_x = x - self.mean
        normalized_x /= (self.std + EPS)
        return normalized_x
    
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="mps", linear=True, weight_decay=0.01):
        # data
        self.scaler0 = Scaler()
        self.scaler1 = Scaler()
        self.scaler0.fit(x0)
        self.scaler1.fit(x1)

        self.x0 = self.scaler0.normalize(x0)
        self.x1 = self.scaler1.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    

        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss


    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs.
        Because it doesn't know which side of the line is correct we just test its ability to separate them.
        """
        x0 = torch.tensor(self.scaler0.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.scaler1.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)

        return acc
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)
                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    

    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return loss
    
    def forward(self, neg, pos, verbose=False):
        x0 = torch.tensor(self.scaler0.normalize(neg), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.scaler1.normalize(pos), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        if verbose:
            print(p0, p1)
        
        avg_confidence = 0.5*(p0 + (1-p1))
        return avg_confidence.detach().cpu().numpy()

    
    def predict(self, neg, pos):
        avg_confidence = self.forward(neg, pos)
        predictions = (avg_confidence < 0.5).astype(int)[:, 0]
        return predictions