import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .config import SEED, WINDOW

#Scikit learn for Logistic Regression and Random Forest; PyTorch for LSTM
#scaler lives in the model object so that we can transform new data in predict_proba()
def build_logreg(C=0.1):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=2000))])

#500-tree Random Forest with max_depth and min_samples_leaf as hyperparameters
def build_rf(max_depth=5, min_samples_leaf=20):
    return RandomForestClassifier(n_estimators=500, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  n_jobs=-1, random_state=SEED)
#small hyperparams lists to search over for walk-forward CV; can be expanded
LOGREG_GRID = [{"C": c} for c in [0.001, 0.01, 0.1, 1, 10]]
RF_GRID = [{"max_depth": d, "min_samples_leaf": l} for d in [3, 5, 8] for l in [10, 30]]

#neural network LSTM classifier
#works by sliding a WINDOW-sized look-back over the features, and predicting the next-day target
class TorchLSTM:
    #Initializes the LSTM model with hyperparameters for window size, hidden layer size, dropout rate, 
    # learning rate, number of epochs, early stopping patience, batch size, and random seed.
    def __init__(self, window=WINDOW, hidden=32, dropout=0.3, lr=1e-3,
                 epochs=150, patience=15, batch=64, seed=SEED):
        self.window, self.hidden, self.dropout = window, hidden, dropout
        self.lr, self.epochs, self.patience, self.batch, self.seed = lr, epochs, patience, batch, seed
    #Transforms the input features into sequences of shape (num_samples, window_size, num_features) for 
    # LSTM input
    def _seq(self, X, history=None):
        full = X if history is None else pd.concat([history.tail(self.window - 1), X])
        arr = np.clip(self.scaler.transform(full.values), -5, 5).astype("float32")
        w = self.window
        idx = full.index[w - 1:]
        seqs = np.stack([arr[i - w + 1:i + 1] for i in range(w - 1, len(arr))])
        return seqs, idx
    #Fits the LSTM model to the training data and evaluates it on the validation data
    # early stopping to prevent overfitting. 
    # best model state is saved and returned.
    def fit(self, Xtr, ytr, Xva, yva):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.torch = torch
        self.scaler = StandardScaler().fit(Xtr.values)
        Xt, it = self._seq(Xtr); yt = ytr.loc[it].values.astype("float32")
        Xv, iv = self._seq(Xva, history=Xtr); yv = yva.loc[iv].values.astype("float32")
        #Class Net will be used to define the architecture of the LSTM model, 
        # number of input features, hidden layer size, dropout rate, and output layer
        class Net(nn.Module):
            def __init__(s, n_in, h, p):
                super().__init__()
                s.lstm = nn.LSTM(n_in, h, batch_first=True)
                s.drop = nn.Dropout(p); s.out = nn.Linear(h, 1)
            # Defines the forward pass of the LSTM model
            # takes an input sequence, returns the output of the model
            def forward(s, x):
                o, _ = s.lstm(x)
                return s.out(s.drop(o[:, -1])).squeeze(-1)
        #Initializes the LSTM model, optimizer, and loss function.
        self.net = Net(Xt.shape[2], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4) #Adam optimizer with weight decay for regularization
        lossf = nn.BCEWithLogitsLoss() #Binary Cross-Entropy loss with logits for binary classification
        Xt_t, yt_t = torch.tensor(Xt), torch.tensor(yt) #Converts the training and validation data into PyTorch tensors for model training and evaluation
        Xv_t, yv_t = torch.tensor(Xv), torch.tensor(yv)  
        best, best_state, bad = np.inf, None, 0
        #Trains the LSTM model for a specified number of epochs, using mini-batches of data
        #After each epoch, the model is evaluated on the validation set and early stopping is applied if the validation loss 
        # does not improve for a specified number of epochs
        for ep in range(self.epochs):
            #Sets the model to training mode, shuffles the training data
            self.net.train()
            #Iterates over the training data in mini-batches, computes the loss, performs backpropagation, and updates the model params
            perm = torch.randperm(len(Xt_t))
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                lossf(self.net(Xt_t[b]), yt_t[b]).backward()
                opt.step()
            self.net.eval() #Sets the model to evaluation mode
            with torch.no_grad(): #Disables gradient computation for evaluation, computes the validation loss
                vl = lossf(self.net(Xv_t), yv_t).item()
            #Implements early stopping --> if the validation loss improves, save the model state; if not, increment the bad counter
            # If the bad counter exceeds the patience, stop training
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        #Loads the best model state and returns the trained model
        self.net.load_state_dict(best_state)
        self.best_val_loss = best
        return self
    #Predicts probabilities for the input features using the trained LSTM model
    def predict_proba(self, X, history) -> pd.Series:
        X_, idx = self._seq(X, history=history)
        self.net.eval()
        # runs the forward pass of the LSTM model on the input data and applies the sigmoid function
        with self.torch.no_grad():
            p = self.torch.sigmoid(self.net(self.torch.tensor(X_))).numpy()
        return pd.Series(p, index=idx)