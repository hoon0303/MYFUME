import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(self.y.iloc[idx], dtype=torch.float32)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, dnn_hidden_units):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(dnn_hidden_units)):
            if i == 0:
                layers.append(nn.Linear(input_dim, dnn_hidden_units[i]))
            else:
                layers.append(nn.Linear(dnn_hidden_units[i-1], dnn_hidden_units[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dnn_hidden_units[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze()

def load_data(item_path, review_path):
    item_data = pd.read_csv(item_path)
    review_data = pd.read_csv(review_path)

    data_merge = pd.merge(review_data, item_data, on="N_id")
    data_merge['Target'] = np.where(data_merge['Target'] > 3, 1, 0)

    User_category = data_merge.pivot_table("Target", index="User", columns="Smell", aggfunc="mean")
    User_category_matrix = User_category.fillna(0)

    data_merge = pd.merge(data_merge, User_category_matrix, on="User")

    sparse_features = ['Company', 'Smell', 'Gender', 'Year']
    dense_features = User_category_matrix.columns.tolist()
    for feat in sparse_features:
        lbe = LabelEncoder()
        data_merge[feat] = lbe.fit_transform(data_merge[feat])

    mms = MinMaxScaler()
    data_merge[dense_features] = mms.fit_transform(data_merge[dense_features])

    x = data_merge[sparse_features + dense_features]
    y = data_merge['Target']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=52, shuffle=True, stratify=y)
    
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    return train_data, test_data, sparse_features, dense_features

def prepare_data(train_data, test_data, batch_size):
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(train_loader, input_dim, dnn_hidden_units, epochs, learning_rate):
    model = NeuralNet(input_dim, dnn_hidden_units)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(outputs.numpy())
    
    y_pred_labels = np.where(np.array(y_pred) > 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred_labels)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    log_loss_value = log_loss(y_true, y_pred)

    return accuracy, roc_auc, precision, recall, log_loss_value, y_true, y_pred

def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history['auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

item_path = 'data/item.csv'
review_path = 'data/review.csv'
train_data, test_data, sparse_features, dense_features = load_data(item_path, review_path)

embedding_dims = [4, 8]
batch_sizes = [32, 64, 128, 256]
dnn_hidden_units_list = [[128, 64], [256, 128], [512, 256]]
epochs = 100
learning_rate = 0.001

best_accuracy = 0
best_roc_auc = 0
best_log_loss = float('inf')
best_params = {'embedding_dim': None, 'batch_size': None, 'dnn_hidden_units': None}
best_model = None

input_dim = len(sparse_features + dense_features)

for embedding_dim in embedding_dims:
    for batch_size in batch_sizes:
        for dnn_hidden_units in dnn_hidden_units_list:
            print(f"Training with embedding_dim={embedding_dim}, batch_size={batch_size}, dnn_hidden_units={dnn_hidden_units}")
            train_loader, test_loader = prepare_data(train_data, test_data, batch_size)
            model = train_model(train_loader, input_dim, dnn_hidden_units, epochs, learning_rate)
            accuracy, roc_auc, precision, recall, log_loss_value, y_true, y_pred = evaluate_model(model, test_loader)

            print(f"Accuracy: {accuracy}, ROC-AUC: {roc_auc}, Log Loss: {log_loss_value}")

            if accuracy > best_accuracy and roc_auc > best_roc_auc and log_loss_value < best_log_loss:
                best_accuracy = accuracy
                best_roc_auc = roc_auc
                best_log_loss = log_loss_value
                best_params['embedding_dim'] = embedding_dim
                best_params['batch_size'] = batch_size
                best_params['dnn_hidden_units'] = dnn_hidden_units
                best_params['accuracy'] = accuracy
                best_params['roc_auc'] = roc_auc
                best_params['log_loss_value'] = log_loss_value
                best_model = model

print(f"Best parameters: embedding_dim={best_params['embedding_dim']}, batch_size={best_params['batch_size']}, dnn_hidden_units={best_params['dnn_hidden_units']}")
accuracy, roc_auc, precision, recall, log_loss_value, y_true, y_pred = evaluate_model(best_model, test_loader)
print(f"Final Evaluation - Accuracy: {accuracy}, ROC-AUC: {roc_auc}, Log Loss: {log_loss_value}")
