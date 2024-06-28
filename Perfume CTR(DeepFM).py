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

# Custom Dataset class for loading data
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(self.y.iloc[idx], dtype=torch.float32)

# DeepFM model class
class DeepFM(nn.Module):
    def __init__(self, sparse_input_dims, dense_input_dim, embedding_dim, dnn_hidden_units):
        super(DeepFM, self).__init__()
        
        # Embedding layers for sparse features
        self.embeddings = nn.ModuleList([nn.Embedding(input_dim, embedding_dim) for input_dim in sparse_input_dims])
        
        # Linear part
        self.linear = nn.ModuleList([nn.Embedding(input_dim, 1) for input_dim in sparse_input_dims])
        
        # DNN part
        dnn_input_dim = embedding_dim * len(sparse_input_dims) + dense_input_dim
        layers = []
        for i in range(len(dnn_hidden_units)):
            if i == 0:
                layers.append(nn.Linear(dnn_input_dim, dnn_hidden_units[i]))
            else:
                layers.append(nn.Linear(dnn_hidden_units[i-1], dnn_hidden_units[i]))
            layers.append(nn.ReLU())
        self.dnn = nn.Sequential(*layers)
        self.dnn_output = nn.Linear(dnn_hidden_units[-1], 1)
        
    def forward(self, x_sparse, x_dense):
        # Linear part
        linear_logit = sum([self.linear[i](x_sparse[:, i]) for i in range(x_sparse.shape[1])]).squeeze(1)
        
        # Embedding and interaction part
        embeddings = [self.embeddings[i](x_sparse[:, i]) for i in range(x_sparse.shape[1])]
        fm_logit = sum([torch.sum(embed_i * embed_j, dim=1, keepdim=True) 
                        for i, embed_i in enumerate(embeddings)
                        for j, embed_j in enumerate(embeddings) if i < j]).squeeze(1)
        
        # DNN part
        dnn_input = torch.cat(embeddings + [x_dense], dim=1)
        dnn_logit = self.dnn(dnn_input)
        dnn_logit = self.dnn_output(dnn_logit).squeeze(1)
        
        # Final output
        logit = linear_logit + fm_logit + dnn_logit
        output = torch.sigmoid(logit)
        return output

# Function to load data
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
    sparse_input_dims = []
    for feat in sparse_features:
        lbe = LabelEncoder()
        data_merge[feat] = lbe.fit_transform(data_merge[feat])
        max_val = data_merge[feat].max()
        print(f"{feat}: {len(lbe.classes_)} classes, max value: {max_val}")
        sparse_input_dims.append(len(lbe.classes_) + 1)
        
    mms = MinMaxScaler()
    data_merge[dense_features] = mms.fit_transform(data_merge[dense_features])

    x = data_merge[sparse_features + dense_features]
    y = data_merge['Target']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=52, shuffle=True, stratify=y)
    
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    return train_data, test_data, sparse_features, dense_features, sparse_input_dims

# Function to prepare data loaders
def prepare_data(train_data, test_data, batch_size):
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to train the model
def train_model(train_loader, sparse_input_dims, dense_input_dim, embedding_dim, dnn_hidden_units, epochs, learning_rate):
    model = DeepFM(sparse_input_dims, dense_input_dim, embedding_dim, dnn_hidden_units)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_sparse = X_batch[:, :len(sparse_input_dims)].long()
            X_dense = X_batch[:, len(sparse_input_dims):]
            optimizer.zero_grad()
            outputs = model(X_sparse, X_dense)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model

# Function to evaluate the model
def evaluate_model(model, test_loader, sparse_input_dims):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_sparse = X_batch[:, :len(sparse_input_dims)].long()
            X_dense = X_batch[:, len(sparse_input_dims):]
            outputs = model(X_sparse, X_dense)
            y_true.extend(y_batch.numpy())
            y_pred.extend(outputs.numpy())
    
    y_pred_labels = np.where(np.array(y_pred) > 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred_labels)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    log_loss_value = log_loss(y_true, y_pred)

    return accuracy, roc_auc, precision, recall, log_loss_value, y_true, y_pred

# Main script
item_path = 'data/item.csv'
review_path = 'data/review.csv'
train_data, test_data, sparse_features, dense_features, sparse_input_dims = load_data(item_path, review_path)

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

dense_input_dim = len(dense_features)

print(f"Sparse input dimensions: {sparse_input_dims}")

for embedding_dim in embedding_dims:
    for batch_size in batch_sizes:
        for dnn_hidden_units in dnn_hidden_units_list:
            print(f"Training with embedding_dim={embedding_dim}, batch_size={batch_size}, dnn_hidden_units={dnn_hidden_units}")
            train_loader, test_loader = prepare_data(train_data, test_data, batch_size)
            model = train_model(train_loader, sparse_input_dims, dense_input_dim, embedding_dim, dnn_hidden_units, epochs, learning_rate)
            accuracy, roc_auc, precision, recall, log_loss_value, y_true, y_pred = evaluate_model(model, test_loader, sparse_input_dims)

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
