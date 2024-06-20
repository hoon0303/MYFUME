import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
import tensorflow as tf

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

def prepare_data(train_data, test_data, sparse_features, dense_features, embedding_dim):
    combined_data = pd.concat([train_data, test_data])
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=combined_data[feat].nunique() + 1, embedding_dim=embedding_dim) for feat in sparse_features] + [DenseFeat(feat, 1,) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_input = {name: train_data[name].values for name in feature_names}
    test_input = {name: test_data[name].values for name in feature_names}

    return train_input, test_input, linear_feature_columns, dnn_feature_columns

def train(data, model_input, linear_feature_columns, dnn_feature_columns, batch_size, dnn_hidden_units):
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_hidden_units=dnn_hidden_units)

    metrics = [
        tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
        tf.keras.metrics.AUC(name='auc')
    ]

    class_weight = {0: 1, 1: len(data[data['Target'] == 0]) / len(data[data['Target'] == 1])}

    model.compile("adam", "binary_crossentropy", metrics=metrics)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=10, restore_best_weights=True)
    hist = model.fit(model_input, data['Target'].values, batch_size=batch_size, epochs=1000, verbose=1, validation_split=0.2, class_weight=class_weight, callbacks=[early_stopping])

    return model, hist

def evaluate(model, data, test_input, batch_size):
    pred_ans = model.predict(test_input, batch_size=batch_size)

    y_true = data['Target'].values
    y_pred = np.where(pred_ans > 0.5, 1, 0)
    
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, pred_ans)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    log_loss_value = log_loss(y_true, pred_ans)

    return accuracy, roc_auc, precision, recall, log_loss_value, y_true, pred_ans


def plot_history(hist):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['auc'])
    plt.plot(hist.history['val_auc'])
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['binary_crossentropy'])
    plt.plot(hist.history['val_binary_crossentropy'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()


item_path = 'data/item.csv'
review_path = 'data/review.csv'
train_data, test_data, sparse_features, dense_features = load_data(item_path, review_path)

embedding_dims = [4, 8]
batch_sizes = [32, 64, 128, 256]
dnn_hidden_units_list = [[128, 64], [256, 128], [512, 256]]

best_accuracy = 0
best_roc_auc = 0
best_log_loss = float('inf')
best_params = {'embedding_dim': None, 'batch_size': None, 'dnn_hidden_units': None}
best_model = None
best_hist = None

for embedding_dim in embedding_dims:
    for batch_size in batch_sizes:
        for dnn_hidden_units in dnn_hidden_units_list:
            print(f"Training with embedding_dim={embedding_dim}, batch_size={batch_size}, dnn_hidden_units={dnn_hidden_units}")
            train_input, test_input, train_linear_feature_columns, train_dnn_feature_columns = prepare_data(train_data, test_data, sparse_features, dense_features, embedding_dim)
            model, hist = train(train_data, train_input, train_linear_feature_columns, train_dnn_feature_columns, batch_size, dnn_hidden_units)
            accuracy, roc_auc, precision, recall, log_loss_value, y_true, pred_ans = evaluate(model, test_data, test_input, batch_size)
            
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
                best_hist = hist
                best_y_true = y_true
                best_pred_ans = pred_ans

print(f"Best parameters: embedding_dim={best_params['embedding_dim']}, batch_size={best_params['batch_size']}, dnn_hidden_units={best_params['dnn_hidden_units']}")
accuracy, roc_auc, precision, recall, log_loss_value, y_true, pred_ans = evaluate(best_model, test_data, test_input, best_params['batch_size'])
print(f"Final Evaluation - Accuracy: {accuracy}, ROC-AUC: {roc_auc}, Log Loss: {log_loss_value}")
plot_history(best_hist)
