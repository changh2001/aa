import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast


class AminoAcidDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def read_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                label = int(line.strip().split('_')[-1])
                labels.append(label)
            else:
                data.append(line.strip())
    return data, labels


def one_hot_encode(sequences, max_length):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
    encoded_sequences = []
    for seq in sequences:
        seq = seq[:max_length]
        encoded_seq = np.zeros((max_length, 20))
        for i, aa in enumerate(seq):
            if aa in aa_dict:
                encoded_seq[i, aa_dict[aa]] = 1
        if len(seq) < max_length:
            encoded_seq[len(seq):] = 0
        encoded_sequences.append(encoded_seq)
    return np.array(encoded_sequences)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        return (attention_weights * x).sum(dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x


class OptimizedNet(nn.Module):
    def __init__(self, input_length):
        super(OptimizedNet, self).__init__()
        self.conv1 = ResidualBlock(20, 64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.attention = Attention(128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = self.attention(lstm_out)
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze()


def train_model(model, train_loader, criterion, optimizer, scheduler, scaler, device, epochs):
    model.train()
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_pred_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_probs)
    auc_pr = auc(recall_curve, precision_curve)

    return accuracy, precision, recall, f1, auc_roc, auc_pr


def cross_validate(data, labels, max_length, n_splits=5, epochs=15, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f'\nFold {fold + 1}/{n_splits}')

        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_dataset = AminoAcidDataset(train_data, train_labels)
        val_dataset = AminoAcidDataset(val_data, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OptimizedNet(max_length).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler()

        train_model(model, train_loader, criterion, optimizer, scheduler, scaler, device, epochs)

        model.load_state_dict(torch.load('best_model.pth'))
        accuracy, precision, recall, f1, auc_roc, auc_pr = evaluate_model(model, val_loader, device)
        results.append([accuracy, precision, recall, f1, auc_roc, auc_pr])

        print(f'Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}')

    avg_results = np.mean(results, axis=0)
    print('\nAverage metrics across all folds:')
    print(f'Accuracy: {avg_results[0]:.4f}, Precision: {avg_results[1]:.4f}, '
          f'Recall: {avg_results[2]:.4f}, F1: {avg_results[3]:.4f}, AUC-ROC: {avg_results[4]:.4f}, AUC-PR: {avg_results[5]:.4f}')


if __name__ == "__main__":
    max_length = 23#数据最长的序列为23
    batch_size = 64
    epochs = 20

    train_file_path = 'training_dataset.txt'
    test_file_path = 'testing_dataset.txt'

    train_data, train_labels = read_data(train_file_path)
    test_data, test_labels = read_data(test_file_path)

    encoded_train_data = one_hot_encode(train_data, max_length)
    encoded_test_data = one_hot_encode(test_data, max_length)

    print("Starting cross-validation...")
    cross_validate(encoded_train_data, np.array(train_labels), max_length, n_splits=5, epochs=epochs,
                   batch_size=batch_size)

    print("\nTraining final model on full training set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = AminoAcidDataset(encoded_train_data, train_labels)
    test_dataset = AminoAcidDataset(encoded_test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = OptimizedNet(max_length).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    train_model(model, train_loader, criterion, optimizer, scheduler, scaler, device, epochs)

    model.load_state_dict(torch.load('best_model.pth'))
    accuracy, precision, recall, f1, auc_roc, auc_pr = evaluate_model(model, test_loader, device)

    print("\nFinal Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
