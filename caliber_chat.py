import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score, average_precision_score, \
    confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 氨基酸映射
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AA_LIST)}
PAD_IDX = 0

class EpitopeDataset(Dataset):
    def __init__(self, csv_path):
        """
        csv_path: path to CSV file with columns 'sequence' and 'label'
        """
        df = pd.read_csv(csv_path)
        self.seqs = df['sequence'].astype(str).tolist()
        self.labels = df['label'].tolist()
        self.labels = [float(l) for l in self.labels]


    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        seq_idx = [AA_TO_IDX.get(aa, len(AA_LIST)+1) for aa in seq]
        label = float(self.labels[idx])
        return seq_idx, label

def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    padded = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    lengths = torch.tensor(lengths, dtype=torch.float)
    return padded, labels, lengths

    
class BiLSTMModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, mlp_hidden_dim, vocab_size):
        super(BiLSTMModel, self).__init__()
        # 随机嵌入层 (Random Embedding)
        # 嵌入矩阵 E ∈ R^{vocab_size × embed_dim}, embed_dim 默认为10
        # 初始化方式: 从均匀分布 U(-1,1) 中随机采样
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        nn.init.uniform_(self.embedding.weight.data, -1.0, 1.0)
        # BiLSTM 层, 隐藏单元维度 hidden_dim，所以双向输出维度为 2*hidden_dim
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Dropout 应用于 BiLSTM 输出, 以增强正则化 (rate=0.25)
        self.dropout = nn.Dropout(p=0.25)
        # MLP 分类器
        self.fc1 = nn.Linear(hidden_dim * 2, mlp_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu().int(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # 应用 Dropout
        out = self.dropout(out)
        mask = (x != PAD_IDX).float().unsqueeze(2)
        summed = (out * mask).sum(dim=1)
        avg = summed / lengths.unsqueeze(1)
        h = self.relu(self.fc1(avg))
        y = self.sigmoid(self.fc2(h).squeeze(1))
        return y

# 训练函数
# 训练函数
def train_model(model, loader, criterion, optimizer, device):
    """
    训练一个 epoch，并显示进度条
    """
    model.train()
    total_loss = 0
    # 使用 tqdm 包装 DataLoader 以显示进度
    for x, y, lengths in tqdm(loader, desc="Training", unit="batch"):
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        preds = model(x, lengths)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# 评估函数
def evaluate_model(model, loader, device,args):
    """
    在评估前可选地加载模型参数，然后在测试集上评估性能。
    model: PyTorch 模型实例
    loader: DataLoader
    device: 运行设备
    ckpt_path: 如果提供，则从该路径加载模型参数
    返回字典包含各项指标和预测结果
    """
    # 可选地加载模型权重
    if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
        print(f"Loaded model weights from {args.ckpt_path}")
    model.to(device)

    model.eval()
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            prob = model(x, lengths)
            pred = (prob > 0.5).float()
            trues.extend(y.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            probs.extend(prob.cpu().numpy().tolist())
    trues = np.array(trues)
    preds = np.array(preds)
    probs = np.array(probs)
        # Calculate metrics
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    mcc = matthews_corrcoef(trues, preds) if len(np.unique(trues)) > 1 else 0.0
    bacc = balanced_accuracy_score(trues, preds)
    try:
        auc_roc = roc_auc_score(trues, probs)
    except:
        auc_roc = float('nan')
    try:
        auc_pr = average_precision_score(trues, probs)
    except:
        auc_pr = float('nan')
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'mcc': mcc, 'bacc': bacc, 'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'preds': preds, 'probs': probs, 'trues': trues
    }

    # return {
    #     'ACC': accuracy_score(trues, preds),
    #     'Pre': precision_score(trues, preds, zero_division=0),
    #     'Recall': recall_score(trues, preds, zero_division=0),
    #     'F1': f1_score(trues, preds, zero_division=0),
    #     'MCC': matthews_corrcoef(trues, preds) if len(np.unique(trues))>1 else 0,
    #     'BACC': balanced_accuracy_score(trues, preds),
    #     'AUC-ROC': roc_auc_score(trues, probs) if len(np.unique(trues))>1 else np.nan,
    #     'AUC-PR': average_precision_score(trues, probs) if len(np.unique(trues))>1 else np.nan,
    #     'trues': trues, 'preds': preds, 'probs': probs
    # }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CALIBER BiLSTM+GCN model")
    parser.add_argument('--train_path', type=str, default='training_dataset.csv')
    parser.add_argument('--test_path', type=str, default='testing_dataset.csv')
    parser.add_argument('--learning_rates', type=float, default=1e-3)
    parser.add_argument('--hidden_dims', type=int, default=128)
    parser.add_argument('--batch_sizes', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints_caliber/model.pth")

    args = parser.parse_args()

    os.makedirs('checkpoint_CALIBER', exist_ok=True)

    # 检查是否有可用的GPU (MPS for Apple Silicon, CUDA for Nvidia)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    # 加载数据
    train_ds = EpitopeDataset(args.train_path)
    test_ds = EpitopeDataset(args.test_path)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)


    # best_params = {'lr':1e-3, 'hidden':128, 'bs':32}
    # print("params:", best_params)

    # 用最优参数训练全量数据
    vocab_size = len(AA_LIST)+2
    model = BiLSTMModel(30, args.hidden_dims, args.hidden_dims, vocab_size).to(device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rates)
    crit = nn.BCELoss()
    loader = DataLoader(train_ds, batch_size=args.batch_sizes, shuffle=True, collate_fn=collate_fn)

    train_losses = []
    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}")
        loss = train_model(model, loader, crit, opt, device)
        train_losses.append(loss)
        print(f"Epoch {epoch+1}, Loss={loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'checkpoint_CALIBER/model.pth')
    model_path = os.path.join('checkpoint_CALIBER', 'model.pth')
    print(f"\nSaved model weights to {model_path}")


    # 评估测试集
    metrics = evaluate_model(model, test_loader, device, args)
    # print(metrics)
    print("\nTest Set Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # 保存预测结果
    pd.DataFrame({'Predicted':metrics['preds'], 'True':metrics['trues']}) \
        .to_csv('checkpoint_CALIBER/predictions.csv', index=False)

    # Save predictions and true labels to CSV
    preds_df = pd.DataFrame({'Predicted': metrics['preds'], 'True': metrics['trues']})
    preds_df.to_csv(os.path.join('checkpoint_CALIBER', 'predictions.csv'), index=False)
    print("Saved predictions and true labels to ./checkpoint_CALIBER/predictions.csv")



    # Plot and save training loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join('checkpoint_CALIBER', 'loss.png'))
    plt.close()
    print("Saved training loss plot to ./checkpoint_CALIBER/loss.png")
    
    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(metrics['trues'], metrics['probs'])
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']:.4f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join('checkpoint_CALIBER', 'roc_curve.png'))
    plt.close()
    print("Saved ROC curve to ./checkpoint_CALIBER/roc_curve.png")
    
    # Plot and save Precision-Recall curve
    precision, recall, _ = precision_recall_curve(metrics['trues'], metrics['probs'])
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {metrics['auc_pr']:.4f}")
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join('checkpoint_CALIBER', 'pr_curve.png'))
    plt.close()
    print("Saved Precision-Recall curve to ./checkpoint_CALIBER/pr_curve.png")
    
    # Plot and save confusion matrix
    cm = confusion_matrix(metrics['trues'], metrics['preds'])
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Neg', 'Pos'])
    plt.yticks(tick_marks, ['Neg', 'Pos'])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join('checkpoint_CALIBER', 'confusion_matrix.png'))
    plt.close()
    print("Saved confusion matrix to ./checkpoint_CALIBER/confusion_matrix.png")


    


    # # 画图并保存
    # plt.figure(); plt.plot(losses); plt.title('Loss'); plt.savefig('checkpoint_CALIBER/loss.png')
    # fpr, tpr, _ = roc_curve(metrics['trues'], metrics['probs']); plt.figure(); plt.plot(fpr,tpr); plt.savefig('checkpoint_CALIBER/roc_curve.png')
    # prec, rec, _ = precision_recall_curve(metrics['trues'], metrics['probs']); plt.figure(); plt.plot(rec,prec); plt.savefig('checkpoint_CALIBER/pr_curve.png')
    # cm = confusion_matrix(metrics['trues'], metrics['preds']); plt.figure(); plt.imshow(cm); plt.savefig('checkpoint_CALIBER/confusion_matrix.png')
