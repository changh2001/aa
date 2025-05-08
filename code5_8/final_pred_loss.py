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
import math



# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 氨基酸映射
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i+2 for i, aa in enumerate(AA_LIST)}
PAD_IDX = 0


# class EpitopeDataset(Dataset):
#     def __init__(self, csv_path):
#         df = pd.read_csv(csv_path)
#         self.seqs = df['sequence'].astype(str).tolist()
#         self.labels = df['label'].astype(float).tolist()

#     def __len__(self):
#         return len(self.seqs)

#     def __getitem__(self, idx):
#         seq = self.seqs[idx]
#         seq_idx = [AA_TO_IDX.get(aa, 1) for aa in seq]
#         return seq_idx, self.labels[idx]


# def collate_fn(batch):
#     # 过滤掉长度为0的序列，避免 pack_padded_sequence 错误
#     filtered = [(seq, label) for seq, label in batch if len(seq) > 0]
#     if not filtered:
#         raise ValueError("All sequences in a batch are empty.")
#     seqs, labels = zip(*filtered)
#     lengths = [len(s) for s in seqs]
#     max_len = max(lengths)
#     padded = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
#     for i, s in enumerate(seqs):
#         padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
#     return padded, torch.tensor(labels, dtype=torch.float), torch.tensor(lengths, dtype=torch.long)



# 修改 EpitopeDataset 类
class EpitopeDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # 添加过滤空序列的逻辑 (建议保留，避免 pack_padded_sequence 错误)
        df['sequence'] = df['sequence'].astype(str)
        df = df[df['sequence'].str.len() > 0].reset_index(drop=True)

        self.seqs = df['sequence'].tolist()
        # 将标签读取为整数类型 <-- 修改这里
        self.labels = df['label'].astype(int).tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        # 注意 AA_TO_IDX.get(aa, 1) 中的 1 是未知氨基酸索引
        seq_idx = [AA_TO_IDX.get(aa, 1) for aa in seq]
        # 返回整数类型的标签 <-- 修改这里
        return seq_idx, self.labels[idx]

# 修改 collate_fn 函数
def collate_fn(batch):
    # 过滤掉长度为0的序列 (保留)
    filtered = [(seq, label) for seq, label in batch if len(seq) > 0]
    if not filtered:
        raise ValueError("All sequences in a batch are empty.")

    seqs, labels = zip(*filtered)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    padded = torch.full((len(seqs), max_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)

    # 将标签转换为 Long 类型张量 <-- 修改这里
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # 将长度转换为 Long 类型张量 <-- 修改这里 (之前是 float)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded, labels_tensor, lengths_tensor






class Attention(nn.Module):
    def __init__(self, input_feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_feature_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, 2*h]
        weights = torch.softmax(self.attention(x), dim=1)  # [batch, seq_len, 1]
        # 加权求和得到序列表示
        return (weights * x).sum(dim=1)  # [batch, 2*h]

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
        # x: [batch, C, L]
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class OptimizedNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, conv_channels=64, lstm_hidden=128,
                 attention_hidden=256, mlp_hidden=128):
        super(OptimizedNet, self).__init__()
        # 随机初始化嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        nn.init.uniform_(self.embedding.weight.data, -1.0, 1.0)
        # 第一层卷积残差块
        self.resblock = ResidualBlock(embed_dim, conv_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        # BiLSTM 层
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.attention = Attention(lstm_hidden *2)
        self.dropout2 = nn.Dropout(0.5)
        # MLP 分类器
        self.fc1 = nn.Linear(lstm_hidden * 2, mlp_hidden)
        self.bn = nn.BatchNorm1d(mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # x: [batch, L]
        emb = self.embedding(x)  # [batch, L, d]
        # 转为 (batch, d, L) 用于 Conv1d
        conv_in = emb.permute(0, 2, 1)
        conv_out = self.resblock(conv_in)
        conv_out = self.pool(conv_out)
        conv_out = self.dropout1(conv_out)
        # 转回 (batch, L', C)
        lstm_in = conv_out.permute(0, 2, 1)
        # 因为经过 MaxPool1d(kernel_size=2)，序列长度缩减为 ceil(original_length/2)


        # 计算池化后每条序列的新长度
        # 原长度 lengths: LongTensor [B]
        new_len = ((lengths + 1) // 2).long()
        # 确保 new_len 不超过 x2 的实际长度
        max_pooled_len = lstm_in.size(1)
        new_len = new_len.clamp(max=max_pooled_len)

        # Pack for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            lstm_in,
            new_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        attn = self.attention(out)  # [batch, 2*h]
        x = self.dropout2(attn)
        x = self.relu(self.bn(self.fc1(x)))
        logits = self.fc2(x)
        # x = self.sigmoid(self.fc2(x).squeeze(1))
        return logits
    





# --- Loss Function ---
# (Focal Logit-Adjusted Softmax Loss - 公式 4 & 9)
# (Focal Logit-Adjusted Softmax Loss - Eq 4 & 9)
class FocalLogitAdjustedLoss(nn.Module):
    def __init__(self, class_priors, gamma=1.0, tau=1.0):
        """
        Args:
            class_priors (list or np.array): 各类的先验概率 (Prior probability for each class)
            gamma (float): Focal loss focusing parameter.
            tau (float): Logit adjustment scaling parameter.
        """
        super().__init__()
        if not isinstance(class_priors, torch.Tensor):
            class_priors = torch.tensor(class_priors, dtype=torch.float)
        if not torch.is_floating_point(class_priors):
             class_priors = class_priors.float()

        self.class_priors = class_priors.to(device)
        self.gamma = gamma
        self.tau = tau
        self.num_classes = len(class_priors)

        # 预计算 log 先验 (Precompute log priors)
        self.log_priors = torch.log(self.class_priors + 1e-9) # 避免 log(0) (Avoid log(0))

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits (batch_size, num_classes)
            targets (torch.Tensor): True labels (batch_size,)
        Returns:
            torch.Tensor: Calculated loss (scalar)
        """
        # 1. Logit Adjustment (公式 4 分子分母中的指数项)
        # (Logit Adjustment - exponent term in numerator/denominator of Eq 4)
        adjusted_logits = logits + self.tau * self.log_priors.unsqueeze(0) # (b, C) + (1, C)

        # 2. 计算调整后的 Softmax 概率 (Calculate adjusted Softmax probabilities)
        # (p_y in Eq 9)
        probs = torch.softmax(adjusted_logits, dim=1) # (b, C)

        # 3. 获取目标类的概率 (Get probabilities for the target class)
        # targets.unsqueeze(1) 形状变为 (b, 1) (targets.unsqueeze(1) shape becomes (b, 1))
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1) # (b,)

        # 4. 计算 Focal Loss (Calculate Focal Loss - Eq 9)
        focal_term = (1 - target_probs) ** self.gamma
        cross_entropy_term = -torch.log(target_probs + 1e-9) # 避免 log(0) (Avoid log(0))

        loss = focal_term * cross_entropy_term

        return loss.mean() # 返回批次的平均损失 (Return mean loss for the batch)











# # 训练函数
# def train_model(model, loader, criterion, optimizer, device):
#     """
#     训练一个 epoch，并显示进度条
#     """
#     model.train()
#     total_loss = 0
#     # 使用 tqdm 包装 DataLoader 以显示进度
#     for x, y, lengths in tqdm(loader, desc="Training", unit="batch"):
#         x, y, lengths = x.to(device), y.to(device), lengths.to(device)
#         optimizer.zero_grad()
#         preds = model(x, lengths)
#         loss = criterion(preds, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * x.size(0)
#     return total_loss / len(loader.dataset)




# --- Training and Evaluation Functions ---
def train_model(model, dataloader, criterion, optimizer, scheduler, device, args,epoch):
    """训练一个轮次 (Train for one epoch)"""
    model.train() # 设置为训练模式 (Set model to training mode)
    total_loss = 0.0
    # progress_bar = tqdm(dataloader, desc="Training", leave=False, unit='batch')
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}", leave=False,unit="batch")

    for batch in train_bar:
        sequences, labels, lengths = batch
        sequences, labels = sequences.to(device), labels.to(device)
        lengths = lengths.to('cpu') # pack_padded_sequence 需要 CPU 上的长度 (needs lengths on CPU)

        # 前向传播 (Forward pass)
        optimizer.zero_grad() # 清零梯度 (Zero gradients)
        logits = model(sequences, lengths)

        # 计算损失 (Calculate loss)
        loss = criterion(logits, labels)

        # 反向传播和优化 (Backward pass and optimization)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        optimizer.step()
        scheduler.step() # 更新学习率 (Update learning rate)

        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    return total_loss / len(dataloader)



# # 评估函数
# def evaluate_model(model, loader, device,args):
#     """
#     在评估前可选地加载模型参数，然后在测试集上评估性能。
#     model: PyTorch 模型实例
#     loader: DataLoader
#     device: 运行设备
#     ckpt_path: 如果提供，则从该路径加载模型参数
#     返回字典包含各项指标和预测结果
#     """
#     # 可选地加载模型权重
#     if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
#         model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
#         print(f"Loaded model weights from {args.ckpt_path}")
#     model.to(device)

#     model.eval()
#     trues, preds, probs = [], [], []
#     with torch.no_grad():
#         for x, y, lengths in loader:
#             x, y, lengths = x.to(device), y.to(device), lengths.to(device)
#             prob = model(x, lengths)
#             pred = (prob > 0.5).float()
#             trues.extend(y.cpu().numpy().tolist())
#             preds.extend(pred.cpu().numpy().tolist())
#             probs.extend(prob.cpu().numpy().tolist())
#     trues = np.array(trues)
#     preds = np.array(preds)
#     probs = np.array(probs)
#         # Calculate metrics
#     acc = accuracy_score(trues, preds)
#     prec = precision_score(trues, preds, zero_division=0)
#     rec = recall_score(trues, preds, zero_division=0)
#     f1 = f1_score(trues, preds, zero_division=0)
#     mcc = matthews_corrcoef(trues, preds) if len(np.unique(trues)) > 1 else 0.0
#     bacc = balanced_accuracy_score(trues, preds)
#     cm = confusion_matrix(trues, preds)

#     try:
#         auc_roc = roc_auc_score(trues, probs)
#     except:
#         auc_roc = float('nan')
#     try:
#         auc_pr = average_precision_score(trues, probs)
#     except:
#         auc_pr = float('nan')
#     return {
#         'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
#         'mcc': mcc, 'bacc': bacc, 'auc_roc': auc_roc, 'auc_pr': auc_pr,
#         'preds': preds, 'probs': probs, 'trues': trues,'confusion_matrix':cm
#     }


def evaluate_model(model, dataloader, criterion, device,args):
    """评估模型 (Evaluate the model)"""
    model.eval() # 设置为评估模式 (Set model to evaluation mode)
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False, unit='batch')
    with torch.no_grad(): # 关闭梯度计算 (Disable gradient calculation)
        for batch in progress_bar:
            sequences, labels, lengths = batch
            sequences, labels = sequences.to(device), labels.to(device)
            lengths = lengths.to('cpu')

            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            scores = torch.softmax(logits, dim=1)[1] # 获取正类（表位）的概率 (Get probability for positive class (epitope))
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    trues = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_scores)
        # Calculate metrics
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    mcc = matthews_corrcoef(trues, preds) if len(np.unique(trues)) > 1 else 0.0
    bacc = balanced_accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)

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
        'preds': preds, 'probs': probs, 'trues': trues,'confusion_matrix':cm,'avg_loss' : avg_loss
    }




    # accuracy = accuracy_score(all_labels, all_preds)
    # balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    # auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    # precision = precision_score(all_labels, all_preds, zero_division=0)
    # recall = recall_score(all_labels, all_preds, zero_division=0)
    # f1 = f1_score(all_labels, all_preds, zero_division=0)

    # metrics = {
    #     'loss': avg_loss,
    #     'accuracy': accuracy,
    #     'balanced_accuracy': balanced_acc,
    #     'auc': auc,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1
    # }
    # return metrics, all_labels, all_preds, all_scores # 返回指标和预测结果 (Return metrics and predictions)






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


# 模型超参数 (Model Hyperparameters)
ENCODING_DIM = 21 # BLOSUM62 Eigen + UNK
LSTM_HIDDEN_DIM = 64 # LSTM 隐藏层维度 (LSTM hidden dimension)
LSTM_LAYERS = 1 # BiLSTM 层数 (Number of BiLSTM layers) - Fig 1 shows one BiLSTM layer block
TRANSFORMER_DIM = 128 # Transformer 维度 (Transformer dimension)
TRANSFORMER_HEADS = 8 # Transformer 多头注意力头数 (Number of Transformer heads)
TRANSFORMER_FF_DIM = 512 # Transformer 前馈网络维度 (Transformer feedforward dimension)
TRANSFORMER_LAYERS = 2 # Transformer Block 数量 (Number of Transformer blocks)
DROPOUT_RATE = 0.1 # Dropout 比率 (Dropout rate)
NUM_CLASSES = 2 # 二分类: 表位 vs 非表位 (Binary classification: Epitope vs Non-epitope)



# 训练超参数 (Training Hyperparameters)
BATCH_SIZE = 128 # 批次大小 (Batch size)
LEARNING_RATE = 1e-4 # 学习率 (Learning rate)
WEIGHT_DECAY = 0.01 # 权重衰减 (Weight decay)
EPOCHS = 10 # 训练轮数 (Number of epochs) - 可能需要更多轮次 (Might need more)
VALIDATION_SPLIT = 0.1 # 从训练集中划分出的验证集比例 (Validation set split ratio from training data)
WARMUP_STEPS = 200 # 学习率预热步数 (Learning rate warmup steps)
LOSS_GAMMA = 1.0 # Focal Loss gamma 参数 (Focal Loss gamma parameter)
LOSS_TAU = 1.0 # Logit Adjustment tau 参数 (Logit Adjustment tau parameter)
SEED = 42 # 随机种子 (Random seed for reproducibility)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CALIBER BiLSTM+GCN model")
    parser.add_argument('--train_path', type=str, default='training_dataset.csv')
    parser.add_argument('--test_path', type=str, default='testing_dataset.csv')
    parser.add_argument('--learning_rates', type=float, default=1e-3)
    parser.add_argument('--hidden_dims', type=int, default=128)
    parser.add_argument('--batch_sizes', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--path', type=str, default="./checkpoints_final_loss")
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints_final_loss/model.pth")
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--loss_gamma', type=float, default=1.0)
    parser.add_argument('--loss_tau', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=200)


    args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_sizes, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_sizes, shuffle=False, collate_fn=collate_fn)


        # 2. 计算类别先验概率 (Calculate Class Priors for Loss Function)
    print("Calculating class priors...")
    train_labels = [label for _, label in train_ds] # 获取训练集所有标签 (Get all labels from train subset)
    class_counts = np.bincount(train_labels, minlength=args.num_class)
    class_priors = class_counts / len(train_labels)
    print(f"Class counts (Train): {class_counts}")
    print(f"Class priors (Train): {class_priors}")


    
    # 用最优参数训练全量数据
    vocab_size = len(AA_LIST)+2
    model = OptimizedNet(
        vocab_size=len(AA_LIST) + 2,
        embed_dim=10,
        conv_channels=64,
        lstm_hidden=128,
        attention_hidden=128,
        mlp_hidden=128
    ).to(device)
    # opt = optim.Adam(model.parameters(), lr=args.learning_rates)
    # crit = nn.BCELoss()
    # loader = DataLoader(train_ds, batch_size=args.batch_sizes, shuffle=True, collate_fn=collate_fn)

    criterion = FocalLogitAdjustedLoss(class_priors=class_priors, gamma=args.loss_gamma, tau=args.loss_tau).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rates, weight_decay=WEIGHT_DECAY, amsgrad=True)

    # 4. 初始化学习率调度器 (Initialize Learning Rate Scheduler)
    num_training_steps = len(train_loader) * args.epoch
    num_warmup_steps = args.warmup_steps

    def lr_lambda(current_step):
        # Warmup 阶段 (Warmup phase)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine Decay 阶段 (Cosine Decay phase)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.ckpt_path is None or not os.path.exists(args.ckpt_path):

        model.to(device)
        model.train()

        # for epoch in range(args.epoch):
        #     print(f"Epoch {epoch+1}")
        #     loss = train_model(model, loader, crit, opt, device)
        #     train_losses.append(loss)
        #     print(f"Epoch {epoch+1}, Loss={loss:.4f}")


        train_losses = [] 

        # 训练循环，带进度条
        for epoch in range(args.epoch):
            loss = train_model(model, train_loader, criterion, optimizer, scheduler, device,args,epoch)
            train_losses.append(loss)
            print(f"Epoch {epoch+1}, Loss={loss:.4f}")
            # total_loss = 0.0
            # # tqdm 包装 dataloader，显示批次进度
            # train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}", unit="batch")
            # for seqs, labels, lengths in train_bar:
            #     seqs, labels, lengths = seqs.to(device),  labels.to(device) ,lengths.to(device)
            #     optimizer.zero_grad()
            #     logits = model(seqs, lengths)
            #     loss = criterion(logits, labels)
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            #     optimizer.step()
            #     scheduler.step()

            #     batch_loss = loss.item() * seqs.size(0)
            #     # 更新进度条的后缀信息
            #     # train_bar.set_postfix(loss=batch_loss / seqs.size(0))
            #     train_bar.set_postfix(loss=batch_loss, lr=scheduler.get_last_lr()[0])

            train_losses.append(loss)
            print(f"Epoch {epoch+1}/{args.epoch} 完成，平均损失: {loss:.4f}")
  
        # 保存模型
        model_path = os.path.join(args.path, 'model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved model weights to {model_path}")




    # 评估测试集
    
    metrics = evaluate_model(model, test_loader, criterion, device,args)
    # print(metrics)
    # print("\nTest Set Metrics:")
    # for k, v in metrics.items():
    #     if isinstance(v, float):
    #         print(f"  {k}: {v:.4f}")

    print(metrics)
    # 保存预测结果
    pd.DataFrame({'Predicted':metrics['preds'], 'True':metrics['trues']}) \
        .to_csv(os.path.join(args.path, 'predictions.csv'), index=False)

    # Save predictions and true labels to CSV
    preds_df = pd.DataFrame({'Predicted': metrics['preds'], 'True': metrics['trues'],'probs':metrics['probs']})
    preds_df.to_csv(os.path.join(args.path, 'predictions.csv'), index=False)
    print("Saved predictions and true labels to ./checkpoint_CALIBER/predictions.csv")



    # Plot and save training loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.path, 'loss.png'))
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
    plt.savefig(os.path.join(args.path, 'roc_curve.png'))
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
    plt.savefig(os.path.join(args.path, 'pr_curve.png'))
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
    plt.savefig(os.path.join(args.path, 'confusion_matrix.png'))
    plt.close()
    print("Saved confusion matrix to ./checkpoint_CALIBER/confusion_matrix.png")


    


    # # 画图并保存
    # plt.figure(); plt.plot(losses); plt.title('Loss'); plt.savefig('checkpoint_CALIBER/loss.png')
    # fpr, tpr, _ = roc_curve(metrics['trues'], metrics['probs']); plt.figure(); plt.plot(fpr,tpr); plt.savefig('checkpoint_CALIBER/roc_curve.png')
    # prec, rec, _ = precision_recall_curve(metrics['trues'], metrics['probs']); plt.figure(); plt.plot(rec,prec); plt.savefig('checkpoint_CALIBER/pr_curve.png')
    # cm = confusion_matrix(metrics['trues'], metrics['preds']); plt.figure(); plt.imshow(cm); plt.savefig('checkpoint_CALIBER/confusion_matrix.png')
