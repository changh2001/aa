import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score, average_precision_score, \
    confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split # For splitting train into train/val
import math
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm # For progress bars
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- Configuration and Hyperparameters ---
# 数据路径 (Data Paths) - !! 请根据需要修改 (Modify if needed) !!
TRAIN_CSV_PATH = 'training_dataset.csv'
TEST_CSV_PATH = 'testing_dataset.csv'

save_dir = "./BeeTLE_Gem_checkpoints1_0.5"
os.makedirs(save_dir, exist_ok=True)

MODEL_SAVE_PATH = f"{save_dir}/best_beetle_model.pth" # 保存最佳模型的路径 (Path to save the best model)

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
EPOCHS = 5 # 训练轮数 (Number of epochs) - 可能需要更多轮次 (Might need more)
VALIDATION_SPLIT = 0.1 # 从训练集中划分出的验证集比例 (Validation set split ratio from training data)
WARMUP_STEPS = 200 # 学习率预热步数 (Learning rate warmup steps)
LOSS_GAMMA = 0.5 # Focal Loss gamma 参数 (Focal Loss gamma parameter)
LOSS_TAU = 1 # Logit Adjustment tau 参数 (Logit Adjustment tau parameter)
SEED = 42 # 随机种子 (Random seed for reproducibility)

# 设备设置 (Device Configuration)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


print(f"Using device: {DEVICE}")

# 设置随机种子 (Set random seed)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Amino Acid Encoding ---
# (来自仓库 yuanx749/bcell/src/encoder.py 的简化和整合)
# (Simplified and integrated from yuanx749/bcell/src/encoder.py)

# 标准氨基酸 (Standard Amino Acids)
AAS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AAS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AAS)}
VOCAB_SIZE = len(AAS) # 20
UNK_IDX = VOCAB_SIZE # 20
PAD_IDX = VOCAB_SIZE + 1 # 21

# BLOSUM62 矩阵 (来自 Biopython Bio.SubsMat.MatrixInfo.blosum62)
# (From Biopython Bio.SubsMat.MatrixInfo.blosum62)
# 为了简洁，直接使用预计算的特征分解结果 (For brevity, using precomputed eigen decomposition results)
# 在实际应用中，您应该加载或计算 BLOSUM62 并执行特征分解
# (In practice, you should load/compute BLOSUM62 and perform eigen decomposition)
# 下面的 U 和 S 是示例值，您需要替换为实际计算结果
# (The U and S below are placeholder values, replace with actual computed results)
# 例如，可以使用 numpy.linalg.eigh(blosum62_exp_matrix)
# (e.g., using numpy.linalg.eigh(blosum62_exp_matrix))

# 假设这是 BLOSUM62 指数化后的特征分解 U * sqrt(Sigma) (Equation 1)
# (Assuming this is the U * sqrt(Sigma) from eigen decomposition of exponentiated BLOSUM62)
# !! 注意: 这里的 blosum_embed 是一个占位符，需要替换为实际计算的 BLOSUM62 特征向量 !!
# !! NOTE: blosum_embed here is a PLACEHOLDER, replace with actual computed BLOSUM62 eigenvectors !!
# 维度应为 (20, 20) (Dimension should be (20, 20))
# blosum_embed = torch.randn(VOCAB_SIZE, VOCAB_SIZE) # Placeholder

# 定义20种标准氨基酸的顺序（A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y）
AA_LIST = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
aa_to_idx = {aa:i for i, aa in enumerate(AA_LIST)}
aa_to_idx['X'] = 20  # 未知氨基酸
aa_to_idx['PAD'] = 21  # 填充符号

# 构建BLOSUM62矩阵（20x20）
from Bio.Align import substitution_matrices
blosum = substitution_matrices.load("BLOSUM62")
blosum_embed = np.zeros((20,20), dtype=float)
for i, aa1 in enumerate(AA_LIST):
    for j, aa2 in enumerate(AA_LIST):
        if (aa1, aa2) in blosum:
            blosum_embed[i,j] = blosum[(aa1, aa2)]
        else:
            blosum_embed[i,j] = blosum[(aa2, aa1)]


# 取指数
P = np.exp(blosum_embed)
# 特征分解 P = U * diag(vals) * U^T
eigvals, eigvecs = np.linalg.eigh(P)
# 取sqrt(矩阵)：E = U * sqrt(diag(eigvals))
sqrt_vals = np.sqrt(np.maximum(eigvals, 0))
E = eigvecs @ np.diag(sqrt_vals)  # 20x20


# 创建最终的编码矩阵 (Create the final encoding matrix) (22, 21)
# 20 AA + 1 UNK + 1 PAD x 20 BLOSUM_dim + 1 UNK_dim
embedding_matrix = np.zeros((VOCAB_SIZE + 2, ENCODING_DIM), dtype=float)
embedding_matrix[:VOCAB_SIZE, :VOCAB_SIZE] = E # BLOSUM 部分 (BLOSUM part)
embedding_matrix[UNK_IDX, VOCAB_SIZE] = 1.0 # UNK one-hot 部分 (UNK one-hot part)
# PAD 保持为零向量 (PAD remains zero vector)

embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)


class AminoAcidEncoder(nn.Module):
    """氨基酸编码器 (Amino Acid Encoder)"""
    def __init__(self, embedding_matrix_data): # 重命名输入参数 (Rename input parameter)
        super().__init__()
        # --- 错误修复: 确保输入是 Tensor ---
        # --- Error Fix: Ensure input is a Tensor ---
        if not isinstance(embedding_matrix_data, torch.Tensor):
            print("Warning: embedding_matrix_data was not a Tensor. Converting from numpy array.")
            # 从 NumPy 数组或其他类型转换 (Convert from numpy array or other types)
            embedding_matrix_tensor = torch.tensor(embedding_matrix_data, dtype=torch.float)
        else:
            # 确保数据类型正确 (Ensure correct data type)
            embedding_matrix_tensor = embedding_matrix_data.float()

        # 使用转换后的 Tensor 创建 Parameter (Create Parameter using the converted Tensor)
        self.embedding_matrix = nn.Parameter(embedding_matrix_tensor, requires_grad=False) # 不训练编码 (Do not train encoding)
        # --- 修复结束 ---

    def forward(self, indices):
        """
        Args:
            indices (torch.Tensor): Shape (batch_size, seq_len) - 氨基酸索引 (Amino acid indices)
        Returns:
            torch.Tensor: Shape (batch_size, seq_len, encoding_dim) - 编码后的向量 (Encoded vectors)
        """
        # 使用 Parameter (Use the Parameter)
        return self.embedding_matrix[indices]


# --- Dataset and DataLoader ---
class PeptideDataset(Dataset):
    """肽段数据集类 (Peptide Dataset Class)"""
    def __init__(self, csv_path, seq_col='sequence', label_col='label', sep='\t', max_len=None):
        """
        Args:
            csv_path (str): CSV 文件路径 (Path to the CSV file)
            seq_col (str): 序列列名 (Sequence column name)
            label_col (str): 标签列名 (Label column name)
            sep (str): CSV 分隔符 (CSV separator)
            max_len (int, optional): 最大序列长度，用于截断或填充 (Maximum sequence length for truncation/padding). Defaults to None.
        """
        self.max_len = max_len
        try:
            # self.data = pd.read_csv(csv_path, sep=sep)
            self.data = pd.read_csv(csv_path)
            # 清理数据 (Clean data)
            self.data = self.data[[seq_col, label_col]].dropna()
            self.data[seq_col] = self.data[seq_col].astype(str).str.strip().str.upper()
            # 过滤无效序列 (Filter invalid sequences)
            valid_chars = set(AAS)
            self.data = self.data[self.data[seq_col].apply(lambda s: all(c in valid_chars for c in s))]
            self.data[label_col] = pd.to_numeric(self.data[label_col])

            self.sequences = self.data[seq_col].tolist()
            self.labels = self.data[label_col].astype(int).tolist() # 确保标签是整数 (Ensure labels are integers)

        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            self.sequences, self.labels = [], []
        except KeyError as e:
             print(f"Error: Column {e} not found in {csv_path}. Check seq_col/label_col.")
             self.sequences, self.labels = [], []
        except Exception as e:
            print(f"Error loading dataset {csv_path}: {e}")
            self.sequences, self.labels = [], []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 序列转索引 (Sequence to indices)
        indices = [AA_TO_IDX.get(aa, UNK_IDX) for aa in seq]

        # 截断 (Truncate if necessary)
        if self.max_len and len(indices) > self.max_len:
            indices = indices[:self.max_len]

        seq_len = len(indices) # 实际长度 (Actual length)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long), seq_len

def collate_fn(batch):
    """
    处理批次数据，进行填充 (Processes a batch of data, performs padding).
    """
    sequences, labels, lengths = zip(*batch)

    # 获取最大长度 (Get max length in batch)
    max_len = max(lengths)

    # 填充序列 (Pad sequences)
    padded_sequences = torch.full((len(sequences), max_len), PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_sequences[i, :end] = seq[:end]

    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_sequences, labels, lengths

# --- Model Architecture ---
# (基于图 1 和论文描述，参考仓库代码结构)
# (Based on Fig 1 and paper description, referencing repo code structure)

class AttentionPool(nn.Module):
    """Attention Pooling Layer (公式 2, 3)"""
    def __init__(self, input_dim, scaled=True):
        super().__init__()
        self.input_dim = input_dim
        self.scaled = scaled
        # 可学习的查询向量 q (Learnable query vector q)
        self.query = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, lengths):
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, input_dim)
            lengths (torch.Tensor): Original lengths of sequences (batch_size,)
        Returns:
            torch.Tensor: Pooled output (batch_size, input_dim)
        """
        batch_size, seq_len, _ = x.size()

        # 计算注意力分数 (Calculate attention scores) (batch_size, seq_len)
        scores = torch.matmul(x, self.query) # (b, s, d) * (d) -> (b, s)
        if self.scaled:
            scores = scores / math.sqrt(self.input_dim)

        # 创建掩码以忽略填充部分 (Create mask to ignore padding)
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        scores.masked_fill_(~mask, -float('inf')) # 用负无穷填充填充位置 (Fill padding with -inf)

        # 计算注意力权重 (Calculate attention weights)
        alpha = torch.softmax(scores, dim=1) # (batch_size, seq_len)

        # 计算加权和 (Calculate weighted sum)
        pooled = torch.sum(x * alpha.unsqueeze(-1), dim=1) # (b, s, d) * (b, s, 1) -> sum over s -> (b, d)
        return pooled

class BeeTLeNet(nn.Module):
    """BeeTLe 模型 (BeeTLe Model)"""
    def __init__(self, encoding_dim, lstm_hidden_dim, lstm_layers,
                 transformer_dim, transformer_heads, transformer_ff_dim,
                 transformer_layers, num_classes, dropout_rate, embedding_matrix):
        super().__init__()
        self.encoder = AminoAcidEncoder(embedding_matrix)

        # BiLSTM 层 (BiLSTM Layer)
        self.bilstm = nn.LSTM(encoding_dim, lstm_hidden_dim, lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout_rate if lstm_layers > 1 else 0)

        # LSTM 输出维度 -> Transformer 输入维度 的线性层
        # (Linear layer from LSTM output dim -> Transformer input dim)
        lstm_output_dim = lstm_hidden_dim * 2
        self.lstm_to_transformer = nn.Linear(lstm_output_dim, transformer_dim)

        # Transformer Encoder 层 (Transformer Encoder Layers)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout_rate,
            activation='relu', # 论文中使用 ReLU (Paper uses ReLU)
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(transformer_dim) # 添加最终的 LayerNorm (Add final LayerNorm)
        )

        # Attention Pooling 层 (Attention Pooling Layer)
        self.attention_pool = AttentionPool(transformer_dim)

        # 分类器头 (Classifier Head) - 仅用于表位预测 (Only for epitope prediction)
        self.classifier_epitope = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_dim // 2, num_classes) # 输出 logits (Output logits)
        )

    def forward(self, indices, lengths):
        """
        Args:
            indices (torch.Tensor): Input indices (batch_size, seq_len)
            lengths (torch.Tensor): Original lengths (batch_size,)
        Returns:
            torch.Tensor: Epitope logits (batch_size, num_classes)
        """
        # 1. 编码 (Encoding)
        x = self.encoder(indices) # (b, s, enc_dim)

        # 2. BiLSTM
        # 打包序列以处理变长输入 (Pack sequence for variable length input)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.bilstm(packed_input)
        # 解包序列 (Unpack sequence)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True) # (b, s, lstm_out_dim)

        # 3. LSTM -> Transformer 维度转换 (LSTM -> Transformer dim projection)
        transformer_input = self.lstm_to_transformer(lstm_output) # (b, s, trans_dim)

        # 4. Transformer Encoder
        # 创建 Transformer 掩码 (Create Transformer mask) - True 表示忽略 (True means ignore)
        padding_mask = (indices == PAD_IDX) # (b, s)
        transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=padding_mask) # (b, s, trans_dim)

        # 5. Attention Pooling
        pooled_output = self.attention_pool(transformer_output, lengths) # (b, trans_dim)

        # 6. Classifier
        epitope_logits = self.classifier_epitope(pooled_output) # (b, num_classes)

        return epitope_logits

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

        self.class_priors = class_priors.to(DEVICE)
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

# --- Training and Evaluation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """训练一个轮次 (Train for one epoch)"""
    model.train() # 设置为训练模式 (Set model to training mode)
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        sequences, labels, lengths = batch
        sequences, labels = sequences.to(device), labels.to(device)
        lengths = lengths.to(device) # pack_padded_sequence 需要 CPU 上的长度 (needs lengths on CPU)

        # 前向传播 (Forward pass)
        optimizer.zero_grad() # 清零梯度 (Zero gradients)
        logits = model(sequences, lengths)

        # 计算损失 (Calculate loss)
        loss = criterion(logits, labels)

        # 反向传播和优化 (Backward pass and optimization)
        loss.backward()
        optimizer.step()
        scheduler.step() # 更新学习率 (Update learning rate)

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """评估模型 (Evaluate the model)"""
    model.eval() # 设置为评估模式 (Set model to evaluation mode)
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad(): # 关闭梯度计算 (Disable gradient calculation)
        for batch in progress_bar:
            sequences, labels, lengths = batch
            sequences, labels = sequences.to(device), labels.to(device)
            lengths = lengths.to(device)

            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            scores = torch.softmax(logits, dim=1)[:, 1] # 获取正类（表位）的概率 (Get probability for positive class (epitope))
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.to(device).numpy())
            all_preds.extend(preds.to(device).numpy())
            all_scores.extend(scores.to(device).numpy())

    
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

    # avg_loss = total_loss / len(dataloader)
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

# --- Main Execution ---
if __name__ == "__main__":
    # 1. 加载和准备数据 (Load and Prepare Data)
    print("Loading and preparing data...")
    full_train_dataset = PeptideDataset(TRAIN_CSV_PATH, sep='\t')
    test_dataset = PeptideDataset(TEST_CSV_PATH, sep='\t')

    if not full_train_dataset.sequences or not test_dataset.sequences:
        print("Error loading data. Exiting.")
        exit()

    # 划分训练集和验证集 (Split training data into train and validation)
    train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # 创建 DataLoaders (Create DataLoaders)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True if DEVICE == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True if DEVICE == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=True if DEVICE == 'cuda' else False)

    # 2. 计算类别先验概率 (Calculate Class Priors for Loss Function)
    print("Calculating class priors...")
    train_labels = [label for _, label, _ in train_dataset] # 获取训练集所有标签 (Get all labels from train subset)
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_priors = class_counts / len(train_labels)
    print(f"Class counts (Train): {class_counts}")
    print(f"Class priors (Train): {class_priors}")

    # 3. 初始化模型、损失函数、优化器 (Initialize Model, Loss, Optimizer)
    print("Initializing model...")
    model = BeeTLeNet(
        encoding_dim=ENCODING_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS,
        transformer_dim=TRANSFORMER_DIM,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_ff_dim=TRANSFORMER_FF_DIM,
        transformer_layers=TRANSFORMER_LAYERS,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        embedding_matrix=embedding_matrix # 使用预定义的编码矩阵 (Use predefined encoding matrix)
    ).to(DEVICE)

    criterion = FocalLogitAdjustedLoss(class_priors=class_priors, gamma=LOSS_GAMMA, tau=LOSS_TAU).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)

    # 4. 初始化学习率调度器 (Initialize Learning Rate Scheduler)
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = WARMUP_STEPS

    def lr_lambda(current_step):
        # Warmup 阶段 (Warmup phase)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine Decay 阶段 (Cosine Decay phase)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. 训练和验证循环 (Training and Validation Loop)
    print("Starting training...")
    best_val_auc = 0.0
    train_losses, val_losses, val_aucs = [], [], []

    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        val_metrics= evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        # val_losses.append(val_metrics['avg_loss'])
        # val_aucs.append(val_metrics['auc_roc'])

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['avg_loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"  Val BalAcc: {val_metrics['bacc']:.4f}")
        print(f"  Val AUC:    {val_metrics['auc_roc']:.4f}")
        print(f"  Epoch Time: {epoch_duration:.2f}s")

        # 保存最佳模型 (Save the best model based on validation AUC)
        if val_metrics['auc_roc'] > best_val_auc:
            print(f"Validation AUC improved ({best_val_auc:.4f} -> {val_metrics['auc_roc']:.4f}). Saving model...")
            best_val_auc = val_metrics['auc_roc']
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time:.2f}s")

    # 6. 加载最佳模型并进行最终测试 (Load best model and perform final testing)
    print("\nLoading best model for final evaluation on test set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    metrics = evaluate(model, test_loader, criterion, DEVICE)

    # print("\n--- Test Set Evaluation Results ---")
    # print(f"Test Loss:   {test_metrics['loss']:.4f}")
    # print(f"Test Acc:    {test_metrics['accuracy']:.4f}")
    # print(f"Test BalAcc: {test_metrics['balanced_accuracy']:.4f}")
    # print(f"Test AUC:    {test_metrics['auc']:.4f}")
    # print(f"Test Prec:   {test_metrics['precision']:.4f}")
    # print(f"Test Recall: {test_metrics['recall']:.4f}")
    # print(f"Test F1:     {test_metrics['f1']:.4f}")

    print(metrics)
    # 保存预测结果
    pd.DataFrame({'Predicted':metrics['preds'], 'True':metrics['trues']}) \
        .to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

    # Save predictions and true labels to CSV
    preds_df = pd.DataFrame({'Predicted': metrics['preds'], 'True': metrics['trues'],'probs':metrics['probs']})
    preds_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    print("Saved predictions and true labels to ./checkpoint_CALIBER/predictions.csv")



    if MODEL_SAVE_PATH is None or not os.path.exists(MODEL_SAVE_PATH):

        # Plot and save training loss curve
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_dir, 'loss.png'))
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
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
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
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
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
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    print("Saved confusion matrix to ./checkpoint_CALIBER/confusion_matrix.png")


    







    # # 绘制混淆矩阵 (Plot Confusion Matrix for Test Set)
    # try:
    #     cm = confusion_matrix(metrics['trues'],metrics['preds'])
    #     plt.figure(figsize=(6, 4))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #                 xticklabels=['Predicted Non-Epitope (0)', 'Predicted Epitope (1)'],
    #                 yticklabels=['Actual Non-Epitope (0)', 'Actual Epitope (1)'])
    #     plt.ylabel('Actual Label')
    #     plt.xlabel('Predicted Label')
    #     plt.title('Test Set Confusion Matrix')
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/test_confusion_matrix.png")
    #     print("\nTest confusion matrix plot saved as 'test_confusion_matrix.png'")
    # except Exception as plot_err:
    #     print(f"\nCould not generate test confusion matrix plot: {plot_err}")

    # # 绘制训练过程图 (Plot Training Progress)
    # try:
    #     plt.figure(figsize=(12, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    #     plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss')
    #     plt.legend()

    #     plt.subplot(1, 2, 2)
    #     plt.plot(range(1, EPOCHS + 1), val_aucs, label='Validation AUC')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('AUC')
    #     plt.title('Validation AUC')
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/training_progress.png")
    #     print("Training progress plot saved as 'training_progress.png'")
    # except Exception as plot_err:
    #     print(f"\nCould not generate training progress plot: {plot_err}")




    # try:
    #     # 假设 y_true, y_prob 已在评估后得到
    #     # y_true: 真实标签数组，y_prob: 正类概率数组

    #     # 1. 计算 ROC 曲线坐标
    #     fpr, tpr, thresholds = roc_curve(test_labels, test_preds)
    #     roc_auc = auc(fpr, tpr)

    #     # 2. 绘制 ROC 曲线
    #     plt.figure(figsize=(6,6))
    #     plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    #     # 对角线（随机分类器）
    #     plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate (1 - Specificity)')
    #     plt.ylabel('True Positive Rate (Sensitivity)')
    #     plt.title('ROC Curve (Test Set)')
    #     plt.legend(loc='lower right')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/roc.png")
    #     plt.show()
    # except Exception as plot_err:
    #     print(f"\nCould not generate testing ROC plot: {plot_err}")

    # print("\nScript finished.")