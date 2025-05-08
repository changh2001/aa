import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    average_precision_score, balanced_accuracy_score
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


# 0. 配置参数和设备
# Hyperparameters (部分参考论文，部分为常用设置)
EMBEDDING_DIM = 100       # 嵌入维度 (论文提及)
LSTM_HIDDEN_DIM = 128     # LSTM 隐藏层维度 (论文提及)
MLP_HIDDEN_DIM = 64       # MLP 隐藏层维度 (论文提及)
LEARNING_RATE = 0.001     # 学习率 (论文提及)
BATCH_SIZE = 64           # 批处理大小 (论文提及)
NUM_EPOCHS = 5           # 训练轮数 (可根据实际情况调整)
MAX_SEQ_LENGTH = None     # 将根据训练数据动态确定最大序列长度
VOCAB = None              # 词汇表
PAD_TOKEN = "<PAD>"       # padding标记
UNK_TOKEN = "<UNK>"       # 未知氨基酸标记 (虽然理论上氨基酸种类固定，但以防万一)

# 检查是否有可用的GPU (MPS for Apple Silicon, CUDA for Nvidia)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# 1. 数据预处理和Dataset类
class EpitopeDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_seq_length):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.pad_idx = vocab[PAD_TOKEN]
        self.unk_idx = vocab[UNK_TOKEN]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 将氨基酸序列转换为数字索引
        numericalized_seq = [self.vocab.get(aa, self.unk_idx) for aa in seq]
        
        # 记录原始序列长度
        seq_len = len(numericalized_seq)

        # Padding
        if len(numericalized_seq) < self.max_seq_length:
            numericalized_seq += [self.pad_idx] * (self.max_seq_length - len(numericalized_seq))
        else:
            numericalized_seq = numericalized_seq[:self.max_seq_length]
            seq_len = self.max_seq_length # 如果截断，长度设为最大长度

        return {
            "sequence": torch.tensor(numericalized_seq, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "length": torch.tensor(seq_len, dtype=torch.long) # 传递原始长度用于pack_padded_sequence
        }

def build_vocab(sequences):
    """从序列数据构建词汇表"""
    all_chars = Counter()
    for seq in sequences:
        all_chars.update(list(seq))
    
    vocab = {token: i + 2 for i, token in enumerate(all_chars.keys())} # 0和1留给PAD和UNK
    vocab[PAD_TOKEN] = 0
    vocab[UNK_TOKEN] = 1
    return vocab

def load_and_preprocess_data(csv_path, vocab=None, max_seq_length=None, is_train=True):
    """加载CSV数据，并进行预处理"""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['sequence', 'label']) # 确保序列和标签不为空
    df['sequence'] = df['sequence'].astype(str) # 确保序列是字符串类型
    
    sequences = df['sequence'].tolist()
    labels = df['label'].astype(int).tolist()

    if is_train:
        global VOCAB, MAX_SEQ_LENGTH
        VOCAB = build_vocab(sequences)
        MAX_SEQ_LENGTH = max(len(s) for s in sequences)
        print(f"词汇表大小: {len(VOCAB)}")
        print(f"最大序列长度 (训练集): {MAX_SEQ_LENGTH}")
        current_vocab = VOCAB
        current_max_seq_length = MAX_SEQ_LENGTH
    else:
        if vocab is None or max_seq_length is None:
            raise ValueError("测试集必须提供词汇表和最大序列长度")
        current_vocab = vocab
        current_max_seq_length = max_seq_length
        # 确保测试集中的序列长度不超过训练时确定的最大长度
        # (EpitopeDataset的__getitem__中会处理截断)

    dataset = EpitopeDataset(sequences, labels, current_vocab, current_max_seq_length)
    return dataset, current_vocab, current_max_seq_length

# 2. CALIBER 模型 (线性表位，序列级别分类)
class CALIBERLinearSeq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, mlp_hidden_dim, output_dim, pad_idx):
        super(CALIBERLinearSeq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_dim,
            bidirectional=True,
            batch_first=True  # 输入和输出张量将以 (batch, seq, feature) 的形式提供
        )
        
        # MLP层
        # BiLSTM的输出维度是 lstm_hidden_dim * 2 (因为是双向的)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, sequence_batch, lengths_batch):
        # sequence_batch: (batch_size, seq_len)
        # lengths_batch: (batch_size)
        
        embedded = self.embedding(sequence_batch)
        # embedded: (batch_size, seq_len, embedding_dim)

        # 打包填充序列，提高LSTM效率
        # 注意：在将长度传递给 pack_padded_sequence 之前，需要将它们移到CPU上，因为pack_padded_sequence期望长度在CPU上
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_batch.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # 解包序列 (不是严格需要，因为我们将进行池化，但如果想查看每个时间步的输出则有用)
        # output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=self.embedding.num_embeddings) # total_length应为max_seq_length
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output: (batch_size, seq_len, lstm_hidden_dim * 2)

        # 池化操作：这里使用最大池化 (Max Pooling over time)
        # 我们在序列长度维度 (dim=1) 上取最大值
        pooled_output = torch.max(output, dim=1)[0]
        # pooled_output: (batch_size, lstm_hidden_dim * 2)
        
        # 或者，可以使用最后时间步的隐藏状态 (更复杂一些，因为要处理双向和padding)
        # hidden: (num_layers * num_directions, batch_size, lstm_hidden_dim)
        # last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) # 合并双向LSTM的最后一个隐藏状态
        # pooled_output = last_hidden

        mlp_output = self.mlp(pooled_output)
        # mlp_output: (batch_size, output_dim) -> 对于二分类，output_dim=1
        
        return mlp_output # 返回logits

# 3. 训练函数
def train_model(model, dataloader, optimizer, criterion, device):
    model.train() # 设置模型为训练模式
    epoch_loss = 0
    all_preds = []
    all_labels = []
    dataloader_progress = tqdm(dataloader, desc="Training", leave=True)

    for batch_idx, batch in enumerate(dataloader_progress):
    # for batch_idx, batch in enumerate(dataloader):
        sequences = batch["sequence"].to(device)
        labels = batch["label"].to(device).unsqueeze(1) # 调整标签维度以匹配输出
        lengths = batch["length"].to(device) # 长度信息

        optimizer.zero_grad() # 清零梯度
        
        predictions_logits = model(sequences, lengths) # 前向传播，得到logits
        
        loss = criterion(predictions_logits, labels) # 计算损失
        
        loss.backward() # 反向传播
        optimizer.step() # 更新权重
        
        epoch_loss += loss.item()

        # 收集预测和标签用于计算批次指标（可选）
        # preds_binary = (torch.sigmoid(predictions_logits) > 0.5).float()
        # all_preds.extend(preds_binary.cpu().numpy())
        # all_labels.extend(labels.cpu().numpy())
        # if batch_idx % 100 == 0:
            # print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return epoch_loss / len(dataloader)

# 4. 评估函数
def evaluate_model(model, dataloader, criterion, device):
    pth = r"C:\Users\freemiss\Downloads\code\checkpoints_caliber\caliber_linear_seq_model.pth"
    if os.path.exists(r"C:\Users\freemiss\Downloads\code\checkpoints_caliber\caliber_linear_seq_model.pth"):
        model.load_state_dict(torch.load(pth, map_location = device))
        
    model.to(device) # 设置模型为评估模式

    model.eval() # 设置模型为评估模式
    epoch_loss = 0
    all_predictions_probs = [] # 存储概率用于AUC等指标
    all_predictions_binary = [] # 存储二元预测 (0或1)
    all_labels_list = []

    with torch.no_grad(): # 在评估阶段不计算梯度
        for batch in tqdm(dataloader, desc="Testing", leave=True):
            sequences = batch["sequence"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            lengths = batch["length"].to(device)

            predictions_logits = model(sequences, lengths)
            loss = criterion(predictions_logits, labels)
            epoch_loss += loss.item()

            probabilities = torch.sigmoid(predictions_logits) # 转换为概率
            binary_preds = (probabilities > 0.5).float()      # 根据0.5阈值转换为0/1

            all_predictions_probs.extend(probabilities.cpu().numpy().flatten())
            all_predictions_binary.extend(binary_preds.cpu().numpy().flatten())
            all_labels_list.extend(labels.cpu().numpy().flatten())
    
    avg_loss = epoch_loss / len(dataloader)
    
    # 计算各项评估指标
    y_true = np.array(all_labels_list)
    y_pred_binary = np.array(all_predictions_binary)
    y_pred_probs = np.array(all_predictions_probs)

    # 确保 y_true 和 y_pred_probs 包含至少两个类别以计算 ROC AUC
    if len(np.unique(y_true)) < 2:
        print("警告: 真实标签中只存在一个类别，无法计算 ROC AUC 和 PR AUC。")
        roc_auc = float('nan')
        pr_auc = float('nan')
    else:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        pr_auc = average_precision_score(y_true, y_pred_probs) # AUC-PR

    acc = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall_sn = recall_score(y_true, y_pred_binary, zero_division=0) # Sn (Sensitivity) / Recall
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    bacc = balanced_accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 计算特异性 (Specificity) 用于验证 BACC
    # TN = cm[0, 0]
    # FP = cm[0, 1]
    # specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    # bacc_manual = (recall_sn + specificity) / 2

    print(f"评估损失: {avg_loss:.4f}")
    print(f"准确率 (Accuracy/ACC): {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC (Average Precision): {pr_auc:.4f}")
    print(f"精确率 (Precision/Pre): {precision:.4f}")
    print(f"召回率 (Recall/Sensitivity/Sn): {recall_sn:.4f}")
    print(f"F1 分数 (F1-Score): {f1:.4f}")
    print(f"马修斯相关系数 (MCC): {mcc:.4f}")
    print(f"平衡准确率 (Balanced Accuracy/BACC): {bacc:.4f}")
    # print(f"  (手动计算BACC): {bacc_manual:.4f}") # 验证
    print("混淆矩阵 (Confusion Matrix):")
    print(cm)
    save_dir = "./checkpoints_caliber"
    os.makedirs(save_dir, exist_ok=True)

    # torch.save(model.state_dict(), f"{save_dir}/final_model.pt")
    # print(f"模型已保存至 {save_dir}/final_model.pt")

    # 可视化混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('./checkpoints_caliber/confu_matric.png')
    plt.show()

    try:
        # 假设 y_true, y_prob 已在评估后得到
        # y_true: 真实标签数组，y_prob: 正类概率数组

        # 1. 计算 ROC 曲线坐标
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_binary)
        # roc_auc = auc(fpr, tpr)

        # 2. 绘制 ROC 曲线
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        # 对角线（随机分类器）
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve (Test Set)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./checkpoints_caliber/roc_curve.png')
        plt.show()
    except Exception as plot_err:
        print(f"\nCould not generate testing ROC plot: {plot_err}")



    return avg_loss, acc, roc_auc


# 5. 主执行流程
if __name__ == "__main__":
    # --- 创建虚拟数据文件 (用于本地测试，实际使用时请替换为您的文件路径) ---
    # 请确保您的工作目录下有 'training_dataset.csv' 和 'testing_dataset.csv'
    # 如果没有，以下代码会创建它们 (数据量较小，仅用于演示结构)
    
    # 检查文件是否存在，如果不存在则创建
    if not os.path.exists("training_dataset.csv"):
        print("创建虚拟 training_dataset.csv...")
        train_data = {
            'sequence': [
                "PEMLSCNKYPKDNDM", "AGKASCTLSEQLDFID", "FYPRLQASADWKPGHA", "IFLIVAAIVFITLCF",
                "IVGCNNADCVACSKSA", "QGPGAPQGPGAPQGPGAP", "KLKVSTNLFEGDKFVG", "IH",
                "WLLWPVTLACFVLAAV", "IRSLVLQTLP", "AYDIAVTAGPYV", "EADFRLNDSHKHKDKHKDRE"
            ] * 100, # 少量重复数据以增加数量
            'label': [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0] * 100
        }
        train_df = pd.DataFrame(train_data)
        # 确保有足够的正负样本，至少各1000个，以满足较大训练集的要求
        num_samples_needed = 2000 
        if len(train_df) < num_samples_needed:
            factor = (num_samples_needed // len(train_df)) + 1
            train_df = pd.concat([train_df] * factor, ignore_index=True).head(num_samples_needed)
        train_df.to_csv("training_dataset.csv", index=False)
        print("虚拟 training_dataset.csv 创建完成。")

    if not os.path.exists("testing_dataset.csv"):
        print("创建虚拟 testing_dataset.csv...")
        test_data = {
            'sequence': [
                "MALLGLLHRGTS", "GFFKEGSSVTLKHFFF", "NRGEDIQLLKNA", "FRLMRTNFLIKFLLI",
                "LLCEDGCLAL", "LIGTPVCINGLM", "GLVGVLTLDNQDLYGQ", "TQISGYTTAATV",
                "MRVIHFGAGSDKGVAP", "PRKLPWPTPKTHPVK", "QRVRVSWS", "EKYVKITGLYPT"
            ] * 50,
            'label': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0] * 50
        }
        test_df = pd.DataFrame(test_data)
        num_test_samples_needed = 500
        if len(test_df) < num_test_samples_needed:
            factor = (num_test_samples_needed // len(test_df)) + 1
            test_df = pd.concat([test_df] * factor, ignore_index=True).head(num_test_samples_needed)
        test_df.to_csv("testing_dataset.csv", index=False)
        print("虚拟 testing_dataset.csv 创建完成。")
    # --- 虚拟数据文件创建结束 ---

    train_file_path = "training_dataset.csv"
    test_file_path = "testing_dataset.csv"

    print("开始加载和预处理训练数据...")
    train_dataset, vocab, max_len = load_and_preprocess_data(train_file_path, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("训练数据加载完成。")

    print("\n开始加载和预处理测试数据...")
    # 测试集使用从训练集得到的词汇表和最大长度
    test_dataset, _, _ = load_and_preprocess_data(test_file_path, vocab=vocab, max_seq_length=max_len, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("测试数据加载完成。")

    # 初始化模型、损失函数和优化器
    VOCAB_SIZE = len(vocab)
    PAD_IDX = vocab[PAD_TOKEN]
    OUTPUT_DIM = 1 # 二分类问题，输出一个logit

    model = CALIBERLinearSeq(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        LSTM_HIDDEN_DIM,
        MLP_HIDDEN_DIM,
        OUTPUT_DIM,
        PAD_IDX
    ).to(DEVICE)

    # 打印模型结构
    print("\n模型结构:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数数量: {num_params}")


    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss() # 内置sigmoid，更稳定

    # print(f"\n开始在 {DEVICE} 上训练模型，共 {NUM_EPOCHS} 轮...")
    # for epoch in range(NUM_EPOCHS):
    #     print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    #     train_loss = train_model(model, train_dataloader, optimizer, criterion, DEVICE)
    #     print(f"Epoch {epoch+1} 结束, 训练损失: {train_loss:.4f}")
        
    #     # (可选) 在每个epoch后评估测试集，以观察性能变化
    #     if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS -1 : # 每5轮或最后一轮评估
    #          print(f"\n在测试集上评估 Epoch {epoch+1}:")
    #          test_loss, test_acc, test_roc_auc = evaluate_model(model, test_dataloader, criterion, DEVICE)
    #          print("-" * 30)

    # print("\n训练完成!")
    # print("=" * 50)


    
    print("在测试集上进行最终评估:")
    final_test_loss, final_test_acc, final_test_roc_auc = evaluate_model(model, test_dataloader, criterion, DEVICE)
    print("=" * 50)
    print("最终评估完成。")

    # 保存模型 (可选)
    # torch.save(model.state_dict(), "./checkpoints_caliber/caliber_linear_seq_model.pth")
    # print("模型已保存到 caliber_linear_seq_model.pth")
