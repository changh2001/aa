import pandas as pd

# 读取训练集与测试集
train_df = pd.read_csv('/Users/changhao/Documents/研一下/统计计算/homework/code/training dataset.csv')  # 假设文件名为 training.csv
test_df  = pd.read_csv('/Users/changhao/Documents/研一下/统计计算/homework/code/testing dataset.csv')   # 假设文件名为 testing.csv

# 查看前几行，确认读取正常
print(train_df.head())
print(test_df.head())

# 检查空值并删除（如有）
train_df.dropna(subset=['sequence','label'], inplace=True)
test_df.dropna(subset=['sequence','label'], inplace=True)


import torch
import numpy as np
from transformers import BertModel, BertTokenizer

# 使用 BioSequenceAnalysis/Bert-Protein 预训练模型
tokenizer = BertTokenizer.from_pretrained('BioSequenceAnalysis/Bert-Protein', do_lower_case=False)
model     = BertModel.from_pretrained('BioSequenceAnalysis/Bert-Protein')
model.eval()  # 推理模式

def embed_sequence(seq: str) -> np.ndarray:
    """
    将氨基酸序列转为 BERT 嵌入向量：
    - 在每个字符之间添加空格，使 tokenizer 以「单氨基酸」为 token
    - 取最后一层隐藏状态的均值作为固定长度特征
    """
    spaced = ' '.join(list(seq))
    inputs = tokenizer(spaced, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # [batch_size, seq_len, hidden_size] -> 均值池化 -> [hidden_size]
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# 对训练集和测试集批量提取嵌入（也可并行加速）
X_train = np.vstack(train_df['sequence'].apply(embed_sequence).values)
X_test  = np.vstack(test_df['sequence'].apply(embed_sequence).values)
y_train = train_df['label'].values
y_test  = test_df['label'].values
