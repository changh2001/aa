import pandas as pd

# 读取 CSV
df = pd.read_csv('/Users/changhao/Documents/研一下/统计计算/homework/code/training_dataset.csv')

# 计算每条序列的长度，并取最大值
max_len = df['sequence'].str.len().max()

print(f'最大序列长度为：{max_len}')

# 找到长度等于 max_len 的所有序列
longest_seqs = df[df['sequence'].str.len() == max_len]['sequence'].tolist()
print('达到最大长度的序列有：', longest_seqs)


# 读取 CSV
df = pd.read_csv('/Users/changhao/Documents/研一下/统计计算/homework/code/testing_dataset.csv')

# 计算每条序列的长度，并取最大值
max_len = df['sequence'].str.len().max()

print(f'最大序列长度为：{max_len}')

# 找到长度等于 max_len 的所有序列
longest_seqs = df[df['sequence'].str.len() == max_len]['sequence'].tolist()
print('达到最大长度的序列有：', longest_seqs)
