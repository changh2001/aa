# data_utils.py
import pandas as pd

def load_csv_data(file_path):
    """
    加载CSV格式的训练/测试数据，过滤非法序列
    :param file_path: CSV文件路径（包含'sequence'和'label'列）
    :return: 序列列表，标签列表
    """
    df = pd.read_csv(file_path)
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    # 过滤包含非标准氨基酸的序列
    valid_mask = df['sequence'].apply(lambda s: all(c in valid_amino_acids for c in s))
    return df[valid_mask]['sequence'].tolist(), df[valid_mask]['label'].tolist()


