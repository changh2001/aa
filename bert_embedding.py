# bert_embedding.py
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from data_utils import load_csv_data
from tqdm import tqdm  # 确保在文件顶部导入tqdm

from huggingface_hub import hf_hub_download



class ProteinBERTEmbedder:
    def __init__(self, model_name="Rostlab/prot_bert", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()  # 推理模式

    def get_embeddings(self, sequences, output_path, batch_size=32):
        """
        提取序列的BERT嵌入（CLS标记）并保存为CSV
        :param sequences: 蛋白质序列列表
        :param output_path: 输出路径（CSV格式）
        :param batch_size: 批次大小
        """
        embeddings = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size  # 计算总批次数
        
        with torch.no_grad():
            # 添加带进度条的批次循环
            for i in tqdm(
                range(0, len(sequences), batch_size),
                total=total_batches,
                desc="Extracting embeddings",
                unit="batch"
            ):
                batch = sequences[i:i+batch_size]
                # 处理特殊字符（ProtBERT需要空格分隔氨基酸）
                batch = [" ".join(seq) for seq in batch]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=23  # 根据实际数据调整最大长度
                ).to(self.device)
                outputs = self.model(**inputs)
                # 提取CLS标记的嵌入（第一个token）
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)
        
        # 合并并保存为CSV（索引从0开始）
        embeddings = np.concatenate(embeddings, axis=0)
        df = pd.DataFrame(embeddings)

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, header=False, index_label="index")

if __name__ == "__main__":
    # 初始化嵌入器（首次运行会下载预训练模型）
    embedder = ProteinBERTEmbedder()

    # 获取模型文件的本地路径（以config.json为例）
    model_name = "Rostlab/prot_bert"
    local_config_path = hf_hub_download(
        repo_id=model_name,
        filename="config.json",
        # 可选：指定自定义缓存路径
        # cache_dir="/your/custom/cache/path"
    )

    # 获取模型根目录（需从文件路径中提取）
    cache_dir = os.path.dirname(local_config_path)
    print(f"模型缓存目录: {cache_dir}")

    # 提取训练集BERT嵌入（保存到bertfea/training目录）
    train_seqs, _ = load_csv_data("/Users/changhao/Documents/研一下/统计计算/homework/code/training_dataset.csv")
    embedder.get_embeddings(train_seqs, "./bertfea/training/CLS_fea.txt")
    
    # 提取测试集BERT嵌入（保存到bertfea/testing目录）
    test_seqs, _ = load_csv_data("/Users/changhao/Documents/研一下/统计计算/homework/code/testing_dataset.csv")
    embedder.get_embeddings(test_seqs, "./bertfea/testing/CLS_fea.txt")