import pandas as pd

def fasta_to_csv(input_file, output_file):
    data = []
    current_label = None
    current_sequence = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存前一个序列（如果存在）
                if current_label is not None:
                    data.append({
                        'sequence': ''.join(current_sequence),
                        'label': current_label
                    })
                # 解析新header
                parts = line[1:].split('_')
                current_label = int(parts[-1])  # 提取标签
                current_sequence = []
            else:
                # 新增：移除所有横线并转换为大写
                cleaned_line = line.replace('-', '').upper()
                if cleaned_line:  # 过滤空行
                    current_sequence.append(cleaned_line)
        
        # 添加最后一个序列
        if current_label is not None:
            data.append({
                'sequence': ''.join(current_sequence),
                'label': current_label
            })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"转换完成，共处理{len(data)}条数据，示例：")
    print(df.head())

# 使用示例
fasta_to_csv('training dataset.fasta', 'training dataset.csv')

fasta_to_csv('testing dataset.fasta', 'testing dataset.csv')

