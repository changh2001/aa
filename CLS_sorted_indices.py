import numpy as np
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_feature_importance_indices(bert_fea_path, labels_path, save_path="./models/CLS_sorted_indices.pkl"):
    """
    生成BERT特征重要性排序索引并保存
    :param bert_fea_path: BERT嵌入文件路径（CSV格式，每行768维）
    :param labels_path: 训练集标签路径（CSV格式，包含'label'列）
    :param save_path: 索引保存路径（.pkl格式）
    """
    # 加载BERT嵌入和标签
    X_bert = np.array(pd.read_csv(bert_fea_path, header=None, usecols=range(1, 769)))  # 跳过索引列
    y = pd.read_csv(labels_path)['label'].values
    
    # 标准化数据（XGBoost对标准化不敏感，但有助于加速训练）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bert)
    
    # 训练XGBoost模型（仅用于特征重要性计算）
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # 获取特征重要性并排序（从高到低）
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]  # 降序排列
    
    # 保存排序后的索引（前160个重要特征，可根据需求调整）
    joblib.dump(sorted_indices[:160], save_path)
    print(f"已保存前160个重要特征索引到 {save_path}")

if __name__ == "__main__":
    # 示例路径（根据实际数据调整）
    BERT_TRAIN_FEATURES = "bertfea/training/CLS_fea.txt"
    TRAIN_LABELS = "/Users/changhao/Documents/研一下/统计计算/homework/code/training_dataset.csv"
    
    generate_feature_importance_indices(BERT_TRAIN_FEATURES, TRAIN_LABELS)