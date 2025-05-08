# predict.py
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from data_utils import load_csv_data
from xgb_retrain import combine_features  # 复用特征组合函数

def evaluate_model(test_seqs, test_labels, model_path="./models/xgb-ibce-final.pickle"):
    # 加载模型
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    model = model_info['model']
    scaler = model_info['scaler']
    feature_list = model_info['feature_list']
    
    # 提取测试集特征
    X = combine_features(test_seqs, feature_list)
    X_scaled = scaler.transform(X)
    
    # 预测
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    # 评估指标
    print("=== 测试集评估结果 ===")
    print(f"准确率: {accuracy_score(test_labels, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(test_labels, y_proba):.4f}")
    print(f"F1分数: {f1_score(test_labels, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, y_pred))
    
    # 保存预测结果
    pd.DataFrame({
        'sequence': test_seqs,
        'true_label': test_labels,
        'pred_label': y_pred,
        'pred_prob': y_proba
    }).to_csv("test_predictions.csv", index=False)

if __name__ == "__main__":
    # 加载测试数据（参数为测试集路径）
    test_seqs, test_labels = load_csv_data(sys.argv[1])
    # 评估模型
    evaluate_model(test_seqs, test_labels)