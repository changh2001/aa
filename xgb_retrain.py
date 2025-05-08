# xgb_retrain.py
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import numpy as np
import pandas as pd
import warnings
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import pickle
from pydpi.pypro import PyPro  # 确保已安装pydpi库
from data_utils import load_csv_data

# 忽略警告
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

protein = PyPro()

# ---------------------- 特征提取函数 ----------------------
def readAAP(file):
    """读取AAP特征文件"""
    try:
        aap_dic = {}
        with open(file, 'r') as f:
            for line in f:
                aap_dic[line.split()[0]] = float(line.split()[1])
        return aap_dic
    except:
        print("Error reading AAP file")
        sys.exit()

def readAAT(file):
    """读取AAT特征文件"""
    try:
        aat_dic = {}
        with open(file, 'r') as f:
            for line in f:
                aat_dic[line.split()[0][:3]] = float(line.split()[1])
        return aat_dic
    except:
        print("Error reading AAT file")
        sys.exit()

def calculate_AAP(pep, aap_dic, avg=1):
    """计算AAP特征"""
    features = []
    for seq in pep:
        score, count = 0.0, 0
        for i in range(len(seq) - 1):
            try:
                score += aap_dic[seq[i:i+2]]
                count += 1
            except KeyError:
                score -= 1.0
                count += 1
        features.append(score / count if count > 0 else 0.0)
    return np.array(features).reshape(-1, 1)

def calculate_AAT(pep, aat_dic, avg=1):
    """计算AAT特征"""
    features = []
    for seq in pep:
        score, count = 0.0, 0
        for i in range(len(seq) - 2):
            try:
                score += aat_dic[seq[i:i+3]]
                count += 1
            except KeyError:
                score -= 1.0
                count += 1
        features.append(score / count if count > 0 else 0.0)
    return np.array(features).reshape(-1, 1)

def calculate_AAC(pep):
    """计算AAC特征"""
    features = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        features.append(list(protein.GetAAComp().values()))
    return np.array(features)

def calculate_BERT_fea(file):
    """读取BERT嵌入特征（已预处理）"""
    idx_sorted = joblib.load('./models/CLS_sorted_indices.pkl')  # 假设已预计算特征选择索引
    x = np.array(pd.read_csv(file, header=None, usecols=range(1, 769)))
    return x[:, idx_sorted[:160]]  # 选择前160个重要特征

def combine_features(pep, feature_list):
    """组合多种特征"""
    features = []
    
    if 'aac' in feature_list:
        aac = calculate_AAC(pep)
        features.append(aac)
    
    if 'aap' in feature_list:
        aap_dic = readAAP("./models/aap-general.txt.normal")
        aap = calculate_AAP(pep, aap_dic)
        features.append(aap)
    
    if 'aat' in feature_list:
        aat_dic = readAAT("./models/aat-general.txt.normal")
        aat = calculate_AAT(pep, aat_dic)
        features.append(aat)
    
    if 'bertfea' in feature_list:
        bert_fea = calculate_BERT_fea('./bertfea/training/CLS_fea.txt')  # 训练集BERT路径
        features.append(bert_fea)
    
    return np.hstack(features) if features else np.array([])

# ---------------------- 模型训练主函数 ----------------------
def train_model(train_seqs, train_labels, save_path="./models/xgb-ibce-final.pickle"):
    # 组合特征
    feature_list = ['aac', 'aap', 'aat', 'bertfea']
    X = combine_features(train_seqs, feature_list)
    y = np.array(train_labels)
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 定义评估指标
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score)
    }
    
    # XGBoost参数调优
    param_grid = {
        'max_depth': [8, 10],
        'learning_rate': [0.03, 0.05],
        'n_estimators': [1000, 1500]
    }
    model = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', random_state=42),
        param_grid=param_grid,
        scoring=scoring,
        refit='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # 保存模型及元数据
    pickle_info = {
        'model': model.best_estimator_,
        'scaler': scaler,
        'feature_list': feature_list,
        'best_params': model.best_params_
    }
    with open(save_path, 'wb') as f:
        pickle.dump(pickle_info, f)
    
    print(f"最佳参数: {model.best_params_}")
    print(f"最佳ROC-AUC: {model.best_score_:.4f}")
    return model

if __name__ == "__main__":
    # 加载训练数据（参数为训练集路径）
    train_seqs, train_labels = load_csv_data(sys.argv[1])
    # 训练模型
    train_model(train_seqs, train_labels)