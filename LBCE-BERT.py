import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_score, recall_score, f1_score)
from xgboost import XGBClassifier
from pydpi.pypro import PyPro
from transformers import BertModel, BertTokenizer
import torch

warnings.filterwarnings("ignore")

# ---------------------- 全局配置 ----------------------
DATA_DIR = "./datasets/training/"
BERT_PROTEIN_DIR = "./Bert-Protein"
VOCAB_FILE = f"{BERT_PROTEIN_DIR}/vocab/vocab_1kmer.txt"
BERT_CONFIG_FILE = f"{BERT_PROTEIN_DIR}/bert_config_1.json"
INIT_CHECKPOINT = f"{BERT_PROTEIN_DIR}/model/1kmer_model/model.ckpt"
MODEL_SAVE_PATH = "./models/"
AAP_FILE = "./models/aap-general.txt.normal"
AAT_FILE = "./models/aat-general.txt.normal"
FEATURE_LIST = ['aac', 'dpc', 'aap', 'aat', 'bertfea']
MAX_SEQ_LENGTH = 128

# ---------------------- 数据加载 ----------------------
def load_data(train_csv, test_csv):
    train_df = pd.read_csv(f"{DATA_DIR}{train_csv}")
    test_df = pd.read_csv(f"{DATA_DIR}{test_csv}")
    for df in [train_df, test_df]:
        if not all(col in df.columns for col in ['sequence', 'label']):
            raise ValueError("数据必须包含 'sequence' 和 'label' 列")
    return (
        train_df['sequence'].tolist(), train_df['label'].tolist(),
        test_df['sequence'].tolist(), test_df['label'].tolist()
    )

# ---------------------- 特征提取核心函数 ----------------------
class FeatureExtractor:
    def __init__(self):
        self.protein = PyPro()
        self.aap_dic = self._read_aap(AAP_FILE)
        self.aat_dic = self._read_aat(AAT_FILE)
        self.tokenizer = BertTokenizer.from_pretrained(VOCAB_FILE)
        self.bert_model = BertModel.from_pretrained(BERT_PROTEIN_DIR)
        self.bert_model.eval()

    @staticmethod
    def _read_aap(file):
        aap = {}
        with open(file, 'r') as f:
            for line in f:
                aa_pair, score = line.strip().split()
                aap[aa_pair] = float(score)
        return aap

    @staticmethod
    def _read_aat(file):
        aat = {}
        with open(file, 'r') as f:
            for line in f:
                aa_triple, score = line.strip().split()
                aat[aa_triple[:3]] = float(score)
        return aat

    def aac(self, sequences):
        return np.array([
            list(self.protein.GetAAComp(seq).values()) 
            for seq in sequences
        ])

    def dpc(self, sequences):
        return np.array([
            list(self.protein.GetDPComp(seq).values()) 
            for seq in sequences
        ])

    def aap(self, sequences):
        features = []
        for seq in sequences:
            score = 0.0
            count = 0
            for i in range(len(seq)-1):
                key = seq[i:i+2]
                score += self.aap_dic.get(key, -1)
                count += 1
            features.append(score/count if count > 0 else 0.0)
        return np.array(features).reshape(-1, 1)

    def aat(self, sequences):
        features = []
        for seq in sequences:
            score = 0.0
            count = 0
            for i in range(len(seq)-2):
                key = seq[i:i+3]
                score += self.aat_dic.get(key, -1)
                count += 1
            features.append(score/count if count > 0 else 0.0)
        return np.array(features).reshape(-1, 1)

    def bert_fea(self, sequences):
        all_embeddings = []
        for seq in sequences:
            inputs = self.tokenizer(seq, return_tensors='pt', max_length=MAX_SEQ_LENGTH, 
                                    truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(cls_embedding.flatten())
        return np.array(all_embeddings)

# ---------------------- 特征组合与标准化 ----------------------
def combine_features(sequences, extractor):
    feature_dict = {
        'aac': extractor.aac(sequences),
        'dpc': extractor.dpc(sequences),
        'aap': extractor.aap(sequences),
        'aat': extractor.aat(sequences),
        'bertfea': extractor.bert_fea(sequences)
    }
    selected_features = [feature_dict[fea] for fea in FEATURE_LIST]
    return np.hstack(selected_features)

# ---------------------- 模型训练与交叉验证 ----------------------
def train_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier(
        max_depth=10,
        learning_rate=0.04,
        n_estimators=1300,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]
        metrics = {
            'auc_roc': round(roc_auc_score(y_val, y_proba_val), 4),
            'auc_pr': round(average_precision_score(y_val, y_proba_val), 4),
            'precision': round(precision_score(y_val, y_pred_val), 4),
            'recall': round(recall_score(y_val, y_pred_val), 4),
            'f1': round(f1_score(y_val, y_pred_val), 4)
        }
        cv_metrics.append(metrics)
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    test_metrics = {
        'auc_roc': round(roc_auc_score(y_test, y_proba_test), 4),
        'auc_pr': round(average_precision_score(y_test, y_proba_test), 4),
        'precision': round(precision_score(y_test, y_pred_test), 4),
        'recall': round(recall_score(y_test, y_pred_test), 4),
        'f1': round(f1_score(y_test, y_pred_test), 4)
    }
    model_info = {
        'model': model,
        'scaler': scaler,
        'feature_list': FEATURE_LIST,
        'aap_dic': extractor.aap_dic,
        'aat_dic': extractor.aat_dic,
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics
    }
    with open(f"{MODEL_SAVE_PATH}lbce_bert_new_model.pickle", 'wb') as f:
        pickle.dump(model_info, f)
    return model_info

# ---------------------- 主流程 ----------------------
if __name__ == "__main__":
    train_seqs, train_labels, test_seqs, test_labels = load_data(
        train_csv="training_dataset.csv", 
        test_csv="testing_dataset.csv"
    )
    extractor = FeatureExtractor()
    print("正在提取训练集特征...")
    X_train = combine_features(train_seqs, extractor)
    print("正在提取测试集特征...")
    X_test = combine_features(test_seqs, extractor)
    print("开始模型训练...")
    model_info = train_model(X_train, train_labels, X_test, test_labels)
    print("\n========== 5折交叉验证平均指标 ==========")
    for metric in ['auc_roc', 'auc_pr', 'precision', 'recall', 'f1']:
        avg = np.mean([m[metric] for m in model_info['cv_metrics']])
        print(f"{metric}: {avg:.4f}")
    print("\n========== 测试集最终指标 ==========")
    for k, v in model_info['test_metrics'].items():
        print(f"{k}: {v}")
    print("\n模型已保存至: ", f"{MODEL_SAVE_PATH}lbce_bert_new_model.pickle")    