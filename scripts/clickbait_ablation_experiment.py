# ============================================================
# 피처 추가 실험 (Ablation Study)
# ============================================================
# 1-4: Part1 + KPF-BERT + 코사인 유사도
# 2-4: Part2 + KPF-BERT + 주제 일관성 비율
# 2-5: Part2 + KPF-BERT + 주제 일관성 + 코사인 유사도
# ============================================================


# %% [markdown]
# ## 1. 라이브러리 설치 + GPU 확인

# %%
!pip install transformers accelerate sentence-transformers -q

import torch
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import torch.nn as nn

print(f"GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# %% [markdown]
# ## 2. 데이터 로드 + 30% 샘플링

# %%
# === Kaggle 경로 ===
TR_PATH = '/kaggle/input/datasets/tlarkdus/dataset/TR_preprocessed.csv'
VL_PATH = '/kaggle/input/datasets/tlarkdus/dataset/VL_preprocessed.csv'
SAVE_DIR = '/kaggle/working/models'
os.makedirs(SAVE_DIR, exist_ok=True)

tr_df = pd.read_csv(TR_PATH)
vl_df = pd.read_csv(VL_PATH)
print(f"원본 Training: {len(tr_df):,}건 / Validation: {len(vl_df):,}건")

# Part별 분리
train_p1 = tr_df[tr_df['partNum'] == 'P1'].reset_index(drop=True)
train_p2 = tr_df[tr_df['partNum'] == 'P2'].reset_index(drop=True)
vl_p1 = vl_df[vl_df['partNum'] == 'P1'].reset_index(drop=True)
vl_p2 = vl_df[vl_df['partNum'] == 'P2'].reset_index(drop=True)

# 30% 층화 샘플링 — Part별 87,000건
tr_p1, _ = train_test_split(train_p1, train_size=87000, stratify=train_p1['clickbaitClass'], random_state=42)
tr_p2, _ = train_test_split(train_p2, train_size=87000, stratify=train_p2['clickbaitClass'], random_state=42)
tr_p1 = tr_p1.reset_index(drop=True)
tr_p2 = tr_p2.reset_index(drop=True)

print(f"\n✅ 30% 샘플링 완료!")
print(f"[Part1] Training: {len(tr_p1):,}건 / Validation: {len(vl_p1):,}건")
print(f"[Part2] Training: {len(tr_p2):,}건 / Validation: {len(vl_p2):,}건")


# %% [markdown]
# ## 3. 코사인 유사도 계산 (Sentence-BERT)

# %%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# Sentence-BERT 모델 로드 (한국어 지원)
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Sentence-BERT 모델 로드 완료!")

def compute_cosine_similarity(titles, contents, batch_size=256):
    """제목과 본문의 코사인 유사도를 배치로 계산"""
    similarities = []
    for i in tqdm(range(0, len(titles), batch_size), desc="코사인 유사도 계산"):
        batch_titles = titles[i:i+batch_size]
        batch_contents = contents[i:i+batch_size]

        # 본문이 길면 앞부분만 사용 (Sentence-BERT 입력 제한)
        batch_contents = [c[:512] if len(c) > 512 else c for c in batch_contents]

        title_embs = sbert_model.encode(batch_titles, show_progress_bar=False)
        content_embs = sbert_model.encode(batch_contents, show_progress_bar=False)

        for t_emb, c_emb in zip(title_embs, content_embs):
            sim = cos_sim([t_emb], [c_emb])[0][0]
            similarities.append(sim)

    return np.array(similarities)

# Part1 코사인 유사도 계산 (fakeTitle vs newsContent)
print("\n📌 Part1 Training 코사인 유사도 계산...")
tr_p1_titles = tr_p1['fakeTitle'].fillna('').tolist()
tr_p1_contents = tr_p1['newsContent'].fillna('').tolist()
tr_p1['cosine_sim'] = compute_cosine_similarity(tr_p1_titles, tr_p1_contents)

print("📌 Part1 Validation 코사인 유사도 계산...")
vl_p1_titles = vl_p1['fakeTitle'].fillna('').tolist()
vl_p1_contents = vl_p1['newsContent'].fillna('').tolist()
vl_p1['cosine_sim'] = compute_cosine_similarity(vl_p1_titles, vl_p1_contents)

# Part2 코사인 유사도 계산 (newsTitle vs newsContent)
print("\n📌 Part2 Training 코사인 유사도 계산...")
tr_p2_titles = tr_p2['newsTitle'].fillna('').tolist()
tr_p2_contents = tr_p2['newsContent'].fillna('').tolist()
tr_p2['cosine_sim'] = compute_cosine_similarity(tr_p2_titles, tr_p2_contents)

print("📌 Part2 Validation 코사인 유사도 계산...")
vl_p2_titles = vl_p2['newsTitle'].fillna('').tolist()
vl_p2_contents = vl_p2['newsContent'].fillna('').tolist()
vl_p2['cosine_sim'] = compute_cosine_similarity(vl_p2_titles, vl_p2_contents)

print("\n✅ 코사인 유사도 계산 완료!")
print(f"Part1 Training 코사인 유사도 평균: {tr_p1['cosine_sim'].mean():.4f}")
print(f"Part2 Training 코사인 유사도 평균: {tr_p2['cosine_sim'].mean():.4f}")

# Sentence-BERT 모델 메모리 해제
del sbert_model
torch.cuda.empty_cache()


# %% [markdown]
# ## 4. 입력 텍스트 구성

# %%
# Part1: fakeTitle [SEP] newsContent
tr_p1['input_text'] = tr_p1['fakeTitle'].fillna('') + ' [SEP] ' + tr_p1['newsContent'].fillna('')
vl_p1['input_text'] = vl_p1['fakeTitle'].fillna('') + ' [SEP] ' + vl_p1['newsContent'].fillna('')

# Part2: newsTitle [SEP] newsContent
tr_p2['input_text'] = tr_p2['newsTitle'].fillna('') + ' [SEP] ' + tr_p2['newsContent'].fillna('')
vl_p2['input_text'] = vl_p2['newsTitle'].fillna('') + ' [SEP] ' + vl_p2['newsContent'].fillna('')

print("✅ 입력 텍스트 구성 완료!")


# %% [markdown]
# ## 5. 피처 추가 모델 정의

# %%
class BertWithFeatures(nn.Module):
    """BERT [CLS] 출력 + 추가 피처를 결합하는 분류 모델"""
    def __init__(self, model_name, num_features, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # [CLS] hidden (768) + 추가 피처 (num_features) → 분류
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, features):
        # BERT 출력에서 [CLS] 토큰 추출
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        # [CLS] + 피처 결합
        combined = torch.cat([cls_output, features], dim=1)  # [batch, 768 + num_features]

        # 분류
        logits = self.classifier(combined)
        return logits


class ClickbaitFeatureDataset(Dataset):
    """텍스트 + 추가 피처를 함께 제공하는 Dataset"""
    def __init__(self, texts, labels, features, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.features = features  # numpy array (N, num_features)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

print("✅ 피처 추가 모델 + Dataset 정의 완료!")


# %% [markdown]
# ## 6. 학습/평가 함수 (피처 포함)

# %%
def train_epoch_with_features(model, dataloader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="학습 중"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_with_features(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="평가 중"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    print(f"\n📊 평가 결과:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['낚시성(0)', '정상(1)'])}")

    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def run_feature_experiment(exp_name, model_name, tr_texts, tr_labels, tr_features,
                           vl_texts, vl_labels, vl_features, num_features,
                           num_epochs=3, batch_size=16, lr=1e-5):
    """피처 추가 실험 실행 함수"""
    print(f"\n{'=' * 60}")
    print(f"🚀 {exp_name} 시작")
    print(f"  모델: {model_name}")
    print(f"  추가 피처 수: {num_features}")
    print(f"  Training: {len(tr_texts):,}건 / Validation: {len(vl_texts):,}건")
    print(f"{'=' * 60}")

    # 토크나이저 + 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertWithFeatures(model_name, num_features=num_features, num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Dataset + DataLoader
    tr_dataset = ClickbaitFeatureDataset(tr_texts, tr_labels, tr_features, tokenizer)
    vl_dataset = ClickbaitFeatureDataset(vl_texts, vl_labels, vl_features, tokenizer)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(vl_dataset, batch_size=batch_size, shuffle=False)

    # 옵티마이저 + 손실 함수
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0
    best_results = None

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        avg_loss = train_epoch_with_features(model, tr_loader, optimizer, device, loss_fn)
        print(f"평균 Loss: {avg_loss:.4f}")

        print(f"\n[Validation 평가]")
        results = evaluate_with_features(model, vl_loader, device)

        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_results = results
            save_path = os.path.join(SAVE_DIR, f"{exp_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"💾 최고 모델 저장! (F1: {best_f1:.4f})")

    print(f"\n{'=' * 60}")
    print(f"✅ {exp_name} 완료! 최고 F1: {best_f1:.4f}")
    print(f"{'=' * 60}")

    del model, tokenizer, tr_dataset, vl_dataset, tr_loader, vl_loader, optimizer
    torch.cuda.empty_cache()

    return {'experiment': exp_name, 'model': model_name, 'best_f1': best_f1, **best_results}

print("✅ 피처 실험 실행 함수 정의 완료!")


# %% [markdown]
# ## 7. 실험 1-4: Part1 + KPF-BERT + 코사인 유사도

# %%
# 피처: 코사인 유사도 (1개)
tr_p1_features = tr_p1[['cosine_sim']].values  # (87000, 1)
vl_p1_features = vl_p1[['cosine_sim']].values

results_1_4 = run_feature_experiment(
    exp_name='exp1_4_P1_KPF_BERT_cosine',
    model_name='jinmang2/kpfbert',
    tr_texts=tr_p1['input_text'].tolist(),
    tr_labels=tr_p1['clickbaitClass'].tolist(),
    tr_features=tr_p1_features,
    vl_texts=vl_p1['input_text'].tolist(),
    vl_labels=vl_p1['clickbaitClass'].tolist(),
    vl_features=vl_p1_features,
    num_features=1
)


# %% [markdown]
# ## 8. 실험 2-4: Part2 + KPF-BERT + 주제 일관성 비율

# %%
# 피처: subjectConsistencyRatio (1개)
tr_p2_features_scr = tr_p2[['subjectConsistencyRatio']].fillna(0).values  # (87000, 1)
vl_p2_features_scr = vl_p2[['subjectConsistencyRatio']].fillna(0).values

results_2_4 = run_feature_experiment(
    exp_name='exp2_4_P2_KPF_BERT_scr',
    model_name='jinmang2/kpfbert',
    tr_texts=tr_p2['input_text'].tolist(),
    tr_labels=tr_p2['clickbaitClass'].tolist(),
    tr_features=tr_p2_features_scr,
    vl_texts=vl_p2['input_text'].tolist(),
    vl_labels=vl_p2['clickbaitClass'].tolist(),
    vl_features=vl_p2_features_scr,
    num_features=1
)


# %% [markdown]
# ## 9. 실험 2-5: Part2 + KPF-BERT + 주제 일관성 + 코사인 유사도

# %%
# 피처: subjectConsistencyRatio + cosine_sim (2개)
tr_p2_features_both = tr_p2[['subjectConsistencyRatio', 'cosine_sim']].fillna(0).values  # (87000, 2)
vl_p2_features_both = vl_p2[['subjectConsistencyRatio', 'cosine_sim']].fillna(0).values

results_2_5 = run_feature_experiment(
    exp_name='exp2_5_P2_KPF_BERT_both',
    model_name='jinmang2/kpfbert',
    tr_texts=tr_p2['input_text'].tolist(),
    tr_labels=tr_p2['clickbaitClass'].tolist(),
    tr_features=tr_p2_features_both,
    vl_texts=vl_p2['input_text'].tolist(),
    vl_labels=vl_p2['clickbaitClass'].tolist(),
    vl_features=vl_p2_features_both,
    num_features=2
)


# %% [markdown]
# ## 10. Ablation Study 결과 정리

# %%
all_results = []
for r in [results_1_4, results_2_4, results_2_5]:
    if r is not None:
        all_results.append(r)

results_df = pd.DataFrame(all_results)
print("\n📊 Ablation Study 결과:")
print(results_df[['experiment', 'model', 'accuracy', 'f1', 'precision', 'recall']].to_string(index=False))

# CSV 저장
results_csv_path = os.path.join(SAVE_DIR, 'ablation_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\n💾 결과 저장 완료: {results_csv_path}")
