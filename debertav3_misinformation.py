import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import os
import json
from typing import List, Dict, Tuple
import redis
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# 1. CONFIGURATION
class Config:
    # Model settings
    MODEL_NAME = 'microsoft/deberta-v3-base'
    EXPLANATION_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE_BERT = 2e-5
    LEARNING_RATE_CLF = 1e-3
    METADATA_DIM = 0
    NUM_CLASSES = 2
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # RAG settings
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    TOP_K_RETRIEVAL = 3
    VECTOR_DIM = 384
    
    # Paths
    ARTICLES_PATH = 'fact_check_articles_averitec.csv'
    VECTOR_STORE_INDEX = 'fact_check_vectors'


# 2. RAG PIPELINE - VECTOR STORE MANAGEMENT
class RedisVectorStore:
    
    def __init__(self, host: str, port: int, db: int, embedding_model: str):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.redis_client.ping()
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"Connected to Redis at {host}:{port}")
            print(f"Loaded embedding model: {embedding_model}")
            self.available = True
        except Exception as e:
            print(f"Could not connect to Redis: {e}")
            print("RAG features will be disabled")
            self.available = False
    
    def index_articles(self, articles: List[Dict], index_name: str = 'fact_check'):
        if not self.available:
            return
        
        print(f"\n Indexing {len(articles)} articles")
        
        for i, article in enumerate(tqdm(articles, desc="Embedding articles")):
            text = f"{article['title']} {article['text']}"
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            key = f"{index_name}:{article['id']}"
            self.redis_client.hset(key, mapping={
                'embedding': embedding.tobytes(),
                'metadata': json.dumps({
                    'text': article['text'],
                    'title': article['title'],
                    'url': article.get('url', ''),
                    'verdict': article.get('verdict', ''),
                    'id': article['id']
                })
            })
        
        print(f"Indexed {len(articles)} articles in Redis")
    
    def search(self, query: str, top_k: int = 3, index_name: str = 'fact_check') -> List[Dict]:
        if not self.available:
            return []
        
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        keys = self.redis_client.keys(f"{index_name}:*")
        results = []
        
        for key in keys:
            data = self.redis_client.hgetall(key)
            if b'embedding' in data and b'metadata' in data:
                stored_embedding = np.frombuffer(data[b'embedding'], dtype=np.float32)
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                metadata = json.loads(data[b'metadata'].decode('utf-8'))
                metadata['similarity'] = float(similarity)
                results.append(metadata)
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def clear_index(self, index_name: str = 'fact_check'):
        if not self.available:
            return
        keys = self.redis_client.keys(f"{index_name}:*")
        if keys:
            self.redis_client.delete(*keys)
        print(f"Cleared index: {index_name}")


# 3. DATA PREPARATION
def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file at: {filepath}")
        
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state', 'party', 'barely_true_ct', 'false_ct', 'half_true_ct', 
        'mostly_true_ct', 'pants_fire_ct', 'context'
    ]
    df = pd.read_csv(filepath, sep='\t', header=None, names=columns)
    
    count_cols = ['barely_true_ct', 'false_ct', 'half_true_ct', 'mostly_true_ct', 'pants_fire_ct']
    df[count_cols] = df[count_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def load_fact_check_articles(filepath: str) -> List[Dict]:
    if not os.path.exists(filepath):
        print(f"Warning: Articles file not found at {filepath}")
        return []
    
    df = pd.read_csv(filepath)
    articles = []
    for _, row in df.iterrows():
        articles.append({
            'id': str(row.get('id', '')),
            'title': str(row.get('title', '')),
            'text': str(row.get('text', '')),
            'url': str(row.get('url', '')),
            'verdict': str(row.get('verdict', ''))
        })
    return articles


def convert_to_binary(label):
    reliable = ['true', 'mostly-true', 'half-true']
    return 1 if label in reliable else 0


def preprocess_features_with_rag(df: pd.DataFrame, tokenizer, vector_store: RedisVectorStore) -> Tuple:
    count_cols = ['barely_true_ct', 'false_ct', 'half_true_ct', 'mostly_true_ct', 'pants_fire_ct']
    metadata = df[count_cols].apply(np.log1p)
    
    total_statements = df[count_cols].sum(axis=1)
    metadata['log_total_statements'] = np.log1p(total_statements)
    
    def map_party(p):
        if p == 'democrat': return 0
        if p == 'republican': return 1
        return 2
    
    metadata['party_enc'] = df['party'].apply(map_party)
    meta_features = metadata.values.astype(np.float32)
    
    sep = tokenizer.sep_token
    texts = []
    retrieved_contexts = []
    
    use_rag = vector_store and vector_store.available
    
    if use_rag:
        print("\n Retrieving relevant articles")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing", disable=not use_rag):
        statement = str(row['statement'])
        
        if use_rag:
            retrieved_articles = vector_store.search(statement, top_k=Config.TOP_K_RETRIEVAL)
            context_parts = []
            for i, article in enumerate(retrieved_articles):
                context_parts.append(f"[Source {i+1}]: {article['title'][:100]}")
            retrieved_context = " | ".join(context_parts)
            retrieved_contexts.append(retrieved_context)
        else:
            retrieved_context = "No RAG context"
            retrieved_contexts.append(retrieved_context)
        
        parts = [
            f"Statement: {statement}",
            f"Evidence: {retrieved_context}",
            f"Context: {str(row['context'])}",
            f"Subject: {str(row['subject'])}",
            f"Speaker: {str(row['speaker'])}"
        ]
        text = f" {sep } ".join([p for p in parts if p and 'nan' not in p.lower()])
        texts.append(text)
    
    return texts, meta_features, df['label'].values, retrieved_contexts



# 4. DATASET CLASS
class EnhancedMisinformationDataset(Dataset):
    def __init__(self, texts, metadata, labels, tokenizer, max_len):
        self.texts = texts
        self.metadata = metadata
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        meta = self.metadata[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(meta, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 5. MODEL ARCHITECTURE
class HybridDeBERTa(nn.Module):
    def __init__(self, base_model_name, num_classes, metadata_dim):
        super(HybridDeBERTa, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.text_hidden_size = self.bert.config.hidden_size
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        fusion_dim = self.text_hidden_size + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, metadata):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]
        meta_features = self.meta_mlp(metadata)
        combined = torch.cat((text_features, meta_features), dim=1)
        logits = self.classifier(combined)
        return logits


# 6. EXPLANATION GENERATION
class ExplanationGenerator:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = Config.EXPLANATION_MODEL
        
        print(f"\n Initializing explanation generator")
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )
            print("Explanation model loaded")
        except Exception as e:
            print(f"LLM not available: {e}")
            print("Using rule-based explanations")
            self.pipeline = None
    
    def create_prompt(self, statement: str, prediction: int, speaker: str, 
                     party: str, retrieved_context: str, speaker_history: Dict) -> str:
        label = "UNRELIABLE" if prediction == 0 else "RELIABLE"
        
        prompt = f"""Analyze this statement:

STATEMENT: "{statement}"
SPEAKER: {speaker} ({party})
PREDICTION: {label}

SPEAKER HISTORY:
- False: {speaker_history['false_ct']}
- Barely true: {speaker_history['barely_true_ct']}
- Half true: {speaker_history['half_true_ct']}
- Mostly true: {speaker_history['mostly_true_ct']}
- Pants on fire: {speaker_history['pants_fire_ct']}

RETRIEVED EVIDENCE:
{retrieved_context}

Explain in 2-3 sentences why this is {label}:"""
        
        return prompt
    
    def generate(self, statement: str, prediction: int, speaker: str, 
                party: str, retrieved_context: str, speaker_history: Dict) -> str:
        
        if self.pipeline is None:
            label = "unreliable" if prediction == 0 else "reliable"
            return (f"Based on analysis of the statement by {speaker}, combined with "
                    f"retrieved fact-checking evidence, this statement is classified as {label}. "
                    f"The model considered both content and speaker credibility.")
        
        prompt = self.create_prompt(statement, prediction, speaker, party, 
                                   retrieved_context, speaker_history)
        
        try:
            result = self.pipeline(prompt, max_new_tokens=300)[0]['generated_text']
            if "Explain in" in result:
                result = result.split("Explain in")[-1].split("sentences")[-1].strip()
            return result[:400]
        except:
            return self._mock_explanation(statement, prediction, speaker)
    
    def _mock_explanation(self, statement: str, prediction: int, speaker: str) -> str:
        label = "unreliable" if prediction == 0 else "reliable"
        return f"Statement by {speaker} classified as {label} based on content and credibility analysis."


# 7. TRAINING FUNCTIONS
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(dataloader), correct / total


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, metadata)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return total_loss / len(dataloader), correct / total, all_labels, all_preds


def eval_with_explanations(model, dataloader, df, contexts, explainer, device, n=10):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, metadata)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print(f"\n Generating {n} explanations")
    explanations = []
    indices = np.random.choice(len(df), min(n, len(df)), replace=False)
    
    for idx in tqdm(indices, desc="Explanations"):
        row = df.iloc[idx]
        pred = all_preds[idx]
        
        history = {
            'false_ct': int(row['false_ct']),
            'barely_true_ct': int(row['barely_true_ct']),
            'half_true_ct': int(row['half_true_ct']),
            'mostly_true_ct': int(row['mostly_true_ct']),
            'pants_fire_ct': int(row['pants_fire_ct'])
        }
        
        exp = explainer.generate(
            row['statement'], pred, row['speaker'], row['party'], contexts[idx], history
        )
        
        explanations.append({
            'statement': row['statement'],
            'speaker': row['speaker'],
            'true_label': 'Unreliable' if all_labels[idx] == 0 else 'Reliable',
            'predicted': 'Unreliable' if pred == 0 else 'Reliable',
            'confidence': float(all_probs[idx][pred]),
            'explanation': exp,
            'retrieved': contexts[idx][:200] + '...'
        })
    
    return all_labels, all_preds, explanations


# 8. MAIN
if __name__ == "__main__":
    print("MISINFORMATION DETECTION WITH RAG & EXPLANATIONS")
    print(f"Device: {Config.DEVICE}\n")
    
    # RAG Setup
    print("STEP 1: RAG PIPELINE")
    
    vector_store = RedisVectorStore(
        Config.REDIS_HOST, Config.REDIS_PORT, Config.REDIS_DB, Config.EMBEDDING_MODEL
    )
    
    if vector_store.available:
        articles = load_fact_check_articles(Config.ARTICLES_PATH)
        if articles:
            print(f"Loaded {len(articles)} articles")
            vector_store.clear_index(Config.VECTOR_STORE_INDEX)
            vector_store.index_articles(articles, Config.VECTOR_STORE_INDEX)
    
    # Load Data
    print("STEP 2: LOAD DATA")
    
    train_df = load_data('train.tsv')
    valid_df = load_data('valid.tsv')
    test_df = load_data('test.tsv')
    
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")
    
    train_df['label'] = train_df['label'].apply(convert_to_binary)
    valid_df['label'] = valid_df['label'].apply(convert_to_binary)
    test_df['label'] = test_df['label'].apply(convert_to_binary)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_texts, train_meta, train_y, train_ctx = preprocess_features_with_rag(train_df, tokenizer, vector_store)
    valid_texts, valid_meta, valid_y, valid_ctx = preprocess_features_with_rag(valid_df, tokenizer, vector_store)
    test_texts, test_meta, test_y, test_ctx = preprocess_features_with_rag(test_df, tokenizer, vector_store)
    
    Config.METADATA_DIM = train_meta.shape[1]
    
    train_ds = EnhancedMisinformationDataset(train_texts, train_meta, train_y, tokenizer, Config.MAX_LEN)
    valid_ds = EnhancedMisinformationDataset(valid_texts, valid_meta, valid_y, tokenizer, Config.MAX_LEN)
    test_ds = EnhancedMisinformationDataset(test_texts, test_meta, test_y, tokenizer, Config.MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=Config.VALID_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=Config.VALID_BATCH_SIZE)
    
    # Train Model
    print("STEP 3: TRAIN MODEL")
    
    model = HybridDeBERTa(Config.MODEL_NAME, Config.NUM_CLASSES, Config.METADATA_DIM).to(Config.DEVICE)
    
    weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(Config.DEVICE))
    
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': Config.LEARNING_RATE_BERT},
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': Config.LEARNING_RATE_CLF}
    ])
    
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_acc, _, _ = eval_model(model, valid_loader, criterion, Config.DEVICE)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.1%}")
        print(f"Valid: Loss={val_loss:.4f}, Acc={val_acc:.1%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_rag.pt')
            print("Saved")
    
    # Evaluate
    print("STEP 4: EVALUATE")
    
    model.load_state_dict(torch.load('best_model_rag.pt'))
    explainer = ExplanationGenerator()
    
    true_labels, preds, explanations = eval_with_explanations(
        model, test_loader, test_df, test_ctx, explainer, Config.DEVICE, n=10
    )
    
    acc = accuracy_score(true_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', pos_label=0)
    
    print("RESULTS")
    print(f"Accuracy: {acc:.1%}")
    print(f"Precision: {prec:.1%}")
    print(f"Recall: {rec:.1%}")
    print(f"F1: {f1:.3f}")
    
    print("\n" + classification_report(true_labels, preds, target_names=['Unreliable', 'Reliable']))
    
    # Save
    with open('explanations.json', 'w') as f:
        json.dump(explanations, f, indent=2)
    
    with open('metrics.json', 'w') as f:
        json.dump({
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }, f, indent=2)
    
    print("Files: best_model_rag.pt, explanations.json, metrics.json")