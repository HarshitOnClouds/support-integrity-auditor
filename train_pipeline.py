import pandas as pd
import numpy as np
import os
import re
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import torch

def setup_directories():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/sia_classifier', exist_ok=True)
    os.makedirs('outputs/dossiers', exist_ok=True)

def preprocess_data(df):
    print("Preprocessing data...")
    
    REQUIRED_COLUMNS = ['Ticket_Subject', 'Ticket_Description', 'Priority_Level', 
                        'Ticket_Channel', 'Resolution_Time_Hours', 'Customer_Email', 'Issue_Category', 'Ticket_ID']
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {list(df.columns)}")
        
    df.fillna('', inplace=True)
    
    def get_tier(email):
        if not email: return 0
        domain = email.split('@')[-1].lower()
        free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
        if domain in free_domains:
            return 1
        return 2
        
    df['customer_tier'] = df['Customer_Email'].apply(get_tier)
    
    res_time_col = 'Resolution_Time_Hours'
    if res_time_col in df.columns:
        df[res_time_col] = pd.to_numeric(df[res_time_col], errors='coerce').fillna(0)
        df['resolution_time'] = df[res_time_col]
        df['resolution_time_log'] = np.log1p(df['resolution_time'])
    else:
        df['resolution_time'] = 0.0
        df['resolution_time_log'] = 0.0
        
    df['combined_text'] = df['Ticket_Subject'].astype(str) + ". " + df['Ticket_Description'].astype(str)
    
    df['channel'] = df['Ticket_Channel'].astype(str) if 'Ticket_Channel' in df.columns else 'unknown'
    df['ticket_type'] = df['Issue_Category'].astype(str) if 'Issue_Category' in df.columns else 'unknown'
    df['assigned_priority'] = df['Priority_Level'].astype(str) if 'Priority_Level' in df.columns else 'Low'
    df['ticket_id'] = df['Ticket_ID'] if 'Ticket_ID' in df.columns else df.index.astype(str)
    
    return df

def generate_pseudo_labels(df):
    print("Generating Pseudo Labels...")
    
    high_urgency = ['outage', 'down', 'broken', 'can\'t access', 'cannot access', 'urgent', 'asap', 
                    'critical', 'data loss', 'not working', 'unresponsive', 'failed', 'error', 
                    'immediately', 'escalate', 'severe', 'system down', 'production issue']
    escalation = ['affecting all users', 'entire team', 'multiple clients', 'revenue impact', 
                  'sla breach', 'legal', 'compliance']
    low_urgency = ['when you get a chance', 'minor', 'cosmetic', 'suggestion', 'feature request', 
                   'low priority', 'no rush', 'informational']
    negation = ['not urgent', 'not critical']
    
    def score_text(text):
        text = text.lower()
        score = 0
        is_negated = any(n in text for n in negation)
        for w in high_urgency:
            if w in text: score += 1
        for w in escalation:
            if w in text: score += 1.5
        for w in low_urgency:
            if w in text: score -= 1
        if is_negated:
            score = -score
        return score

    df['raw_signal_a'] = df['combined_text'].apply(score_text)
    min_a, max_a = df['raw_signal_a'].min(), df['raw_signal_a'].max()
    df['signal_a_score'] = (df['raw_signal_a'] - min_a) / (max_a - min_a) if max_a > min_a else 0.5
        
    print("Computing Signal B (Resolution Regression)...")
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X_text = tfidf.fit_transform(df['combined_text']).toarray()
    
    channel_dummies = pd.get_dummies(df['channel'], prefix='ch').values
    type_dummies = pd.get_dummies(df['ticket_type'], prefix='type').values
    tier_vals = df['customer_tier'].values.reshape(-1, 1)
    
    X_b = np.hstack([X_text, channel_dummies, type_dummies, tier_vals])
    y_b = df['resolution_time_log'].values
    
    from sklearn.model_selection import cross_val_predict
    gbr = GradientBoostingRegressor(n_estimators=50, random_state=42)
    preds = cross_val_predict(gbr, X_b, y_b, cv=5)
    
    min_b, max_b = preds.min(), preds.max()
    df['signal_b_score'] = (preds - min_b) / (max_b - min_b) if max_b > min_b else 0.5
        
    df['inferred_severity'] = 0.5 * df['signal_a_score'] + 0.5 * df['signal_b_score']
    
    def get_assigned_binary(priority):
        return 1 if str(priority).strip().lower() in ['high', 'critical'] else 0
        
    df['assigned_binary'] = df['assigned_priority'].apply(get_assigned_binary)
    
    best_threshold = 0.5
    best_mismatch_rate = None
    for thresh in np.arange(0.3, 0.9, 0.05):
        temp_inferred = (df['inferred_severity'] >= thresh).astype(int)
        mismatch_rate = (temp_inferred != df['assigned_binary']).mean()
        if 0.15 <= mismatch_rate <= 0.35:
            best_threshold = thresh
            best_mismatch_rate = mismatch_rate
            break
            
    if best_mismatch_rate is None:
        best_mismatch_rate = ((df['inferred_severity'] >= best_threshold).astype(int) != df['assigned_binary']).mean()
        print(f"[WARNING] No threshold hit 15-35% mismatch range. Defaulting to {best_threshold}")
        
    print(f"Chosen severity threshold: {best_threshold:.2f} (Mismatch Rate: {best_mismatch_rate:.2%})")
    
    df['inferred_label'] = (df['inferred_severity'] >= best_threshold).astype(int)
    df['mismatch_label'] = (df['inferred_label'] != df['assigned_binary']).astype(int)
    
    def get_mismatch_type(row):
        if row['mismatch_label'] == 1:
            return 'Hidden Crisis' if row['assigned_binary'] == 0 else 'False Alarm'
        return 'Consistent'
        
    df['mismatch_type'] = df.apply(get_mismatch_type, axis=1)
    
    print(f"Mismatch Distribution:\n{df['mismatch_type'].value_counts(normalize=True)}")
    return df

def train_classifier(df):
    print("Training Classifier...")
    
    df['input_text'] = "[Channel: " + df['channel'] + "] [Tier: " + df['customer_tier'].astype(str) + "] [Type: " + df['ticket_type'] + "] " + df['combined_text']
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['mismatch_label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['mismatch_label'])
    
    KEEP_COLS = ['input_text', 'inferred_label']
    
    train_dataset = Dataset.from_pandas(train_df[KEEP_COLS].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[KEEP_COLS].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[KEEP_COLS].reset_index(drop=True))
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_func(examples):
        return tokenizer(examples['input_text'], padding='max_length', truncation=True, max_length=256)
        
    train_dataset = train_dataset.map(tokenize_func, batched=True).rename_column("inferred_label", "labels")
    val_dataset = val_dataset.map(tokenize_func, batched=True).rename_column("inferred_label", "labels")
    test_dataset = test_dataset.map(tokenize_func, batched=True).rename_column("inferred_label", "labels")
    
    labels = train_df['inferred_label'].values
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    from torch import nn
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device).to(logits.dtype))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
            
    training_args = TrainingArguments(
        output_dir='./models/sia_classifier',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,
        logging_steps=10,
        report_to="none"
    )
    
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='macro'),
            'recall_0': recall_score(labels, predictions, pos_label=0),
            'recall_1': recall_score(labels, predictions, pos_label=1)
        }
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    trainer.train()
    
    print("\n--- TEST SET METRICS (MISMATCH DETECTION) ---")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    pred_inferred = np.argmax(logits, axis=-1)
    
    assigned_binary = test_df['assigned_binary'].values
    true_mismatch = test_df['mismatch_label'].values
    pred_mismatch = (pred_inferred != assigned_binary).astype(int)
    
    acc = accuracy_score(true_mismatch, pred_mismatch)
    f1 = f1_score(true_mismatch, pred_mismatch, average='macro')
    rec_0 = recall_score(true_mismatch, pred_mismatch, pos_label=0)
    rec_1 = recall_score(true_mismatch, pred_mismatch, pos_label=1)
    
    print(f"Mismatch Accuracy: {acc:.4f}")
    print(f"Mismatch Macro F1: {f1:.4f}")
    print(f"Mismatch Recall (Consistent): {rec_0:.4f}")
    print(f"Mismatch Recall (Hidden Crisis / False Alarm): {rec_1:.4f}")
    
    if f1 < 0.82:
        print("\n[WARNING] DistilBERT Macro F1 is below 0.82. You may want to switch back to 'microsoft/deberta-v3-small' and ensure fp16=False.")
    
    trainer.save_model('./models/sia_classifier')
    tokenizer.save_pretrained('./models/sia_classifier')
    print("Model saved to ./models/sia_classifier")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/customer_support_tickets.csv')
    args = parser.parse_args()
    
    setup_directories()
    
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    df = preprocess_data(df)
    df.to_csv('data/cleaned_tickets.csv', index=False)
    
    df = generate_pseudo_labels(df)
    df.to_csv('data/pseudo_labeled.csv', index=False)
    
    train_classifier(df)

if __name__ == "__main__":
    main()
