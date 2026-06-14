import pandas as pd
import numpy as np
import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm

def score_text(text):
    text = text.lower()
    high_urgency = ['outage', 'down', 'broken', 'can\'t access', 'cannot access', 'urgent', 'asap', 
                    'critical', 'data loss', 'not working', 'unresponsive', 'failed', 'error', 
                    'immediately', 'escalate', 'severe', 'system down', 'production issue']
    escalation = ['affecting all users', 'entire team', 'multiple clients', 'revenue impact', 
                  'sla breach', 'legal', 'compliance']
    low_urgency = ['when you get a chance', 'minor', 'cosmetic', 'suggestion', 'feature request', 
                   'low priority', 'no rush', 'informational']
    negation = ['not urgent', 'not critical']
    
    score = 0
    is_negated = any(n in text for n in negation)
    matched_features = []
    
    for w in high_urgency:
        if w in text: 
            score += 1
            matched_features.append({"signal": "keyword", "value": w, "weight": "1.0"})
    for w in escalation:
        if w in text: 
            score += 1.5
            matched_features.append({"signal": "keyword", "value": w, "weight": "1.5"})
    for w in low_urgency:
        if w in text: 
            score -= 1
            matched_features.append({"signal": "keyword", "value": w, "weight": "-1.0"})
            
    if is_negated:
        score = -score
        
    return score, matched_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'dossiers'), exist_ok=True)
    
    df = pd.read_csv(args.input)
    df.fillna('', inplace=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained('./models/sia_classifier')
    model = AutoModelForSequenceClassification.from_pretrained('./models/sia_classifier').to(device)
    model.eval()
    
    predictions = []
    
    print(f"Processing {len(df)} tickets...")
    batch_size = 32
    
    for i in tqdm(range(0, len(df), batch_size), desc="Analyzing Tickets"):
        batch_df = df.iloc[i:i+batch_size]
        
        input_texts = []
        for idx, row in batch_df.iterrows():
            combined_text = str(row.get('Ticket_Subject', '')) + ". " + str(row.get('Ticket_Description', ''))
            channel = str(row.get('Ticket_Channel', 'unknown'))
            ticket_type = str(row.get('Issue_Category', 'unknown'))
            email = str(row.get('Customer_Email', ''))
            domain = email.split('@')[-1].lower() if '@' in email else ''
            customer_tier = 1 if domain in ['gmail.com', 'yahoo.com', 'hotmail.com'] else 2 if domain else 0
            priority = str(row.get('Priority_Level', 'Low'))
            input_texts.append(f"[Channel: {channel}] [Tier: {customer_tier}] [Type: {ticket_type}] {combined_text}")
            
        inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        for local_idx, (idx, row) in enumerate(batch_df.iterrows()):
            confidence = probs[local_idx][1].item()
            pred_inferred = 1 if confidence >= 0.5 else 0
            
            combined_text = str(row.get('Ticket_Subject', '')) + ". " + str(row.get('Ticket_Description', ''))
            raw_score, matched_features = score_text(combined_text)
            inferred_severity = confidence
            
            priority = str(row.get('Priority_Level', 'Low'))
            ticket_id = str(row.get('Ticket_ID', str(idx)))
            res_time = row.get('Resolution_Time_Hours', 0)
            assigned_binary = 1 if priority.lower() in ['high', 'critical'] else 0
            
            pred_label = 1 if pred_inferred != assigned_binary else 0
            
            if pred_label == 1:
                mismatch_type = 'Hidden Crisis' if assigned_binary == 0 else 'False Alarm'
            else:
                mismatch_type = 'Consistent'
                
            predictions.append({
                'ticket_id': ticket_id,
                'assigned_priority': priority,
                'mismatch_label': pred_label,
                'confidence': confidence,
                'mismatch_type': mismatch_type,
                'inferred_severity': confidence
            })
            
            if pred_label == 1:
                evidence = matched_features[:2]
                if not evidence:
                    evidence.append({"signal": "keyword", "value": combined_text.split()[0] if combined_text else "", "weight": "0.0"})
                    
                evidence.append({
                    "signal": "resolution_time",
                    "value": str(res_time),
                    "interpretation": f"Resolution time is {res_time}"
                })
                
                dossier = {
                  "ticket_id": ticket_id,
                  "assigned_priority": priority,
                  "inferred_severity": round(inferred_severity, 2),
                  "mismatch_type": mismatch_type,
                  "severity_delta": f"Assigned={priority}, Inferred={'High' if inferred_severity>=0.5 else 'Low'}",
                  "feature_evidence": evidence,
                  "constraint_analysis": f"The ticket exhibits characteristics ({evidence[0]['value'] if evidence else 'various'}) that contradict its assigned {priority} priority.",
                  "confidence": round(confidence, 4)
                }
                
                with open(os.path.join(args.output, 'dossiers', f"{ticket_id}.json"), 'w') as f:
                    json.dump(dossier, f, indent=2)
                
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(os.path.join(args.output, 'predictions.csv'), index=False)
    print(f"Processed {len(pred_df)} tickets. Flagged: {pred_df['mismatch_label'].sum()}")

if __name__ == "__main__":
    main()
