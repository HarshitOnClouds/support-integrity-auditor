import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = './models/sia_classifier'
has_model = os.path.exists(model_path)

st.set_page_config(page_title="Support Integrity Auditor", layout="wide")
st.title("Support Integrity Auditor (SIA)")

if not has_model:
    st.warning("⚠️ Model not found in `./models/sia_classifier`. Please train the model on Google Colab first, download the folder, and place it in the project directory.")
else:
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model

    tokenizer, model = load_model()
    st.success("✅ Model loaded successfully!")

tab1, tab2, tab3 = st.tabs(["Analyze Ticket", "Batch Analysis", "Dashboard"])

with tab1:
    st.header("Analyze Single Ticket")
    with st.form("single_ticket_form"):
        subject = st.text_input("Ticket Subject")
        description = st.text_area("Ticket Description")
        priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])
        channel = st.selectbox("Channel", ["email", "chat", "phone", "social media"])
        res_time = st.number_input("Resolution Time (hours)", min_value=0.0)
        email = st.text_input("Customer Email")
        
        submitted = st.form_submit_button("Analyze")
        
        if submitted and has_model:
            combined_text = f"{subject}. {description}"
            domain = email.split('@')[-1].lower() if '@' in email else ''
            customer_tier = 1 if domain in ['gmail.com', 'yahoo.com', 'hotmail.com'] else 2 if domain else 0
            
            input_text = f"[Channel: {channel}] [Tier: {customer_tier}] [Type: unknown] {combined_text}"
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = probs[0][1].item()
                pred_inferred = 1 if confidence >= 0.5 else 0
                
            assigned_binary = 1 if priority in ['High', 'Critical'] else 0
            pred_label = 1 if pred_inferred != assigned_binary else 0
            
            if pred_label == 1:
                st.error(f"⚠️ PRIORITY MISMATCH DETECTED (Model Severity: {confidence:.2%})")
                mismatch_type = 'Hidden Crisis' if assigned_binary == 0 else 'False Alarm'
                st.write(f"**Type:** {mismatch_type}")
                
                dossier = {
                  "ticket_id": "manual_entry",
                  "assigned_priority": priority,
                  "mismatch_type": mismatch_type,
                  "confidence": round(confidence, 4)
                }
                st.json(dossier)
            else:
                st.success(f"✅ CONSISTENT (Model Severity: {confidence:.2%})")

with tab2:
    st.header("Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file and has_model:
        batch_df = pd.read_csv(uploaded_file)
        batch_df.fillna('', inplace=True)
        
        if len(batch_df) > 100:
            st.warning("UI batch processing is capped at 100 tickets for speed. Truncating.")
            batch_df = batch_df.head(100)
            
        with st.spinner(f"Analyzing {len(batch_df)} tickets via DistilBERT..."):
            input_texts = []
            for _, row in batch_df.iterrows():
                combined_text = str(row.get('Ticket_Subject', '')) + ". " + str(row.get('Ticket_Description', ''))
                channel = str(row.get('Ticket_Channel', 'unknown'))
                ticket_type = str(row.get('Issue_Category', 'unknown'))
                email = str(row.get('Customer_Email', ''))
                domain = email.split('@')[-1].lower() if '@' in email else ''
                customer_tier = 1 if domain in ['gmail.com', 'yahoo.com', 'hotmail.com'] else 2 if domain else 0
                input_texts.append(f"[Channel: {channel}] [Tier: {customer_tier}] [Type: {ticket_type}] {combined_text}")
                
            inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            results = []
            for idx, row in batch_df.iterrows():
                confidence = probs[idx][1].item()
                pred_inferred = 1 if confidence >= 0.5 else 0
                
                priority = str(row.get('Priority_Level', 'Low'))
                ticket_id = str(row.get('Ticket_ID', f"TKT-{idx}"))
                assigned_binary = 1 if priority.lower() in ['high', 'critical'] else 0
                
                pred_label = 1 if pred_inferred != assigned_binary else 0
                
                if pred_label == 1:
                    mismatch_type = '🚨 Hidden Crisis' if assigned_binary == 0 else '⚠️ False Alarm'
                else:
                    mismatch_type = '✅ Consistent'
                    
                results.append({
                    'Ticket ID': ticket_id,
                    'Subject': str(row.get('Ticket_Subject', '')),
                    'Assigned': priority,
                    'AI Verdict': mismatch_type,
                    'Severity': f"{confidence:.1%}"
                })
                
            res_df = pd.DataFrame(results)
            st.success("Batch Processing Complete!")
            st.dataframe(res_df, use_container_width=True)

with tab3:
    st.header("Dashboard")
    if os.path.exists('outputs/predictions.csv'):
        df_preds = pd.read_csv('outputs/predictions.csv')
        st.metric("Total Tickets Processed", len(df_preds))
        
        fig = px.pie(df_preds, names='mismatch_type', title='Mismatch Types')
        st.plotly_chart(fig)
    else:
        st.info("Run Batch Analysis first to generate dashboard data.")
