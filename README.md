# Support Integrity Auditor (SIA)

The Support Integrity Auditor is a self-supervised machine learning pipeline designed to detect **Priority Mismatches** in CRM support tickets. Because no pre-annotated mismatch labels exist in raw ticket data, SIA bootstraps its own supervision signal (pseudo-labels) to isolate the *objective severity* of a ticket.

It then uses a fine-tuned Transformer model (DistilBERT) and a deterministic application-layer logic gate to compare the AI's objective severity against the human-assigned priority, reliably flagging **Hidden Crises** and **False Alarms**.

## System Architecture

```text
[ Raw Tickets ]
      |
      v
[ Self-Supervised Pseudo-Labeling Pipeline ]
      +---> [ Signal A: Rule-Based NLP (Urgency/Escalation) ]
      +---> [ Signal B: Res. Time Regression (TF-IDF + 5-Fold CV) ]
      |
      v
[ Objective Severity Label Generation ]
      |
      v
[ DistilBERT Sequence Classification Fine-Tuning ]
      |
      v
[ Inference Application Layer (Streamlit / CLI) ]
      +---> AI predicts 'Objective Severity' (0.0 to 1.0)
      +---> Deterministic XOR Gate compares AI vs Human Priority
      +---> Yields: Consistent ✅ | Hidden Crisis 🚨 | False Alarm ⚠️
```

## Key Engineering Features
- **Neural Network XOR Avoidance**: Directly training a neural network to learn an XOR logic gate on raw text leads to catastrophic representation collapse. SIA avoids this by training the model strictly on "Objective Severity" and moving the XOR comparison to a deterministic application layer.
- **Leakage-Free Pseudo-Labels**: Signal B uses `cross_val_predict(cv=5)` to ensure the regressor doesn't memorize resolution times, preventing target leakage during the self-supervised labeling phase.
- **Automated Threshold Scoping**: Dynamically sweeps severity thresholds (0.3 to 0.9) to anchor the optimal cutoff that yields a realistic mismatch distribution (15% - 35%).
- **Stratified Datasets**: Filters out noisy metadata and stratifies train/val/test splits strictly on the final mismatch label to ensure balanced evaluation loops.

## User Interface (Streamlit)
SIA includes a fully responsive frontend built in Streamlit (`app.py`), featuring:
1. **Live Single Ticket Inference**: Type in a mock ticket and watch the model load into memory to instantly analyze the text and output a severity confidence score.
2. **Batch Analysis Drag-and-Drop**: Upload a CSV of tickets (e.g., `batch_test.csv`) to process them in bulk, complete with dynamic spinning loaders and color-coded Pandas dataframe rendering.
3. **Analytics Dashboard**: Generates Plotly pie charts breaking down the distribution of Hidden Crises vs False Alarms across your processed data.

## Reproduction Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Pipeline (Colab Recommended)**
   Ensure `data/customer_support_tickets.csv` is in place.
   ```bash
   python train_pipeline.py --data data/customer_support_tickets.csv
   ```
3. **Run Batch Inference (CLI)**
   ```bash
   python predict.py --input batch_test.csv --output outputs/
   ```
4. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```
