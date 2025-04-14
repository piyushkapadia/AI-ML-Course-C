
# Email Signature Detection Project

## Overview
This project extracts **names** and **email addresses** from email signatures using a fine-tuned NER model (`dslim/bert-base-NER`). It includes preprocessing, edge case handling, and model training.

---

## 📁 Dataset Structure

- `sample_emails.json` - Raw email data.
- `clean_data.json` - Cleaned and filtered dataset.
- `missing_signature.json` - Emails without signatures.
- `multiple_emails.json` - Emails with multiple emails in the body.

Each item in the dataset:
```json
{
  "message_id": 1,
  "content": "Email body text"
}
```

---

## ⚙️ Preprocessing Tasks

1. Identify emails without any `@` character (i.e., no email present).
2. Detect emails with multiple `@` characters far apart.
3. Remove control characters like `\n`, `\r`, and `\t`.
4. Split the data into train/validation/test sets.

---

## 🧠 NER Model Customization

### Entity Types:
- `PER` - Person Name
- `EML` - Email Address (Regex-based)

### Steps:
1. Define custom regex rule to extract emails:  
   `\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b`
2. Fine-tune the `dslim/bert-base-NER` on cleaned data.
3. Evaluate on validation dataset.
4. Run inference on test dataset.

---

## 📊 Evaluation Output

Final predictions format:
```json
{
  "message_id": 3,
  "name": "Raj",
  "email": "raj@company.com"
}
```

Missing extraction is counted and summarized.

---

## ▶️ How to Run Preprocessing

```bash
cd code
python preprocess.py
```

---

## 📦 File Structure

```
email_signature_detection_project/
├── code/
│   └── preprocess.py
├── data/
│   ├── sample_emails.json
│   ├── clean_data.json
│   ├── missing_signature.json
│   ├── multiple_emails.json
└── README.md
```


---

## 🏷️ PER Label Annotation

In a real-world setup, you would annotate names (PER) using tools like Prodigy, doccano, or manual tagging.
For this demo, dummy PER labels were simulated.

---

## 📊 Model Evaluation

- **Metrics**: Precision, Recall, F1-score
- **Confusion Matrix** and **Performance Graphs** included.

---

## 📈 Outputs

- `classification_report.txt` - Text metrics
- `confusion_matrix.png` - Heatmap of confusion matrix
- `model_performance.png` - Graphs of loss and accuracy
