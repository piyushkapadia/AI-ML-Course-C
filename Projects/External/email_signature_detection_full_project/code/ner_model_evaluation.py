
import json
import re
import os
import random
from typing import List, Tuple
from sklearn.model_selection import train_test_split

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, load_metric

# Load tokenizer and base model
MODEL_NAME = "../models/dslim-bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3)  # PER, EML, O

LABELS = {"O": 0, "PER": 1, "EML": 2}
ID2LABEL = {v: k for k, v in LABELS.items()}


def extract_entities(text: str) -> List[Tuple[str, str]]:
    tokens = text.split()
    entities = []

    for token in tokens:
        if re.match(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", token):
            entities.append((token, "EML"))
        elif token.istitle():
            entities.append((token, "PER"))
        else:
            entities.append((token, "O"))
    return entities


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def prepare_dataset(clean_data_path: str):
    with open(clean_data_path, "r") as f:
        data = json.load(f)

    dataset = []
    for item in data:
        entities = extract_entities(item["content"])
        tokens = [t[0] for t in entities]
        ner_tags = [LABELS[t[1]] for t in entities]
        dataset.append({"tokens": tokens, "ner_tags": ner_tags})

    train_val, test = train_test_split(dataset, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)  # 60/20/20 split

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test)
    })

    return dataset_dict.map(tokenize_and_align_labels, batched=True)


def train_and_evaluate(dataset_dict):
    args = TrainingArguments(
        output_dir="./ner_output",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=10,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = torch.argmax(torch.tensor(predictions), axis=2)

        true_predictions = [
            [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate on test set
    predictions, labels, _ = trainer.predict(dataset_dict["test"])
    predictions = torch.argmax(torch.tensor(predictions), axis=2)

    test_results = []
    for i, (pred, label_ids) in enumerate(zip(predictions, labels)):
        tokens = dataset_dict["test"][i]["tokens"]
        final_labels = [ID2LABEL[l] for l in label_ids if l != -100]
        final_preds = [ID2LABEL[p] for p, l in zip(pred, label_ids) if l != -100]

        name = next((tok for tok, tag in zip(tokens, final_preds) if tag == "PER"), None)
        email = next((tok for tok, tag in zip(tokens, final_preds) if tag == "EML"), None)
        test_results.append({
            "message_id": i,
            "name": name,
            "email": email
        })

    with open("../data/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Count failures
    missing_name = sum(1 for r in test_results if not r["name"])
    missing_email = sum(1 for r in test_results if not r["email"])

    print(f"Missing name in {missing_name} messages")
    print(f"Missing email in {missing_email} messages")


if __name__ == "__main__":
    dataset_dict = prepare_dataset("../data/clean_data.json")
    train_and_evaluate(dataset_dict)
