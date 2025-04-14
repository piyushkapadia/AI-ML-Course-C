
import json
import re

def load_emails(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def clean_text(text):
    return re.sub(r'[\r\n\t]', ' ', text)

def is_missing_signature(content):
    return '@' not in content

def has_multiple_emails(content):
    return len(re.findall(r'@', content)) > 1

def preprocess_emails(emails):
    clean_data, missing_signature, multiple_emails = [], [], []

    for email in emails:
        content = clean_text(email["content"])
        if is_missing_signature(content):
            missing_signature.append(email)
        elif has_multiple_emails(content):
            multiple_emails.append(email)
        else:
            clean_data.append({
                "message_id": email["message_id"],
                "content": content
            })

    return clean_data, missing_signature, multiple_emails

if __name__ == "__main__":
    emails = load_emails("../data/sample_emails.json")
    clean_data, missing_signature, multiple_emails = preprocess_emails(emails)

    with open("../data/clean_data.json", "w") as f:
        json.dump(clean_data, f, indent=2)

    with open("../data/missing_signature.json", "w") as f:
        json.dump(missing_signature, f, indent=2)

    with open("../data/multiple_emails.json", "w") as f:
        json.dump(multiple_emails, f, indent=2)

    print(f"Clean data count: {len(clean_data)}")
    print(f"Missing signature count: {len(missing_signature)}")
    print(f"Multiple emails count: {len(multiple_emails)}")
