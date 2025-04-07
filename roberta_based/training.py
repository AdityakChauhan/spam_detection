import pandas as pd
# pip install transformers datasets scikit-learn
url = "https://raw.githubusercontent.com/AdityakChauhan/spam_detection/main/dataset/dataset.csv"
df = pd.read_csv(url)
df = df.rename(columns={'spam': 'label'})
df = df[['text', 'label']].fillna("").drop_duplicates()
df.to_csv("data.csv", index=False)
import pandas as pd
from datasets import Dataset

df = pd.read_csv("data.csv")
dataset = Dataset.from_pandas(df)

from transformers import AutoTokenizer

model_name = "roberta-base"  # or any other like "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Split dataset manually
train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_ds = train_test['train']
eval_ds = train_test['test']

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)
trainer.train()
trainer.evaluate()
