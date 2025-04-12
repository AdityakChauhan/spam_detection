import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback, DataCollatorForSeq2Seq
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load your dataset ===
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Drop missing values
    df = df.dropna(subset=["text", "label"])

    # Convert numeric labels to strings for T5 (0 = ham, 1 = spam)
    # label_map = {0: "ham", 1: "spam"}
    # df["label"] = df["label"].map(label_map).astype(str)

    return Dataset.from_pandas(df)

# === Preprocessing ===
def preprocess(example, tokenizer):
    input_text = "classify: " + example["text"]
    target_text = str(example["label"])

    input_enc = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=256
    )
    target_enc = tokenizer(
        target_text,
        padding="max_length",
        truncation=True,
        max_length=10
    )

    # Replace pad token ids with -100 to ignore them in loss
    labels = target_enc["input_ids"]
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": labels
    }


# === Compute metrics ===
def compute_metrics(eval_pred):
    try:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"eval_f1": f1}
    except Exception as e:
        print(f"[⚠️ compute_metrics ERROR]: {e}")
        return {"eval_f1": 0.0}

# === Live plot callback ===

class LivePlotCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        plt.ion()  # Interactive mode ON

        self.losses = []
        self.steps = []

        self.fig, self.ax = plt.subplots(figsize=(18, 6))
        self.line1, = self.ax.plot([], [], label="Loss", color="blue")

        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.ax.grid(True)

        plt.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        loss = logs.get("loss", None)
        print(f"[DEBUG] Step: {step}, Loss: {loss}")

        if loss is not None:
            self.steps.append(step)
            self.losses.append(loss)
            self.line1.set_data(self.steps, self.losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def on_train_end(self, args, state, control, **kwargs):
        plt.ioff()
        plt.show()



# === Main training function ===
def train_and_evaluate(csv_path, model_name="google/flan-t5-small", epochs=3, batch_size=8):
    # Load dataset and tokenizer
    dataset = load_dataset(csv_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    def preprocess_wrapper(example):
        return preprocess(example, tokenizer)

    tokenized_dataset = dataset.map(preprocess_wrapper, remove_columns=["text", "label"])
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # # Preprocess
    # tokenized_dataset = dataset.map(lambda x: preprocess(x, tokenizer))

    # Train-test split
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_ds = split["train"]
    eval_ds = split["test"]

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=[LivePlotCallback],
    )

    print("🔧 Training started...")
    plt.ion()
    trainer.train()
    plt.ioff()
    plt.show()

    print("✅ Training completed.\n")

    print("🧪 Evaluating...")
    eval_output = trainer.predict(eval_ds)

    decoded_preds = tokenizer.batch_decode(eval_output.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(eval_output.label_ids, skip_special_tokens=True)

    acc = accuracy_score(decoded_labels, decoded_preds)
    prec = precision_score(decoded_labels, decoded_preds, pos_label="spam", average="binary")
    rec = recall_score(decoded_labels, decoded_preds, pos_label="spam", average="binary")
    f1 = f1_score(decoded_labels, decoded_preds, pos_label="spam", average="binary")
    cm = confusion_matrix(decoded_labels, decoded_preds, labels=["ham", "spam"])

    print("\n📊 Final Evaluation Metrics:")
    print(f"  Accuracy :  {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall   :  {rec:.4f}")
    print(f"  F1 Score :  {f1:.4f}")
    print("\n🧩 Confusion Matrix:\n", cm)


# === Example usage ===
if __name__ == "__main__":
    # Make sure your CSV has two columns: 'text' and 'label' (label: 0 for ham, 1 for spam)
    
    csv_path = ".\data.csv"
    train_and_evaluate(csv_path, epochs=1, batch_size=12)