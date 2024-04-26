# Imports
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import f1_score
from transformers import EvalPrediction
import pandas as pd


def preprocess_function(samples, tokenizer):
    """
    Tokenization and preprocessing function.

    Args:
        samples (dict): A dictionary containing the input samples.
        tokenizer: The tokenizer object.

    Returns:
        dict: A dictionary containing the tokenized samples.
    """
    return tokenizer(samples["text"], padding="longest", truncation=True)


def compute_metrics(eval_pred: EvalPrediction):
    """
    Custom callback function to compute and log F1 score.

    Args:
        eval_pred (EvalPrediction): The evaluation prediction object.

    Returns:
        dict: Dictionary containing the computed F1 score.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="macro")
    return {"f1": f1}


def main():
    """
    Main function for training and evaluating the model.
    """
    # Set up model and dataset
    model_checkpoint = "roberta-large"
    batch_size = 16
    num_train_epochs = 3

    dataset = load_dataset("rotten_tomatoes")
    label2id = {lab: i for i, lab in enumerate(dataset["train"].features["label"].names)}

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=2, fn_kwargs={'tokenizer': tokenizer})

    # Model configuration
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=len(label2id), id2label={i: lab for lab, i in label2id.items()})
    config.label2id = label2id

    # Instantiate the model
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        report_to=None,  # Do not report to any platform
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model="eval_f1",
        logging_dir="./logs",
        logging_steps=10,  # Log every 10 steps
        save_steps=500,  # Save checkpoints every 500 steps
    )

    # Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Use custom metrics function
    )
    print(tokenized_dataset["train"])
    trainer.train()

    # Evaluate the model on test set
    predictions = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(predictions.predictions, axis=1)
    labels = tokenized_dataset["test"]["label"]
    f1 = f1_score(labels, predictions, average="macro")
    print("Test F1 Score:", f1)


if __name__ == "__main__":
    main()





