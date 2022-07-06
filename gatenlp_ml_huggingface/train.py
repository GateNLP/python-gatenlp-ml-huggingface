"""
Module for finetuning transformer models with data exported from GateNLP.
"""
import os
import json
from typing import Optional
from datasets import Dataset, Features, Value, Sequence, ClassLabel
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import PreTrainedModel

class TextClassificationTrainer:
    def __init__(self, data_dir):
        assert os.path.isdir(data_dir)
        self.data_dir = data_dir


class TokenClassificationTrainer:
    def __init__(
            self,
            data_dir,
            output_dir = "./tokenclassification_model",
            transformer_model="bert-base",
            train_size=0.9,
            eval_size=0.1,
            shuffle=True,
            seed=42,
            learning_rate=2e-5,
            max_epochs=20,
            batch_size=16,
    ):
        assert os.path.isdir(data_dir)
        self.data_dir = data_dir
        with open(os.path.join(self.data_dir, "features.json"), "rt", encoding="utf-8") as infp:
            self.data_features = json.load(infp)
        self.train_size = train_size
        self.eval_size = eval_size
        self.shuffle = shuffle
        self.seed = seed
        self.transformer_model = transformer_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.model: Optional[PreTrainedModel] = None
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.training_file = os.path.join(data_dir, "data.json")
        self.output_dir = output_dir
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            # evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=self.max_epochs,
            weight_decay=0.01,
        )
        self.trainer = None


    def train(self, train_size=None, eval_size=None, shuffle=None, seed=None):
        # create the features: for now we always have id, tokens, labels
        if train_size is None:
            train_size = self.train_size
        if eval_size is None:
            eval_size = self.eval_size
        if shuffle is None:
            shuffle = self.shuffle
        if seed is None:
            seed = self.seed
        features = Features(
            dict(
                id=Value(dtype="string"),
                tokens=Sequence(feature=Value(dtype="string")),
                labels=Sequence(feature=ClassLabel(
                    names=self.data_features["labels"])),
            )
        )
        # convert the training file to a dataset we can use for training:
        # first read the file and check its size
        ds = Dataset.from_json(self.training_file, features=features)
        dsdict = ds.train_test_split(test_size=eval_size, train_size=train_size, shuffle=shuffle, seed=seed)

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_dsdict = dsdict.map(tokenize_and_align_labels, batched=True, batch_size=self.batch_size)

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.transformer_model,
            num_labels=dsdict["train"].features["labels"].feature.num_classes)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dsdict["train"],
            eval_dataset=dsdict["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        self.trainer.train()
        self.model.save_pretrained(self.output_dir)
