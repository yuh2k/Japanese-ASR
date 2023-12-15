


from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import os

# Set file paths and token
audio_folder_path = './jpcorpus/clips'
TOKEN = "your_token_here"

# Load and process the dataset
common_voice = DatasetDict()
common_voice["train"] = load_dataset("jpcorpus", "hi", split="train+validation", token=TOKEN)
common_voice["test"] = load_dataset("jpcorpus", "hi", split="test", token=TOKEN)
common_voice = common_voice.remove_columns(["up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"])

# Update file paths and cast audio column
common_voice = common_voice.map(lambda example: {'path': os.path.join(audio_folder_path, example['path'])})
common_voice = common_voice.cast_column("path", Audio(sampling_rate=16000))

# Initialize feature extractor, tokenizer, and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Japanese", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Japanese", task="transcribe")

# Define dataset preparation function
def prepare_dataset(batch):
    audio = batch["path"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

# Data collator class
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Initialize metric, model, and training arguments
metric = evaluate.load("wer")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

training_args = Seq2SeqTrainingArguments(
    output_dir="./hyusername/jpasr",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    # fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

# Initialize and start training
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=lambda pred: {"wer": 100 * metric.compute(predictions=tokenizer.batch_decode(pred.predictions, skip_special_tokens=True), references=tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True))}
)

trainer.train()
