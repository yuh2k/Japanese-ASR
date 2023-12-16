import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 重新采样音频文件
def resample_audio(audio_path, target_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
        waveform = resampler(waveform)
    return waveform

# 数据集类
class ASRDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = row['audio_filepath']  # 确保这个字段匹配你的csv文件中的列名
        transcript = row['transcript']
        waveform = resample_audio(audio_path)

        # 处理音频
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        # 处理文本
        with processor.as_target_processor():
            labels = processor(transcript, return_tensors="pt").input_ids.squeeze()

        return inputs.input_values.squeeze(), labels

# 加载预训练模型和处理器
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cpu")

# 加载训练数据集
df = pd.read_csv("train_dataset.csv")  # 包含两列：'audio_filepath' 和 'transcript'
train_dataset = ASRDataset(df, processor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 可以调整batch_size大小

# 定义损失函数和优化器
loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id).to("cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 训练模型
model.train()
num_epochs = 3  # 或者你希望运行的epoch数
for epoch in range(num_epochs):
    for batch in train_loader:
        input_values, labels = batch
        input_values = input_values.to("cpu")
        labels = labels.to("cpu")

        # 正向传播
        logits = model(input_values).logits

        # 计算 input_lengths
        input_lengths = torch.full((input_values.size(0),), logits.size(1), dtype=torch.long)

        # 计算 target_lengths
        target_lengths = torch.tensor([label.numel() for label in labels], dtype=torch.long)

        # CTC Loss需要log softmax probabilities
        log_probs = logits.log_softmax(2)

        # 计算损失
        loss = loss_fn(log_probs.permute(1, 0, 2), labels, input_lengths, target_lengths)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")