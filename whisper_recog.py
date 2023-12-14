import whisper
import pandas as pd

model = whisper.load_model("base")
train_data = pd.read_csv('train_dataset.csv')

transcriptions = []

for index, row in train_data.iterrows(): 
    audio_path = row['audio_filepath']
    result = model.transcribe(audio_path)
    transcriptions.append(result["text"])

train_data['transcription'] = transcriptions
train_data.to_csv('train_transcriptions.csv', index=False)