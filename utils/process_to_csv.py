import os
import pandas as pd

audio_dir = 'sounds'
transcript_dir = 'transcripts'

dataframe = pd.DataFrame(columns=['audio_filepath', 'transcript'])

for audio_file in os.listdir(audio_dir):
    if audio_file.endswith('.wav'):
        transcript_file = audio_file.replace('.wav', '.txt')

        transcript_path = os.path.join(transcript_dir, transcript_file)
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()

            dataframe = pd.concat([dataframe, pd.DataFrame({'audio_filepath': [os.path.join(audio_dir, audio_file)], 'transcript': [transcript]})], ignore_index=True)

dataframe.to_csv('corpus.csv', index=False)
