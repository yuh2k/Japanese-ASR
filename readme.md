# Japanese Recognition in ASR Based on Whisper Medium Model and the Importance of Text Conversion Process in Multilingual Application
Authors: Yuanhuan Deng (yuanhuandeng@brandeis.edu) & Hang Yu (yuh@brandeis.edu)

## Overview

This project aims to enhance speech recognition capabilities within the Japanese language using the Whisper medium model. It focuses on the importance of text conversion in ASR systems, particularly for languages with complex scripts like Japanese. Our research outlines the challenges of multilingual ASR application and proposes solutions to improve transcription accuracy.

## How to Run this Programme

### Step 1: Installation and Dependencies

To install the necessary dependencies for this project, ensure you have Python and pip installed on your machine. Then, run the following command:
```
!pip install datasets>=2.6.1
!pip install git+https://github.com/huggingface/transformers
!pip install librosa
!pip install evaluate>=0.30
!pip install jiwer
!pip install gradio
```
### Step 2: Copy the Training Script
Copy the [train_1214.py](https://github.com/yuh2k/Japanese-ASR/blob/main/train_1214.py) script into your Google Colab notebook. You can do this by pasting the code directly into a cell or by using Google Colab's file upload feature.

### Step 3: Prepare the Dataset
Place your dataset in a directory on your Google Drive for easy access.

### Step 4: Mount Google Drive
To access the dataset from your Google Colab notebook, mount your Google Drive using the following command in a cell:
```from google.colab import drive
drive.mount('/content/drive')
```
Follow the prompts to authorize access to your Google Drive.

### Step 5: Adjust Parameters
Before running the training script, adjust the hyperparameters based on your machine's performance capabilities and your specific training needs. Edit the train_1214.py script accordingly.

### Step 6: Start Training
Execute the training script in a cell with the following command:

```
!python /content/drive/MyDrive/path_to_your_script/train_1214.py
```
Replace path_to_your_script with the actual path to the train_1214.py script in your Google Drive.

## Evaluation
We evaluate the model using the following metrics:

·**Training Loss:** Indicates how well the model is learning from the training data.
·**Validation Loss:** Assesses the model's ability to generalize to new data.
·**WER:** Measures the accuracy of the model's transcriptions.

## Language: Standard Japanese

## Conclusion
We conclude that incorporating a conversion step in the ASR process for languages like Japanese is essential. Despite the high WER, we identify potential improvement areas such as increased data and extended training steps.
