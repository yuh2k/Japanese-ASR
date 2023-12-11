# Overview

We developed a Japanese language corpus for ASR from the Matthew, chapters 1 and 2 of Holy Bible, and public datasets on Mozilla Common Voice.

# Corpus Composition and Collection

The original sound recordings are from Bible.is (bible.is), Japanese New Interconfessional Version, and public datasets on Mozilla Common Voice(https://commonvoice.mozilla.org, Common Voice is a crowdsourcing project started by Mozilla to create a free database for speech recognition software, supported by volunteers who record sample sentences with a microphone and review recordings of other users.) The corpus consists of 770 WAV audio files paired with their transcripts, totaling 3894.677 seconds of spoken Japanese by a single male speaker. Initial processing involved manually segmenting the biblical text into sentences to maintain a length of 2-10 seconds per audio clip. Subsequent transcription of each audio segment ensured accurate text representation.

# Alignment and Processing

For alignment, we utilized the Montreal Forced Aligner with the Japanese MFA dictionary v2.0.0 and the Japanese MFA acoustic model v2_0_1a, which facilitated the creation of TextGrid files. This forced alignment ensured precise synchronization between audio and text, forming a reliable foundation for ASR development.

# Metadata Description

## Language: Standard Japanese

## Content:

  -  Matthew Chapter 1 and Chapter 2 of Bible
  -  Sound files from Mozilla Common Voice
  -  Common Voice Delta Segment 11.0
  -  Common Voice Delta Segment 15.0

## Speaker: Multiple male speaker

  -  Bible narrator
  -  Characters appearing in Japanese TV news videos
  -  TV host/narrator
  -  Several Japanese sound contributors on the internet

## Duration: 

3894.677 secs (approximately 65min)

### Format: 

770 WAV sound files, 770 TXT transcripts files, 768 TextGrid Files

# Strengths and Weaknesses

## Strengths:

•	Japanese has a relatively small set of phonemes and a consistent phonetic system, making pronunciation predictable and recognition potentially more accurate. 
•	High-quality and consistent audio recordings.
•	The corpus is scalable because the sound files of Holy Bible are sufficient, we can enlarge it as needed.

## Weaknesses:

•	Limited vocabulary from biblical content.
•	Single speaker and gender representation, need more sounds of female narrator in the future.
•	There is an absence of modern colloquialism.
•	Japanese is not our mother language, manually matching the text files and sound files is really time consuming. 

# File Structure

-	sounds: Sound files 
-	transcripts: Transcript files
-	aligned: TextGrid files