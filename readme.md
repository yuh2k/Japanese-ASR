# Overview
We developed a Japanese language corpus for ASR from the Matthew, chapters 1 and 2 of Holy Bible.

# Corpus Composition and Collection
The original sound recordings are from Bible.is (bible.is), Japanese New Interconfessional Version. The corpus consists of 49 WAV audio files paired with their transcripts, totaling 443.377 seconds of spoken Japanese by a single male speaker. Initial processing involved manually segmenting the biblical text into sentences to maintain a length of 5-10 seconds per audio clip. Subsequent transcription of each audio segment ensured accurate text representation.

# Alignment and Processing
For alignment, we utilized the Montreal Forced Aligner with the Japanese MFA dictionary v2.0.0 and the Japanese MFA acoustic model v2_0_1a, which facilitated the creation of TextGrid files. This forced alignment ensured precise synchronization between audio and text, forming a reliable foundation for ASR development.

# Metadata Description
•	Language: Standard Japanese
•	Content: Matthew Chapter 1 and Chapter 2 of Bible
•	Speaker: One young male narrator
•	Duration: 443.377 seconds
•	Format: 49 WAV sound files, 49 TXT transcripts files

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
-	./sounds: Sound files 
-	./transcripts: Transcript files
-	./aligned: TextGrid files
