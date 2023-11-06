# simple command to resample and rechannel a sound file
ffmpeg -i /path/to/file.wav -ar 16000 -ac 1 /path/to/destination.wav

#try it on a loop (DO NOT put the same path in your input and output or it will overwrite everything)
for i in path/to/files/*.wav; do ffmpeg -i $i -ar 16000 -ac 1 path/to/destination/$i; done

#get file information
soxi path/to/file.wav

#get total duration of wav in a folder
soxi -D *.wav | awk '{SUM += $1} END { printf "%d:%d:%d\n",SUM/3600,SUM%3600/60,SUM%60}'
