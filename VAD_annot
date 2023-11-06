import soundfile as sf
from auditok import split
from praatio import tgio

wav, sr = sf.read("220717_001_r.wav")
region = split("220717_001_r.wav", energy_threshold=70)
duration = len(wav)/sr
tg = tgio.Textgrid()
entryList = []

for elt in region:
    entry = tgio.Interval(elt.meta.start, elt.meta.end, "*")
    entryList.append(entry)

tier = tgio.IntervalTier(str, entryList, 0, duration)
tg.addTier(tier)
tg.save("220717_001.TextGrid")
