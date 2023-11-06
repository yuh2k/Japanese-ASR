import speechbrain as sb

from speechbrain.pretrained import EncoderDecoderASR, EncoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation

import soundfile
import torchaudio

import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
def get_partitions(
    t: int = 100000,
    max_len_s: float = 1280.0,
    fs: int = 16000,
    samples_to_frames_ratio=512,
    overlap: int = 0,
):
    """Obtain partitions
    Note that this is implemented for frontends that discard trailing data.
    Note that the partitioning strongly depends on your architecture.
    A note on audio indices:
        Based on the ratio of audio sample points to lpz indices (here called
        frame), the start index of block N is:
        0 + N * samples_to_frames_ratio
        Due to the discarded trailing data, the end is then in the range of:
        [N * samples_to_frames_ratio - 1 .. (1+N) * samples_to_frames_ratio] ???
    """
    # max length should be ~ cut length + 25%
    cut_time_s = max_len_s / 1.25
    max_length = int(max_len_s * fs)
    cut_length = int(cut_time_s * fs)
    # make sure its a multiple of frame size
    max_length -= max_length % samples_to_frames_ratio
    cut_length -= cut_length % samples_to_frames_ratio
    overlap = int(max(0, overlap))
    if (max_length - cut_length) <= samples_to_frames_ratio * (2 + overlap):
        raise ValueError(
            f"Pick a larger time value for partitions. "
            f"time value: {max_len_s}, "
            f"overlap: {overlap}, "
            f"ratio: {samples_to_frames_ratio}."
        )
    partitions = []
    duplicate_frames = []
    cumulative_lpz_length = 0
    cut_length_lpz_frames = int(cut_length // samples_to_frames_ratio)
    partition_start = 0
    while t > max_length:
        start = int(max(0, partition_start - samples_to_frames_ratio * overlap))
        end = int(
            partition_start + cut_length + samples_to_frames_ratio * (1 + overlap) - 1
        )
        partitions += [(start, end)]
        # overlap - duplicate frames shall be deleted.
        cumulative_lpz_length += cut_length_lpz_frames
        for i in range(overlap):
            duplicate_frames += [
                cumulative_lpz_length - i,
                cumulative_lpz_length + (1 + i),
            ]
        # next partition
        t -= cut_length
        partition_start += cut_length
    else:
        start = int(max(0, partition_start - samples_to_frames_ratio * overlap))
        partitions += [(start, None)]
    partition_dict = {
        "partitions": partitions,
        "overlap": overlap,
        "delete_overlap_list": duplicate_frames,
        "samples_to_frames_ratio": samples_to_frames_ratio,
        "max_length": max_length,
        "cut_length": cut_length,
        "cut_time_s": cut_time_s,
    }
    return partition_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', help ='path to wav')
    parser.add_argument('--trans', help='path to the corresponding transcript')
    parser.add_argument('--out', help='path to the out file')
    args = parser.parse_args()
    audio_path = args.wav
    trans_path = args.trans
    out_path = args.out
    # Requires a model with CTC output
    #asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-fr")
    asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-fr")
    aligner = CTCSegmentation(asr_model, kaldi_style_text=False, gratis_blank=True)

    with open(trans_path, "r", encoding="utf-8") as input_file:
            text = [s.upper().replace('\n','') for s in input_file]

    info = torchaudio.info(audio_path)
    print(info)
    fs = 16000
    sig = sb.dataio.dataio.read_audio(audio_path)
    if info.sample_rate != fs:
        print(f"sample rate of audio clips different than expected {info.sample_rate} vs {fs}")
        print(f"RESAMPLING")
        resampled = torchaudio.transforms.Resample(
                info.sample_rate,
                fs,
            )(sig)
    else:
        resampled = sig 

    #print(resampled.shape)
    speech_len = resampled.shape[0]
    #print(speech_len)
    longest_audio_segments = 320 #default 
    samples_to_frames_ratio = aligner.estimate_samples_to_frames_ratio()
    #print(samples_to_frames_ratio)
    partitions_overlap_frames = 30 #default
    partitions = get_partitions(
                speech_len,
                max_len_s=longest_audio_segments,
                samples_to_frames_ratio=samples_to_frames_ratio,
                fs=fs,
                overlap=partitions_overlap_frames,
            )
    #print(partitions)

    lpzs = [
        torch.tensor(aligner.get_lpz(resampled[start:end]))
        for start, end in tqdm(partitions["partitions"])
    ]
    # concatenate audio vectors
    lpz = torch.cat(lpzs).numpy()
    # delete the overlaps
    lpz = np.delete(lpz, partitions["delete_overlap_list"], axis=0)

    task = aligner.prepare_segmentation_task(text, lpz, name="test1", speech_len=speech_len)
    segments = aligner.get_segments(task)
    #segments = aligner(audio_path, text, name="example1")
    task.set(**segments)
    #print(task)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w", encoding="utf-8") as out:
        out.write(str(task))
