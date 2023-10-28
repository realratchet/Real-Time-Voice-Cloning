import shutil
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from synthesizer import audio
import xml.etree.ElementTree as ET
from encoder import inference as encoder
from synthesizer.hparams import synth_hparams
from encoder.audio import wav_to_mel_spectrogram
from encoder.params_data import partials_n_frames
from synthesizer.utils.symbols import _characters_lt

dir_root = Path("/media/ratchet/hdd/Dataset/liepa2-corpus")

dir_inp = dir_root / "data"
dir_out = dir_root / "result"
dir_out_audio = dir_out / "audio"
dir_out_mels = dir_out / "mels"
dir_out_encoder = dir_out / "encoder"

dir_out_audio.mkdir(parents=True, exist_ok=True)
dir_out_mels.mkdir(parents=True, exist_ok=True)
dir_out_encoder.mkdir(parents=True, exist_ok=True)


def get_pairs(dir_inp, dir_out):
    eafs = set(dir_inp.glob("**/*.eaf"))

    for path_wav in dir_inp.glob("**/*.wav"):
        path_eaf = path_wav.with_suffix(".eaf")

        if path_eaf not in eafs:
            continue

        yield path_wav, path_eaf


def process_annotation(xml_annotation, time_slots):
    xml_ann_align = xml_annotation.find("ALIGNABLE_ANNOTATION")

    begin_time = time_slots[xml_ann_align.get("TIME_SLOT_REF1")]
    finish_time = time_slots[xml_ann_align.get("TIME_SLOT_REF2")]

    if (finish_time - begin_time) < synth_hparams.utterance_min_duration:
        return None

    xml_ann_value = xml_ann_align.find("ANNOTATION_VALUE")

    ann_value = xml_ann_value.text

    if not isinstance(ann_value, str) or ann_value.startswith("+") or ann_value.endswith("+"):
        return None

    ann_value = ann_value.strip().replace('„', '"').replace('–', "-")

    if any(ch not in _characters_lt for ch in ann_value):
        return None

    qmarks = sum(ch == '"' for ch in ann_value)

    if qmarks % 2 == 1:
        return None

    lparen = sum(ch == '(' for ch in ann_value)
    rparen = sum(ch == ')' for ch in ann_value)

    if lparen != rparen:
        return None

    return (ann_value, begin_time, finish_time)


def process_synthesizer():
    pairs = list(get_pairs(dir_inp, dir_out))

    path_train = dir_out / "train.txt"
    train_contents = []

    for pwav, peaf in tqdm(pairs):
        xml_eaf = ET.parse(peaf)
        time_slots = {
            el.get("TIME_SLOT_ID"): int(el.get("TIME_VALUE")) / 1000
            for el in xml_eaf.find("TIME_ORDER")
        }

        xml_annotations = xml_eaf.find("TIER").findall("ANNOTATION")

        full_wav, sr = librosa.load(pwav, synth_hparams.sample_rate)

        if synth_hparams.rescale:
            full_wav = full_wav / np.abs(full_wav).max() * synth_hparams.rescaling_max

        dir_wavs = dir_out_audio / pwav.stem
        dir_mels = dir_out_mels / pwav.stem
        dir_encoder = dir_out_encoder / pwav.stem

        dir_wavs.mkdir(exist_ok=True, parents=True)
        dir_mels.mkdir(exist_ok=True, parents=True)
        dir_encoder.mkdir(exist_ok=True, parents=True)

        i = 0
        added_elements = []

        for text, ann_begin, ann_finish in filter(lambda x: x is not None, (process_annotation(a, time_slots) for a in xml_annotations)):
            start_idx, end_idx = [int(t * sr) for t in [ann_begin, ann_finish]]

            wav = full_wav[start_idx:end_idx]

            # Trim silence
            if synth_hparams.trim_silence:
                wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)

            # Skip utterances that are too short
            if len(wav) < synth_hparams.utterance_min_duration * synth_hparams.sample_rate:
                continue

            mel_spectrogram = audio.melspectrogram(wav, synth_hparams).astype(np.float32)
            mel_frames = mel_spectrogram.shape[1]

            # Skip utterances that are too long
            if mel_frames > synth_hparams.max_mel_frames and synth_hparams.clip_mels_length:
                continue

            enc_spectr = wav_to_mel_spectrogram(wav)
            enc_frames = len(enc_spectr)

            if enc_frames < partials_n_frames:
                continue

            basename = f"{pwav.stem}_{i}"
            wav_fpath = dir_wavs / ("audio-" + basename + ".npy")
            mel_fpath = dir_mels / ("mels-" + basename + ".npy")
            enc_fpath = dir_encoder / ("enc-" + basename + ".npy")

            wav_name = f"{pwav.stem}/{wav_fpath.name}"
            mel_name = f"{pwav.stem}/{mel_fpath.name}"
            emb_name = f"embed-{basename}.npy"

            train_str = "|".join([wav_name, mel_name, emb_name, str(len(wav)), str(mel_frames), text])
            added_elements.append(train_str)

            np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
            np.save(wav_fpath, wav, allow_pickle=False)
            np.save(enc_fpath, enc_spectr, allow_pickle=False)

            i = i + 1

        if len(added_elements) < 10:
            shutil.rmtree(dir_wavs)
            shutil.rmtree(dir_mels)
            shutil.rmtree(dir_encoder)
            continue

        train_contents.extend(added_elements)

    with open(path_train, "w") as f:
        f.write("\n".join(train_contents))


def process_encoder():
    dirs = list(filter(lambda d: d.is_dir(), (x for x in dir_out_encoder.glob("*"))))

    for dir in tqdm(dirs):
        sources = [f"{mel.name},{(dir/'../../audio'/dir.name/mel.name.replace('enc-', 'audio-')).resolve()}" for mel in dir.glob("*.npy")]

        with open(dir/"_sources.txt", "w") as f:
            f.write("\n".join(sources))


process_synthesizer()
process_encoder()
