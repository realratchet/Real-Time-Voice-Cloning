import csv
import json
import mplite
import shutil
import librosa
import numpy as np
import soundfile as sf
from mplite import Task, TaskManager
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from synthesizer import audio
import xml.etree.ElementTree as ET
from encoder import inference as encoder
from synthesizer.hparams import synth_hparams
from encoder.audio import wav_to_mel_spectrogram
from encoder.params_data import partials_n_frames
from synthesizer.utils.symbols import _characters_lt

dir_root = Path("/media/ratchet/hdd/Dataset/liepa2-corpus")
dir_root_oz = Path("/media/ratchet/hdd/Dataset/voice-oz")


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


def process_audiofiles(dir_root):
    dir_inp = dir_root / "data"
    dir_out = dir_root / "metavoice"

    dir_out.mkdir(parents=True, exist_ok=True)

    pairs = list(get_pairs(dir_inp, dir_out))

    for pwav, peaf in tqdm(pairs):
        base_name = dir_out / pwav.with_suffix("").name
        base_name.mkdir(parents=True, exist_ok=True)

        xml_eaf = ET.parse(peaf)
        time_slots = {
            el.get("TIME_SLOT_ID"): int(el.get("TIME_VALUE")) / 1000
            for el in xml_eaf.find("TIME_ORDER")
        }

        xml_annotations = xml_eaf.find("TIER").findall("ANNOTATION")

        full_wav, sr = librosa.load(pwav, synth_hparams.sample_rate)
        
        for i, (text, ann_begin, ann_finish) in enumerate(filter(lambda x: x is not None, (process_annotation(a, time_slots) for a in xml_annotations))):
            start_idx, end_idx = [int(t * sr) for t in [ann_begin, ann_finish]]

            wav = full_wav[start_idx:end_idx]

            # Trim silence
            wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)

            # Skip utterances that are too short
            if len(wav) < synth_hparams.utterance_min_duration * synth_hparams.sample_rate:
                continue

            sf.write(base_name / f"{i}.wav", wav, sr)
            with open(base_name / f"{i}.txt", "w") as f:
                f.write(text)

def _process_synthesizer_cml_task(dir_wavs: Path, dir_mels: Path, dir_encoder: Path, lines):
    i = 0
    added_elements = []

    for pwav, transcript, phonemes, client, gender, age in lines:
        pwav: Path
        wav, _ = librosa.load(pwav, synth_hparams.sample_rate)

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

        basename = pwav.stem
        wav_fpath = dir_wavs / ("audio-" + basename + ".npy")
        mel_fpath = dir_mels / ("mels-" + basename + ".npy")
        enc_fpath = dir_encoder / ("enc-" + basename + ".npy")

        wav_name = f"{client}/{wav_fpath.name}"
        mel_name = f"{client}/{mel_fpath.name}"
        emb_name = f"embed-{basename}.npy"

        train_str = "|".join([wav_name, mel_name, emb_name, str(len(wav)), str(mel_frames), phonemes, client, gender, age])
        added_elements.append(train_str)

        np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
        np.save(wav_fpath, wav, allow_pickle=False)
        np.save(enc_fpath, enc_spectr, allow_pickle=False)

        i = i + 1

    if len(added_elements) < 10:
        shutil.rmtree(dir_wavs)
        shutil.rmtree(dir_mels)
        shutil.rmtree(dir_encoder)
        return []

    return added_elements

def process_synthesizer_cml(dir_root: Path):
    dir_inp = dir_root / "cml_compliant_liepa2"
    dir_out = dir_root / "cml_compliant_liepa2_result"
    dir_out_audio = dir_out / "audio"
    dir_out_mels = dir_out / "mels"
    dir_out_encoder = dir_out / "encoder"

    dir_out_audio.mkdir(parents=True, exist_ok=True)
    dir_out_mels.mkdir(parents=True, exist_ok=True)
    dir_out_encoder.mkdir(parents=True, exist_ok=True)

    path_train = dir_out / "train.txt"
    train_contents = []
    clients = {}
    total_lines = 0

    with open(dir_inp / "metadata_fixed_w_phonemes.csv", "r") as f:
        lines = (s.strip().split("|") for s in f.readlines()[1:])

        for pwav, transcript, phonemes, client, gender, age in lines:
            container: list
            container = clients[client] = clients.get(client, [])
            container.append((dir_inp / pwav, transcript, phonemes, client, gender, age))
            total_lines = total_lines + 1


    # with tqdm(desc="generating mels", total=total_lines) as pbar:
    if True:
        tasks: list[Task] = []
        for client, lines in tqdm(clients.items(), desc="collecting tasks"):
            # i = 0
            # added_elements = []

            dir_wavs: Path = dir_out_audio / client
            dir_mels: Path = dir_out_mels / client
            dir_encoder: Path = dir_out_encoder / client

            dir_wavs.mkdir(exist_ok=True, parents=True)
            dir_mels.mkdir(exist_ok=True, parents=True)
            dir_encoder.mkdir(exist_ok=True, parents=True)

            task = Task(_process_synthesizer_cml_task, dir_wavs, dir_mels, dir_encoder, lines)
            tasks.append(task)

        with TaskManager(12) as tm:
            res = tm.execute(tasks)
            train_contents = list(chain.from_iterable(res))

    #         for pwav, transcript, phonemes, client, gender, age in lines:
    #             pwav: Path
    #             wav, sr = librosa.load(pwav, synth_hparams.sample_rate)

    #             # Skip utterances that are too short
    #             if len(wav) < synth_hparams.utterance_min_duration * synth_hparams.sample_rate:
    #                 pbar.update(1)
    #                 continue

    #             mel_spectrogram = audio.melspectrogram(wav, synth_hparams).astype(np.float32)
    #             mel_frames = mel_spectrogram.shape[1]

    #             # Skip utterances that are too long
    #             if mel_frames > synth_hparams.max_mel_frames and synth_hparams.clip_mels_length:
    #                 pbar.update(1)
    #                 continue

    #             enc_spectr = wav_to_mel_spectrogram(wav)
    #             enc_frames = len(enc_spectr)

    #             if enc_frames < partials_n_frames:
    #                 pbar.update(1)
    #                 continue

    #             basename = pwav.stem
    #             wav_fpath = dir_wavs / ("audio-" + basename + ".npy")
    #             mel_fpath = dir_mels / ("mels-" + basename + ".npy")
    #             enc_fpath = dir_encoder / ("enc-" + basename + ".npy")

    #             wav_name = f"{client}/{wav_fpath.name}"
    #             mel_name = f"{client}/{mel_fpath.name}"
    #             emb_name = f"embed-{basename}.npy"

    #             train_str = "|".join([wav_name, mel_name, emb_name, str(len(wav)), str(mel_frames), phonemes, client, gender, age])
    #             added_elements.append(train_str)

    #             np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    #             np.save(wav_fpath, wav, allow_pickle=False)
    #             np.save(enc_fpath, enc_spectr, allow_pickle=False)

    #             i = i + 1
    #             pbar.update(1)

    #         if len(added_elements) < 10:
    #             shutil.rmtree(dir_wavs)
    #             shutil.rmtree(dir_mels)
    #             shutil.rmtree(dir_encoder)
    #             continue

    #     train_contents.extend(added_elements)

    with open(path_train, "w") as f:
        f.write("\n".join(train_contents))

def process_synthesizer(dir_root: Path):
    dir_inp = dir_root / "data"
    dir_out = dir_root / "result"
    dir_out_audio = dir_out / "audio"
    dir_out_mels = dir_out / "mels"
    dir_out_encoder = dir_out / "encoder"

    dir_out_audio.mkdir(parents=True, exist_ok=True)
    dir_out_mels.mkdir(parents=True, exist_ok=True)
    dir_out_encoder.mkdir(parents=True, exist_ok=True)

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

def process_encoder_cml(dir_root: Path):
    dir_out = dir_root / "cml_compliant_liepa2_result"
    dir_out_encoder = dir_out / "encoder"
    dir_out_encoder.mkdir(parents=True, exist_ok=True)

    dirs = list(filter(lambda d: d.is_dir(), (x for x in dir_out_encoder.glob("*"))))

    for dir in tqdm(dirs):
        sources = [f"{mel.name},{(dir/'../../audio'/dir.name/mel.name.replace('enc-', 'audio-')).resolve()}" for mel in dir.glob("*.npy")]

        with open(dir/"_sources.txt", "w") as f:
            f.write("\n".join(sources))

def process_encoder(dir_root):
    dir_out = dir_root / "result"
    dir_out_encoder = dir_out / "encoder"
    dir_out_encoder.mkdir(parents=True, exist_ok=True)

    dirs = list(filter(lambda d: d.is_dir(), (x for x in dir_out_encoder.glob("*"))))

    for dir in tqdm(dirs):
        sources = [f"{mel.name},{(dir/'../../audio'/dir.name/mel.name.replace('enc-', 'audio-')).resolve()}" for mel in dir.glob("*.npy")]

        with open(dir/"_sources.txt", "w") as f:
            f.write("\n".join(sources))


def process_synthesizer_lsmu(dir_root):
    dir_inp = dir_root / "data"
    dir_out = dir_root / "result"
    dir_out_audio = dir_out / "audio"
    dir_out_mels = dir_out / "mels"
    dir_out_encoder = dir_out / "encoder"

    dir_out_audio.mkdir(parents=True, exist_ok=True)
    dir_out_mels.mkdir(parents=True, exist_ok=True)
    dir_out_encoder.mkdir(parents=True, exist_ok=True)

    pairs = list(((p, p.parent.name)for p in dir_inp.glob("**/*.wav")))

    path_train = dir_out / "train.txt"
    train_contents = []
    text = "Turėjo senelė žilą oželį."

    for pwav, pgroup in tqdm(pairs):
        wav, sr = librosa.load(pwav, synth_hparams.sample_rate)

        if synth_hparams.rescale:
            wav = wav / np.abs(wav).max() * synth_hparams.rescaling_max

        dir_wavs = dir_out_audio / pgroup
        dir_mels = dir_out_mels / pgroup
        dir_encoder = dir_out_encoder / pgroup

        dir_wavs.mkdir(exist_ok=True, parents=True)
        dir_mels.mkdir(exist_ok=True, parents=True)
        dir_encoder.mkdir(exist_ok=True, parents=True)

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

        basename = pwav.stem
        wav_fpath = dir_wavs / ("audio-" + basename + ".npy")
        mel_fpath = dir_mels / ("mels-" + basename + ".npy")
        enc_fpath = dir_encoder / ("enc-" + basename + ".npy")

        wav_name = f"{pgroup}/{wav_fpath.name}"
        mel_name = f"{pgroup}/{mel_fpath.name}"
        emb_name = f"embed-{basename}.npy"

        train_str = "|".join([wav_name, mel_name, emb_name, str(len(wav)), str(mel_frames), text])
        train_contents.append(train_str)

        np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
        np.save(wav_fpath, wav, allow_pickle=False)
        np.save(enc_fpath, enc_spectr, allow_pickle=False)

    with open(path_train, "w") as f:
        f.write("\n".join(train_contents))


def process_audio_wav(dir_out: Path, recording: dict):
    name = recording["name"].replace(".eaf", "")
    pwav = recording["media"]["path"]
    files = []
    
    if not name.startswith("X"):
        speakers = {name: recording["speech"]}
    else:
        speakers = recording["speech"]

    for speaker, values in speakers.items():
        _, _, (gender, age), speaker_id, _ = speaker.split("_")

        base_dir = dir_out / name / speaker
        base_dir.mkdir(parents=True, exist_ok=True)
        
        full_wav, sr = librosa.load(dir_root / pwav, 24000)

        for i, ann in enumerate(values):
            try:
                ann_begin, ann_finish = ann["beg"] / 1000, ann["end"] / 1000
                start_idx, end_idx = [int(t * sr) for t in [ann_begin, ann_finish]]
                wav = full_wav[start_idx:end_idx]
                transcript = ann["val"]

                sf.write(base_dir / f"{i}.wav", wav, sr)
                with open(base_dir / f"{i}.txt", "w") as f:
                    f.write(transcript)

                files.append([f"{name}/{speaker}/{i}.wav", transcript, speaker_id, gender, age])
            except:
                continue

    return files

def process_audio_json(dir_root):
    with open(dir_root / "etc" / "corpus-data.json", "r") as f:
        data = json.load(f)

    dir_out = dir_root / "cml_compliant_liepa2"

    dir_out.mkdir(parents=True, exist_ok=True)

    with mplite.TaskManager(error_mode='exception') as tm:
        tasks = [mplite.Task(process_audio_wav, dir_out, recording) for recording in data.values()]
        tm.execute(tasks)

        # files = list(chain.from_iterable(res))

    # files = []

    # for recording in tqdm(data.values()):
    #     name = recording["name"].replace(".eaf", "")
    #     pwav = recording["media"]["path"]
        
    #     if not name.startswith("X"):
    #         speakers = {name: recording["speech"]}
    #     else:
    #         speakers = recording["speech"]

    #     for speaker, values in speakers.items():
    #         _, _, (gender, age), speaker_id, _ = speaker.split("_")

    #         base_dir = dir_out / name / speaker
    #         base_dir.mkdir(parents=True, exist_ok=True)
            
    #         full_wav, sr = librosa.load(dir_root / pwav, 24000)

    #         for i, ann in enumerate(values):
    #             try:
    #                 ann_begin, ann_finish = ann["beg"] / 1000, ann["end"] / 1000
    #                 start_idx, end_idx = [int(t * sr) for t in [ann_begin, ann_finish]]
    #                 wav = full_wav[start_idx:end_idx]
    #                 transcript = ann["val"]

    #                 sf.write(base_dir / f"{i}.wav", wav, sr)
    #                 with open(base_dir / f"{i}.txt", "w") as f:
    #                     f.write(transcript)

    #                 files.append([f"{name}/{speaker}/{i}.wav", transcript, speaker_id, gender, age])
    #             except:
    #                 continue

    # with open(dir_out / "metadata.csv", "w") as f:
    #     writer = csv.writer(f, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(["wav_filename", "transcript", "client_id", "gender", "age"])
    #     writer.writerows(files)

if __name__ == "__main__":
    # process_audio_json(dir_root)
    # process_audiofiles(dir_root)
    # process_synthesizer(dir_root)
    # process_synthesizer_cml(dir_root)
    process_encoder_cml(dir_root)
    # process_encoder(dir_root)
    # process_synthesizer_lsmu(dir_root_oz)
    # process_encoder(dir_root_oz)