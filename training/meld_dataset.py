import os
import cv2
import torch
import subprocess
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "sadness": 2,
            "joy": 3,
            "neutral": 4,
            "surprise": 5,
            "fear": 6,
        }

        self.sentiment_map = {
            "neutral": 0,
            "positive": 1,
            "negative": 2,
        }

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {video_path}")
            # try and read the first frame to validate the video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read video file {video_path}")

            # reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize the frame to 224x224
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  # normalize the frame
                frames.append(frame)

        except Exception as e:
            raise ValueError("Error loading video {video_path}: {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames found in video file {video_path}")

        # pad or truncate frames to 30
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        """Before permute: [frames, height, width, channels]

        Returns:
            after permute: [frames, channels, height, width]
        """
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    audio_path,
                ],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)

            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                win_length=400,
                hop_length=512,
                n_mels=64,
            )(waveform)

            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()  # Normalize

            if mel_spec.size(2) < 300:
                mel_spec = torch.nn.functional.pad(
                    mel_spec, (0, 300 - mel_spec.size(2)), "constant", 0
                )
            else:
                mel_spec = mel_spec[:, :, :300]
            return mel_spec
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error extracting audio from {video_path}: {str(e)}")
        except Exception as e:
            raise ValueError(
                f"Error extracting audio features from {video_path}: {str(e)}"
            )
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        try:
            video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir, video_filename)
            video_path = os.path.exists(path)

            if video_path == False:
                raise FileNotFoundError(
                    f"Video file {video_filename} not found in {self.video_dir}"
                )
            # print(f"Video file {video_filename} found in {self.video_dir}")
            text_inputs = self.tokenizer(
                row["Utterance"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)

            # map sentiment and emotion labels
            sentiment = self.sentiment_map[row["Sentiment"].lower()]
            emotion = self.emotion_map[row["Emotion"].lower()]

            return {
                "text_inputs": {
                    "input_ids": text_inputs["input_ids"].squeeze(),
                    "attention_mask": text_inputs["attention_mask"].squeeze(),
                },
                "video_frames": video_frames,
                "audio_features": audio_features,
                "emotion_labels": torch.tensor(emotion),
                "sentiment_labels": torch.tensor(sentiment),
            }
        except Exception as e:
            print(f"Error processing item {video_path}: {str(e)}")
            return None


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloader(csv_path, video_dir, batch_size=32, num_workers=4):
    dataset = MELDDataset(csv_path, video_dir)
    data_loder = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return data_loder


if __name__ == "__main__":
    dev_loader = prepare_dataloader(
        "../data/dev/dev_sent_emo.csv", "../data/dev/dev_splits_complete"
    )
    test_loader = prepare_dataloader(
        "../data/test/test_sent_emo.csv", "../data/test/output_repeated_splits_test"
    )
    train_loader = prepare_dataloader(
        "../data/train/train_sent_emo.csv", "../data/train/train_splits"
    )

    for batch in train_loader:
        print(batch["text_inputs"])
        print(batch["video_frames"].shape)
        print(batch["audio_features"].shape)
        print(batch["emotion_labels"])
        print(batch["sentiment_labels"])
        break
