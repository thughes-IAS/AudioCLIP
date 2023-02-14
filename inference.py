import glob
import os
import librosa

from model import AudioCLIP
from utils.transforms import ToTensor1D

import numpy as np
import torch

class AudioCLIPInference(object):

    def __init__(self,  model_filename = 'AudioCLIP-Full-Training.pt'):
        torch.set_grad_enabled(False)
        self.aclp = AudioCLIP(pretrained=f'assets/{model_filename}')

    def obtain_embeddings(self, audio, labels):
        text = [[label] for label in labels]
        ((audio_features, _, _), _), _ = self.aclp(audio=audio)
        ((_, _, text_features), _), _ = self.aclp(text=text)
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)

        scale_audio_text = torch.clamp(self.aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        logits_audio_text = scale_audio_text * audio_features @ text_features.T
        return logits_audio_text

    def preprocess_audio(self, input_dir,  SAMPLE_RATE = 44100):

        audio_transforms = ToTensor1D()

        paths_to_audio = glob.glob(f'{input_dir}/*.wav')
        audio = list()
        for path_to_audio in paths_to_audio:
            track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

            spec = self.aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
            spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
            pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

            audio.append((track, pow_spec))

        audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
        return audio, paths_to_audio

    @staticmethod
    def score_inputs(logits_audio_text, paths_to_audio):
        print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')
        confidence = logits_audio_text.softmax(dim=1)
        for audio_idx in range(len(paths_to_audio)):
            conf_values, ids = confidence[audio_idx].topk(3)

            query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
            results = ', '.join([f'{LABELS[i]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

            print(query + results)

    def __call__(self, input_dir, labels):
        audio, paths_to_audio = self.preprocess_audio(input_dir)
        logits_audio_text = self.obtain_embeddings(audio, LABELS)
        self.score_inputs(logits_audio_text, paths_to_audio)


if __name__ == '__main__':

    self = AudioCLIPInference()
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-f','--input_dir')
    args = parser.parse_args()

    LABELS = ['dog', 'lightning', 'sneezing', 'alarm clock', 'car horn']
    self(args.input_dir,LABELS)

