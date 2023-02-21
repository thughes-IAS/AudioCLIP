import glob
import os
import librosa

from model import AudioCLIP
from utils.transforms import ToTensor1D
from utils.extract_audio import extract_audio

import numpy as np
import torch

class AudioCLIPInference(object):

    def __init__(self, labels,  model_filename = 'AudioCLIP-Full-Training.pt', verbose=True):
        self.labels = labels
        torch.set_grad_enabled(False)
        self.aclp = AudioCLIP(pretrained=f'assets/{model_filename}')

        if verbose:
            parameters = sum([x.numel() for x in self.aclp.parameters()])/(10**6)
            print(f'Parameter count: {parameters:.1f}M')

    def obtain_embeddings(self, audio):
        text = [[label] for label in self.labels]
        ((audio_features, _, _), _), _ = self.aclp(audio=audio)
        ((_, _, text_features), _), _ = self.aclp(text=text)
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)

        scale_audio_text = torch.clamp(self.aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        logits_audio_text = scale_audio_text * audio_features @ text_features.T
        return logits_audio_text

    def preprocess_audio(self, input_dir,  SAMPLE_RATE = 44100, verbose=True):

        audio_transforms = ToTensor1D()

        paths_to_audio = glob.glob(f'{input_dir}/*.wav')
        audio = list()
        for path_to_audio in paths_to_audio:
            track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

            spec = self.aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
            spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
            pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

            audio.append((track, pow_spec))


        tracks,_=zip(*audio)
        if verbose:
            print( [track.shape for track in tracks])

        # maxtrack =  max([track.shape[0] for  track in tracks])
        # import ipdb;ipdb.set_trace()

        transformed_audio = [audio_transforms(track.reshape(1, -1)) for track in tracks]
        maxtrack =  max([ta.shape[-1] for ta in transformed_audio])

        padded = [torch.nn.functional.pad(ta,(0,maxtrack-ta.shape[-1])) for ta in transformed_audio]
        if verbose:
            print( [track.shape for track in padded])

        # import ipdb;ipdb.set_trace()

        # transformed_audio = [audio_transforms(track.reshape(1, -1)) for track, _ in audio]
        # audio = torch.nn.utils.rnn.pad_sequence([x.T for x in transformed_audio]).T

        # if verbose:
            # print( [ta.shape for ta in transformed_audio])


        audio = torch.stack(padded)
        # audio = torch.stack(transformed_audio)
        # audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])


        if verbose:
            print(audio.shape)
        # audio = torch.stack(transformed_audio)


        return audio, paths_to_audio

    def score_inputs(self, logits_audio_text, paths_to_audio):
        print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')
        confidence = logits_audio_text.softmax(dim=1)
        for audio_idx in range(len(paths_to_audio)):
            conf_values, ids = confidence[audio_idx].topk(1)

            query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
            results = ', '.join([f'{self.labels[i]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

            print(query + results)

    def __call__(self, input_dir, verbose=True):
        audio_dir = extract_audio(input_dir)
        audio, paths_to_audio = self.preprocess_audio(audio_dir, verbose=verbose)
        logits_audio_text = self.obtain_embeddings(audio)
        self.score_inputs(logits_audio_text, paths_to_audio)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-f','--input_dir')
    args = parser.parse_args()

    labels = ['dog', 'lightning', 'sneezing', 'alarm clock', 'car horn']

    self = AudioCLIPInference(labels)
    self(args.input_dir)

