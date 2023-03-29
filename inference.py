import glob
import os

import librosa
import numpy as np
import torch
from tqdm import tqdm

from model import AudioCLIP
from utils.extract_audio import extract_audio
from utils.transforms import ToTensor1D


class AudioCLIPInference(object):

    def __init__(self, labels, model_filename='Full', verbose=False, **kwargs):
        self.labels = labels
        torch.set_grad_enabled(False)
        self.aclp = AudioCLIP(
            pretrained=f'assets/AudioCLIP-{model_filename}-Training.pt')

        if verbose:
            parameters = sum([x.numel()
                              for x in self.aclp.parameters()]) / (10 ** 6)
            print(f'Parameter count: {parameters:.1f}M')

    def obtain_embeddings(self, audio, text_features):
        ((audio_features, _, _), _), _ = self.aclp(audio=audio)
        audio_features = audio_features / torch.linalg.norm(
            audio_features, dim=-1, keepdim=True)

        scale_audio_text = torch.clamp(self.aclp.logit_scale_at.exp(),
                                       min=1.0,
                                       max=100.0)

        logits_audio_text = scale_audio_text * audio_features @ text_features.T
        return logits_audio_text

    def preprocess_audio(self,
                         input_dir,
                         SAMPLE_RATE=44100,
                         verbose=False,
                         batch_size=1 << 6,
                         **kwargs):

        audio_transforms = ToTensor1D()

        paths_to_audio = glob.glob(f'{input_dir}/*.wav')
        audio_paths = []
        audio = list()

        for num, path_to_audio in tqdm(enumerate(paths_to_audio, start=1)):
            track, _ = librosa.load(path_to_audio,
                                    sr=SAMPLE_RATE,
                                    dtype=np.float32)

            spec = self.aclp.audio.spectrogram(
                torch.from_numpy(track.reshape(1, 1, -1)))
            spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
            pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
            audio.append((track, pow_spec))
            audio_paths.append(path_to_audio)

            # spec =  self.aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))

            # spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
            # pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

            # audio.append((track, pow_spec))

            if not num % batch_size:

                tracks, _ = zip(*audio)
                if verbose:
                    print([track.shape for track in tracks])

                audio = torch.stack([
                    audio_transforms(track.reshape(1, -1))
                    for track, _ in audio
                ])

                '''
                transformed_audio = [
                    audio_transforms(track.reshape(1, -1)) for track in tracks
                ]

                maxtrack = max([ta.shape[-1] for ta in transformed_audio])

                padded = [
                    torch.nn.functional.pad(ta, (0, maxtrack - ta.shape[-1]))
                    for ta in transformed_audio
                ]
                if verbose:
                    print([track.shape for track in padded])

                audio = torch.stack(padded)
                '''

                if verbose:
                    print(audio.shape)

                yield audio, audio_paths

                audio = []
                audio_paths = []

    def score_inputs(self, logits_audio_text, paths_to_audio, outfile=None):
        print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)',
              end='\n\n')
        confidence = logits_audio_text.softmax(dim=1)
        for audio_idx in range(len(paths_to_audio)):
            conf_values, ids = confidence[audio_idx].topk(1)

            query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
            results = ', '.join([
                f'{self.labels[i]:>15s} ({v:06.2%})'
                for v, i in zip(conf_values, ids)
            ])

            print(query + results)

            if outfile is not None:
                with open(outfile, 'a') as ofh:
                    ofh.write(
                        query.split(' ->')[0].lstrip() + ',' +
                        results.split(' (')[0].lstrip() + '\n')

    def __call__(self,
                 input_dir=None,
                 verbose=False,
                 outfile=None,
                 skip_audio_extraction=False,
                 **kwargs):

        if outfile is not None:
            with open(outfile, 'w') as ofh:
                pass

        text = [[label] for label in self.labels]
        ((_, _, text_features), _), _ = self.aclp(text=text)

        if skip_audio_extraction:
            audio_dir = input_dir
        else:
            audio_dir = extract_audio(input_dir, **kwargs)

        for audio, paths_to_audio in self.preprocess_audio(audio_dir,
                                                           verbose=verbose,
                                                           **kwargs):

            logits_audio_text = self.obtain_embeddings(audio, text_features)
            self.score_inputs(logits_audio_text,
                              paths_to_audio,
                              outfile=outfile)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=2,
                        help='batch size')
    parser.add_argument('-f', '--input_dir')
    parser.add_argument('-s',
                        '--skip_audio_extraction',
                        action='store_true',
                        help='Skip 5s audio chunking step')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='increase verbosity')
    parser.add_argument('-o', '--outfile', help='Output file')
    parser.add_argument('-n',
                        '--num',
                        type=int,
                        help='Limit to first N input files')
    parser.add_argument('-m',
                        '--model_filename',
                        type=str,
                        default='Full',
                        choices=['Full', 'Partial'],
                        help='Full or Partial model artifact')
    args = parser.parse_args()
    kwargs = vars(args) 

    labels = [
        'hen', 'crickets', 'airplane', 'chirping_birds', 'rain',
        'church_bells', 'crackling_fire', 'chainsaw', 'drinking_sipping',
        'footsteps', 'can_opening', 'keyboard_typing', 'clapping', 'fireworks',
        'cow', 'helicopter', 'engine', 'dog', 'snoring', 'door_wood_creaks',
        'frog', 'brushing_teeth', 'pouring_water', 'insects', 'laughing',
        'washing_machine', 'cat', 'hand_saw', 'toilet_flush', 'crying_baby',
        'vacuum_cleaner', 'breathing', 'sea_waves', 'coughing', 'wind',
        'sheep', 'glass_breaking', 'clock_tick', 'clock_alarm', 'crow',
        'rooster', 'door_wood_knock', 'thunderstorm', 'car_horn', 'siren',
        'pig', 'water_drops', 'sneezing', 'mouse_click', 'train'
    ]

    # labels = [
    # 'fireworks', 'train', 'crackling fire', 'pouring water', 'laughing',
    # 'frog', 'chirping birds', 'helicopter', 'breathing', 'crow',
    # 'vacuum cleaner', 'toilet flush', 'airplane', 'snoring',
    # 'door wood creaks', 'crickets', 'chainsaw', 'mouse click', 'sheep',
    # 'brushing teeth', 'crying baby', 'thunderstorm', 'keyboard typing',
    # 'cow', 'can opening', 'footsteps', 'washing machine', 'engine',
    # 'siren', 'church bells', 'clock alarm', 'hen', 'drinking sipping',
    # 'clapping', 'clock tick', 'coughing', 'wind', 'hand saw', 'rain',
    # 'sneezing', 'sea waves', 'pig', 'cat', 'door wood knock', 'dog',
    # 'water drops', 'car horn', 'rooster', 'glass breaking', 'insects'
    # ]

    # labels = []
    extra = []
    # extra =  ['cat', 'thunderstorm', 'coughing', 'alarm clock', 'car horn']
    # extra = ['alarm clock', 'car horn']
    labels += extra

    self = AudioCLIPInference(labels, **kwargs)
    self(**kwargs)
