import glob
import os
import librosa
from model import AudioCLIP
from utils.transforms import ToTensor1D

import numpy as np
import torch

torch.set_grad_enabled(False)



INPUT_DIR = 'demo/audio'
paths_to_audio = glob.glob(f'{INPUT_DIR}/*.wav')

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
SAMPLE_RATE = 44100

aclp = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}')
audio_transforms = ToTensor1D()

LABELS = ['dog', 'lightning', 'sneezing', 'alarm clock', 'car horn']
text = [[label] for label in LABELS]



audio = list()
for path_to_audio in paths_to_audio:
    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

    spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

    audio.append((track, pow_spec))



audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
# images = torch.stack([image_transforms(image) for image in images])
# text = [[label] for label in LABELS]

((audio_features, _, _), _), _ = aclp(audio=audio)
((_, _, text_features), _), _ = aclp(text=text)
audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
logits_audio_text = scale_audio_text * audio_features @ text_features.T

print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')

# calculate model confidence
confidence = logits_audio_text.softmax(dim=1)
for audio_idx in range(len(paths_to_audio)):
    # acquire Top-3 most similar results
    conf_values, ids = confidence[audio_idx].topk(3)

    # format output strings
    query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
    results = ', '.join([f'{LABELS[i]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

    print(query + results)

