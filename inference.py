import glob
import os
import librosa

from model import AudioCLIP
from utils.transforms import ToTensor1D
from utils.extract_audio import extract_audio
from tqdm import tqdm

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
        for path_to_audio in tqdm(paths_to_audio):
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

    def __call__(self, input_dir=None, verbose=True, **kwargs):
        audio_dir = extract_audio(input_dir, **kwargs)
        audio, paths_to_audio = self.preprocess_audio(audio_dir, verbose=verbose)
        logits_audio_text = self.obtain_embeddings(audio)
        self.score_inputs(logits_audio_text, paths_to_audio)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-f','--input_dir')
    parser.add_argument('-n','--num',type=int,help='Limit to first N input files')
    args = parser.parse_args()

    #labels = ['dog', 'lightning', 'sneezing', 'alarm clock', 'car horn']
    labels = ['Music', 'Speech', 'Vehicle', 'Musical instrument', 'Plucked string instrument', 'Singing', 'Car', 'Animal', 'Outside, rural or natural', 'Violin, fiddle', 'Bird', 'Drum', 'Engine', 'Narration, monologue', 'Drum kit', 'Acoustic guitar', 'Dog', 'Child speech, kid speaking', 'Bass drum', 'Rail transport', 'Motor vehicle (road)', 'Water', 'Female speech, woman speaking', 'Siren', 'Railroad car, train wagon', 'Tools', 'Silence', 'Snare drum', 'Wind', 'Bird vocalization, bird call, bird song', 'Fowl', 'Wind instrument, woodwind instrument', 'Emergency vehicle', 'Laughter', 'Chirp, tweet', 'Rapping', 'Cheering', 'Gunshot, gunfire', 'Radio', 'Cat', 'Hi-hat', 'Helicopter', 'Fireworks', 'Stream', 'Bark', 'Baby cry, infant cry', 'Snoring', 'Train horn', 'Double bass', 'Explosion', 'Crowing, cock-a-doodle-doo', 'Bleat', 'Computer keyboard', 'Civil defense siren', 'Bee, wasp, etc.', 'Bell', 'Chainsaw', 'Oink', 'Tick', 'Tabla', 'Liquid', 'Traffic noise, roadway noise', 'Beep, bleep', 'Frying (food)', 'Whack, thwack', 'Sink (filling or washing)', 'Burping, eructation', 'Fart', 'Sneeze', 'Aircraft engine', 'Arrow', 'Giggle', 'Hiccup', 'Cough', 'Cricket', 'Sawing', 'Tambourine', 'Pump (liquid)', 'Squeak', 'Male speech, man speaking', 'Keyboard (musical)', 'Pigeon, dove', 'Motorboat, speedboat', 'Female singing', 'Brass instrument', 'Motorcycle', 'Choir', 'Race car, auto racing', 'Chicken, rooster', 'Idling', 'Sampler', 'Ukulele', 'Synthesizer', 'Cymbal', 'Spray', 'Accordion', 'Scratching (performance technique)', 'Child singing', 'Cluck', 'Water tap, faucet', 'Applause', 'Toilet flush', 'Whistling', 'Vacuum cleaner', 'Meow', 'Chatter', 'Whoop', 'Sewing machine', 'Bagpipes', 'Subway, metro, underground', 'Walk, footsteps', 'Whispering', 'Crying, sobbing', 'Thunder', 'Didgeridoo', 'Church bell', 'Ringtone', 'Buzzer', 'Splash, splatter', 'Fire alarm', 'Chime', 'Babbling', 'Glass', 'Chewing, mastication', 'Microwave oven', 'Air horn, truck horn', 'Growling', 'Telephone bell ringing', 'Moo', 'Change ringing (campanology)', 'Hands', 'Camera', 'Pour', 'Croak', 'Pant', 'Finger snapping', 'Gargling', 'Inside, small room', 'Outside, urban or manmade', 'Truck', 'Bowed string instrument', 'Medium engine (mid frequency)', 'Marimba, xylophone', 'Aircraft', 'Cello', 'Flute', 'Glockenspiel', 'Power tool', 'Fixed-wing aircraft, airplane', 'Waves, surf', 'Duck', 'Clarinet', 'Goat', 'Honk', 'Skidding', 'Hammond organ', 'Electronic organ', 'Thunderstorm', 'Steelpan', 'Slap, smack', 'Battle cry', 'Percussion', 'Trombone', 'Banjo', 'Mandolin', 'Guitar', 'Strum', 'Boat, Water vehicle', 'Accelerating, revving, vroom', 'Electric guitar', 'Orchestra', 'Wind noise (microphone)', 'Effects unit', 'Livestock, farm animals, working animals', 'Police car (siren)', 'Rain', 'Printer', 'Drum machine', 'Fire engine, fire truck (siren)', 'Insect', 'Skateboard', 'Coo', 'Conversation', 'Typing', 'Harp', 'Thump, thud', 'Mechanisms', 'Canidae, dogs, wolves', 'Chuckle, chortle', 'Rub', 'Boom', 'Hubbub, speech noise, speech babble', 'Telephone', 'Blender', 'Whimper', 'Screaming', 'Wild animals', 'Pig', 'Artillery fire', 'Electric shaver, electric razor', 'Baby laughter', 'Crow', 'Howl', 'Breathing', 'Cattle, bovinae', 'Roaring cats (lions, tigers)', 'Clapping', 'Alarm', 'Chink, clink', 'Ding', 'Toot', 'Clock', 'Children shouting', 'Fill (with liquid)', 'Purr', 'Rumble', 'Boing', 'Breaking', 'Light engine (high frequency)', 'Cash register', 'Bicycle bell', 'Inside, large room or hall', 'Domestic animals, pets', 'Bass guitar', 'Electric piano', 'Trumpet', 'Horse', 'Mallet percussion', 'Organ', 'Bicycle', 'Rain on surface', 'Quack', 'Drill', 'Machine gun', 'Lawn mower', 'Smash, crash', 'Trickle, dribble', 'Frog', 'Writing', 'Steam whistle', 'Groan', 'Hammer', 'Doorbell', 'Shofar', 'Cowbell', 'Wail, moan', 'Bouncing', 'Distortion', 'Vibraphone', 'Air brake', 'Field recording', 'Piano', 'Male singing', 'Bus', 'Wood', 'Tap', 'Ocean', 'Door', 'Vibration', 'Television', 'Harmonica', 'Basketball bounce', 'Clickety-clack', 'Dishes, pots, and pans', 'Crumpling, crinkling', 'Sitar', 'Tire squeal', 'Fly, housefly', 'Sizzle', 'Slosh', 'Engine starting', 'Mechanical fan', 'Stir', 'Children playing', 'Ping', 'Owl', 'Alarm clock', 'Car alarm', 'Telephone dialing, DTMF', 'Sine wave', 'Thunk', 'Coin (dropping)', 'Crunch', 'Zipper (clothing)', 'Mosquito', 'Shuffling cards', 'Pulleys', 'Toothbrush', 'Crowd', 'Saxophone', 'Rowboat, canoe, kayak', 'Steam', 'Ambulance (siren)', 'Goose', 'Crackle', 'Fire', 'Turkey', 'Heart sounds, heartbeat', 'Singing bowl', 'Reverberation', 'Clicking', 'Jet engine', 'Rodents, rats, mice', 'Typewriter', 'Caw', 'Knock', 'Ice cream truck, ice cream van', 'Stomach rumble', 'French horn', 'Roar', 'Theremin', 'Pulse', 'Train', 'Run', 'Vehicle horn, car horn, honking', 'Clip-clop', 'Sheep', 'Whoosh, swoosh, swish', 'Timpani', 'Throbbing', 'Firecracker', 'Belly laugh', 'Train whistle', 'Whistle', 'Whip', 'Gush', 'Biting', 'Scissors', 'Clang', 'Single-lens reflex camera', 'Chorus effect', 'Inside, public space', 'Steel guitar, slide guitar', 'Waterfall', 'Hum', 'Raindrop', 'Propeller, airscrew', 'Filing (rasp)', 'Reversing beeps', 'Shatter', 'Sanding', 'Wheeze', 'Hoot', 'Bow-wow', 'Car passing by', 'Tick-tock', 'Hiss', 'Snicker', 'Whimper (dog)', 'Shout', 'Echo', 'Rattle', 'Sliding door', 'Gobble', 'Plop', 'Yell', 'Drip', 'Neigh, whinny', 'Bellow', 'Keys jangling', 'Ding-dong', 'Buzz', 'Scratch', 'Rattle (instrument)', 'Hair dryer', 'Dial tone', 'Tearing', 'Bang', 'Noise', 'Bird flight, flapping wings', 'Grunt', 'Jackhammer', 'Drawer open or close', 'Whir', 'Tuning fork', 'Squawk', 'Jingle bell', 'Smoke detector, smoke alarm', 'Train wheels squealing', 'Caterwaul', 'Mouse', 'Crack', 'Whale vocalization', 'Squeal', 'Zither', 'Rimshot', 'Drum roll', 'Burst, pop', 'Wood block', 'Harpsichord', 'White noise', 'Bathtub (filling or washing)', 'Snake', 'Environmental noise', 'String section', 'Cacophony', 'Maraca', 'Snort', 'Yodeling', 'Electric toothbrush', 'Cupboard open or close', 'Sound effect', 'Tapping (guitar technique)', 'Ship', 'Sniff', 'Pink noise', 'Tubular bells', 'Gong', 'Flap', 'Throat clearing', 'Sigh', 'Busy signal', 'Zing', 'Sidetone', 'Crushing', 'Yip', 'Gurgling', 'Jingle, tinkle', 'Boiling', 'Mains hum', 'Humming', 'Sonar', 'Gasp', 'Power windows, electric windows', 'Splinter', 'Heart murmur', 'Air conditioning', 'Pizzicato', 'Ratchet, pawl', 'Chirp tone', 'Heavy engine (low frequency)', 'Rustling leaves', 'Speech synthesizer', 'Rustle', 'Clatter', 'Slam', 'Eruption', 'Cap gun', 'Synthetic singing', 'Shuffle', 'Wind chime', 'Chop', 'Scrape', 'Squish', 'Foghorn', "Dental drill, dentist's drill", 'Harmonic', 'Static', 'Sailboat, sailing ship', 'Cutlery, silverware', 'Gears', 'Chopping (food)', 'Creak', 'Fusillade', 'Roll', 'Electronic tuner', 'Patter', 'Electronic music', 'Dubstep', 'Techno', 'Rock and roll', 'Pop music', 'Rock music', 'Hip hop music', 'Classical music', 'Soundtrack music', 'House music', 'Heavy metal', 'Exciting music', 'Country', 'Electronica', 'Rhythm and blues', 'Background music', 'Dance music', 'Jazz', 'Mantra', 'Blues', 'Trance music', 'Electronic dance music', 'Theme music', 'Gospel music', 'Music of Latin America', 'Disco', 'Tender music', 'Punk rock', 'Funk', 'Music of Asia', 'Drum and bass', 'Vocal music', 'Progressive rock', 'Music for children', 'Video game music', 'Lullaby', 'Reggae', 'New-age music', 'Christian music', 'Independent music', 'Soul music', 'Music of Africa', 'Ambient music', 'Bluegrass', 'Afrobeat', 'Salsa music', 'Music of Bollywood', 'Beatboxing', 'Flamenco', 'Psychedelic rock', 'Opera', 'Folk music', 'Christmas music', 'Middle Eastern music', 'Grunge', 'Song', 'A capella', 'Sad music', 'Traditional music', 'Scary music', 'Ska', 'Chant', 'Carnatic music', 'Swing music', 'Happy music', 'Jingle (music)', 'Funny music', 'Angry music', 'Wedding music', 'Engine knocking']

    self = AudioCLIPInference(labels)
    self(**vars(args))

