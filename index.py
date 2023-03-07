import json
import os
import numpy as np

from argparse import ArgumentParser
from autofaiss import build_index
from model import AudioCLIP
import torch


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--text_dir', type=str, default=None)
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--model', type=str, default='AudioCLIP-Full-Training.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--chunk_size', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_prepro_workers', type=int, default=8)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--save_entries', type=bool, default=False)
    parser.add_argument('--lower', type=bool, default=True)
    parser.add_argument('--use_line', type=bool, default=False)
    parser.add_argument('--use_unigrams', type=bool, default=False)
    parser.add_argument('--use_bigrams', type=bool, default=False)
    parser.add_argument('--use_trigrams', type=bool, default=False)
    parser.add_argument('--topk_ngrams', type=int, default=10000)
    parser.add_argument('--filter', type=int, default=3)
    parser.add_argument('--metric_type', type=str, default='ip')
    parser.add_argument('--max_index_memory_usage', type=str, default='8GB')
    parser.add_argument('--current_memory_available', type=str, default='16GB')
    parser.add_argument('--max_index_query_time_ms', type=int, default=10)
    args = parser.parse_args()

    labels = ['Music', 'Speech', 'Vehicle', 'Musical instrument', 'Plucked string instrument', 'Singing', 'Car', 'Animal', 'Outside, rural or natural', 'Violin, fiddle', 'Bird', 'Drum', 'Engine', 'Narration, monologue', 'Drum kit', 'Acoustic guitar', 'Dog', 'Child speech, kid speaking', 'Bass drum', 'Rail transport', 'Motor vehicle (road)', 'Water', 'Female speech, woman speaking', 'Siren', 'Railroad car, train wagon', 'Tools', 'Silence', 'Snare drum', 'Wind', 'Bird vocalization, bird call, bird song', 'Fowl', 'Wind instrument, woodwind instrument', 'Emergency vehicle', 'Laughter', 'Chirp, tweet', 'Rapping', 'Cheering', 'Gunshot, gunfire', 'Radio', 'Cat', 'Hi-hat', 'Helicopter', 'Fireworks', 'Stream', 'Bark', 'Baby cry, infant cry', 'Snoring', 'Train horn', 'Double bass', 'Explosion', 'Crowing, cock-a-doodle-doo', 'Bleat', 'Computer keyboard', 'Civil defense siren', 'Bee, wasp, etc.', 'Bell', 'Chainsaw', 'Oink', 'Tick', 'Tabla', 'Liquid', 'Traffic noise, roadway noise', 'Beep, bleep', 'Frying (food)', 'Whack, thwack', 'Sink (filling or washing)', 'Burping, eructation', 'Fart', 'Sneeze', 'Aircraft engine', 'Arrow', 'Giggle', 'Hiccup', 'Cough', 'Cricket', 'Sawing', 'Tambourine', 'Pump (liquid)', 'Squeak', 'Male speech, man speaking', 'Keyboard (musical)', 'Pigeon, dove', 'Motorboat, speedboat', 'Female singing', 'Brass instrument', 'Motorcycle', 'Choir', 'Race car, auto racing', 'Chicken, rooster', 'Idling', 'Sampler', 'Ukulele', 'Synthesizer', 'Cymbal', 'Spray', 'Accordion', 'Scratching (performance technique)', 'Child singing', 'Cluck', 'Water tap, faucet', 'Applause', 'Toilet flush', 'Whistling', 'Vacuum cleaner', 'Meow', 'Chatter', 'Whoop', 'Sewing machine', 'Bagpipes', 'Subway, metro, underground', 'Walk, footsteps', 'Whispering', 'Crying, sobbing', 'Thunder', 'Didgeridoo', 'Church bell', 'Ringtone', 'Buzzer', 'Splash, splatter', 'Fire alarm', 'Chime', 'Babbling', 'Glass', 'Chewing, mastication', 'Microwave oven', 'Air horn, truck horn', 'Growling', 'Telephone bell ringing', 'Moo', 'Change ringing (campanology)', 'Hands', 'Camera', 'Pour', 'Croak', 'Pant', 'Finger snapping', 'Gargling', 'Inside, small room', 'Outside, urban or manmade', 'Truck', 'Bowed string instrument', 'Medium engine (mid frequency)', 'Marimba, xylophone', 'Aircraft', 'Cello', 'Flute', 'Glockenspiel', 'Power tool', 'Fixed-wing aircraft, airplane', 'Waves, surf', 'Duck', 'Clarinet', 'Goat', 'Honk', 'Skidding', 'Hammond organ', 'Electronic organ', 'Thunderstorm', 'Steelpan', 'Slap, smack', 'Battle cry', 'Percussion', 'Trombone', 'Banjo', 'Mandolin', 'Guitar', 'Strum', 'Boat, Water vehicle', 'Accelerating, revving, vroom', 'Electric guitar', 'Orchestra', 'Wind noise (microphone)', 'Effects unit', 'Livestock, farm animals, working animals', 'Police car (siren)', 'Rain', 'Printer', 'Drum machine', 'Fire engine, fire truck (siren)', 'Insect', 'Skateboard', 'Coo', 'Conversation', 'Typing', 'Harp', 'Thump, thud', 'Mechanisms', 'Canidae, dogs, wolves', 'Chuckle, chortle', 'Rub', 'Boom', 'Hubbub, speech noise, speech babble', 'Telephone', 'Blender', 'Whimper', 'Screaming', 'Wild animals', 'Pig', 'Artillery fire', 'Electric shaver, electric razor', 'Baby laughter', 'Crow', 'Howl', 'Breathing', 'Cattle, bovinae', 'Roaring cats (lions, tigers)', 'Clapping', 'Alarm', 'Chink, clink', 'Ding', 'Toot', 'Clock', 'Children shouting', 'Fill (with liquid)', 'Purr', 'Rumble', 'Boing', 'Breaking', 'Light engine (high frequency)', 'Cash register', 'Bicycle bell', 'Inside, large room or hall', 'Domestic animals, pets', 'Bass guitar', 'Electric piano', 'Trumpet', 'Horse', 'Mallet percussion', 'Organ', 'Bicycle', 'Rain on surface', 'Quack', 'Drill', 'Machine gun', 'Lawn mower', 'Smash, crash', 'Trickle, dribble', 'Frog', 'Writing', 'Steam whistle', 'Groan', 'Hammer', 'Doorbell', 'Shofar', 'Cowbell', 'Wail, moan', 'Bouncing', 'Distortion', 'Vibraphone', 'Air brake', 'Field recording', 'Piano', 'Male singing', 'Bus', 'Wood', 'Tap', 'Ocean', 'Door', 'Vibration', 'Television', 'Harmonica', 'Basketball bounce', 'Clickety-clack', 'Dishes, pots, and pans', 'Crumpling, crinkling', 'Sitar', 'Tire squeal', 'Fly, housefly', 'Sizzle', 'Slosh', 'Engine starting', 'Mechanical fan', 'Stir', 'Children playing', 'Ping', 'Owl', 'Alarm clock', 'Car alarm', 'Telephone dialing, DTMF', 'Sine wave', 'Thunk', 'Coin (dropping)', 'Crunch', 'Zipper (clothing)', 'Mosquito', 'Shuffling cards', 'Pulleys', 'Toothbrush', 'Crowd', 'Saxophone', 'Rowboat, canoe, kayak', 'Steam', 'Ambulance (siren)', 'Goose', 'Crackle', 'Fire', 'Turkey', 'Heart sounds, heartbeat', 'Singing bowl', 'Reverberation', 'Clicking', 'Jet engine', 'Rodents, rats, mice', 'Typewriter', 'Caw', 'Knock', 'Ice cream truck, ice cream van', 'Stomach rumble', 'French horn', 'Roar', 'Theremin', 'Pulse', 'Train', 'Run', 'Vehicle horn, car horn, honking', 'Clip-clop', 'Sheep', 'Whoosh, swoosh, swish', 'Timpani', 'Throbbing', 'Firecracker', 'Belly laugh', 'Train whistle', 'Whistle', 'Whip', 'Gush', 'Biting', 'Scissors', 'Clang', 'Single-lens reflex camera', 'Chorus effect', 'Inside, public space', 'Steel guitar, slide guitar', 'Waterfall', 'Hum', 'Raindrop', 'Propeller, airscrew', 'Filing (rasp)', 'Reversing beeps', 'Shatter', 'Sanding', 'Wheeze', 'Hoot', 'Bow-wow', 'Car passing by', 'Tick-tock', 'Hiss', 'Snicker', 'Whimper (dog)', 'Shout', 'Echo', 'Rattle', 'Sliding door', 'Gobble', 'Plop', 'Yell', 'Drip', 'Neigh, whinny', 'Bellow', 'Keys jangling', 'Ding-dong', 'Buzz', 'Scratch', 'Rattle (instrument)', 'Hair dryer', 'Dial tone', 'Tearing', 'Bang', 'Noise', 'Bird flight, flapping wings', 'Grunt', 'Jackhammer', 'Drawer open or close', 'Whir', 'Tuning fork', 'Squawk', 'Jingle bell', 'Smoke detector, smoke alarm', 'Train wheels squealing', 'Caterwaul', 'Mouse', 'Crack', 'Whale vocalization', 'Squeal', 'Zither', 'Rimshot', 'Drum roll', 'Burst, pop', 'Wood block', 'Harpsichord', 'White noise', 'Bathtub (filling or washing)', 'Snake', 'Environmental noise', 'String section', 'Cacophony', 'Maraca', 'Snort', 'Yodeling', 'Electric toothbrush', 'Cupboard open or close', 'Sound effect', 'Tapping (guitar technique)', 'Ship', 'Sniff', 'Pink noise', 'Tubular bells', 'Gong', 'Flap', 'Throat clearing', 'Sigh', 'Busy signal', 'Zing', 'Sidetone', 'Crushing', 'Yip', 'Gurgling', 'Jingle, tinkle', 'Boiling', 'Mains hum', 'Humming', 'Sonar', 'Gasp', 'Power windows, electric windows', 'Splinter', 'Heart murmur', 'Air conditioning', 'Pizzicato', 'Ratchet, pawl', 'Chirp tone', 'Heavy engine (low frequency)', 'Rustling leaves', 'Speech synthesizer', 'Rustle', 'Clatter', 'Slam', 'Eruption', 'Cap gun', 'Synthetic singing', 'Shuffle', 'Wind chime', 'Chop', 'Scrape', 'Squish', 'Foghorn', "Dental drill, dentist's drill", 'Harmonic', 'Static', 'Sailboat, sailing ship', 'Cutlery, silverware', 'Gears', 'Chopping (food)', 'Creak', 'Fusillade', 'Roll', 'Electronic tuner', 'Patter', 'Electronic music', 'Dubstep', 'Techno', 'Rock and roll', 'Pop music', 'Rock music', 'Hip hop music', 'Classical music', 'Soundtrack music', 'House music', 'Heavy metal', 'Exciting music', 'Country', 'Electronica', 'Rhythm and blues', 'Background music', 'Dance music', 'Jazz', 'Mantra', 'Blues', 'Trance music', 'Electronic dance music', 'Theme music', 'Gospel music', 'Music of Latin America', 'Disco', 'Tender music', 'Punk rock', 'Funk', 'Music of Asia', 'Drum and bass', 'Vocal music', 'Progressive rock', 'Music for children', 'Video game music', 'Lullaby', 'Reggae', 'New-age music', 'Christian music', 'Independent music', 'Soul music', 'Music of Africa', 'Ambient music', 'Bluegrass', 'Afrobeat', 'Salsa music', 'Music of Bollywood', 'Beatboxing', 'Flamenco', 'Psychedelic rock', 'Opera', 'Folk music', 'Christmas music', 'Middle Eastern music', 'Grunge', 'Song', 'A capella', 'Sad music', 'Traditional music', 'Scary music', 'Ska', 'Chant', 'Carnatic music', 'Swing music', 'Happy music', 'Jingle (music)', 'Funny music', 'Angry music', 'Wedding music', 'Engine knocking']
    
    # Load clip, compute embeddings and save
    if args.text_dir:
        torch.set_grad_enabled(False)
        model_filename = 'AudioCLIP-Full-Training.pt'
        
        aclp = AudioCLIP(pretrained=f'assets/{model_filename}')
        
        text = [[label] for label in labels]
        ((_, _, text_features), _), _ = aclp(text=text)

    # Compute index
        build_index(embeddings=text_features.numpy(),
                       index_path="faiss-index-audioset-527.index",
                       index_infos_path="faiss-index-audioset-527-infos.json",
                       metric_type=args.metric_type,
                       max_index_memory_usage=args.max_index_memory_usage,
                       max_index_query_time_ms=args.max_index_query_time_ms)


if __name__ == '__main__':
    main()