import os
import argparse
import torch
import time

from . import PianoTranscription, sample_rate, load_audio


def inference(args):
    """Inference template.

    Args:
      model_type: str
      audio_path: str
      cuda: bool
    """

    # Arugments & parameters
    audio_path = args.audio_path
    output_midi_path = args.output_midi_path
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    type0 = args.type0
 
    # Load audio
    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device=device, checkpoint_path=None)
    """device: 'cuda' | 'cpu'
    checkpoint_path: None for default path, or str for downloaded checkpoint path.
    """

    # Transcribe and write out to MIDI file
    transcribe_time = time.time()
    transcribed_dict = transcriptor.transcribe(audio, output_midi_path, type0=type0)
    print('Transcribe time: {:.3f} s'.format(time.time() - transcribe_time))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_midi_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--fem', '--type0', dest='type0', action='store_true', default=False)

    args = parser.parse_args()
    inference(args)
