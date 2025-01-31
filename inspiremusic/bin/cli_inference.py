# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from inspiremusic.cli.inspiremusic import InspireMusic
import time
from inspiremusic.utils.audio_utils import trim_audio, fade_out


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description='inference only with your model')
    parser.add_argument('--model_dir', default="../../pretrained_models/InspireMusic-1.5B-Long", required=False, help='model folder path')
    parser.add_argument('--config', default="../../pretrained_models/InspireMusic-1.5B-Long/inspiremusic.yaml", required=False, help='config file')
    parser.add_argument('--text', default="generate a piece of electric and pop music.", required=False, help='prompt text')
    parser.add_argument('--audio_prompt', default=None, required=False, help='prompt audio')
    parser.add_argument('--flow_model', default="../../pretrained_models/InspireMusic-1.5B-Long/flow.pt", required=False, help='flow model file')
    parser.add_argument('--llm_model', default="../../pretrained_models/InspireMusic-1.5B-Long/llm.pt", required=False, help='flow model file')
    parser.add_argument('--music_tokenizer', default="../../pretrained_models/InspireMusic-1.5B-Long/music_tokenizer", required=False, help='music tokenizer model file')
    parser.add_argument('--wavtokenizer', default="../../pretrained_models/InspireMusic-1.5B-Long/wavtokenizer", required=False, help='wavtokenizer model file')
    parser.add_argument('--chorus', default="intro", required=False, help='chorus tag generation mode, eg. random, verse, chorus, intro.')
    parser.add_argument('--fast', action='store_true', required=False, help='True: fast inference mode, without flow matching for fast inference. False: normal inference mode, with flow matching for high quality.')
    parser.add_argument('--fp16', default=True, type=bool, required=False, help='inference with fp16 model')
    parser.add_argument('--fade_out', default=True, type=bool, required=False, help='add fade out effect to generated audio')
    parser.add_argument('--fade_out_duration', default=1.0, type=float, required=False, help='fade out duration in seconds')
    parser.add_argument('--trim', default=False, type=bool, required=False, help='trim the silence ending of generated audio')
    parser.add_argument('--format', type=str, default="wav", required=False,
                        choices=["wav", "mp3", "m4a", "flac"],
                        help='sampling rate of input audio')
    parser.add_argument('--sample_rate', type=int, default=24000, required=False,
                        help='sampling rate of input audio')
    parser.add_argument('--output_sample_rate', type=int, default=48000, required=False, choices=[24000, 48000],
                        help='sampling rate of generated output audio')
    parser.add_argument('--time_start', type=float, default=0.0, required=False,
                        help='the start time in seconds')
    parser.add_argument('--time_end', type=float, default=30.0, required=False,
                        help='the end time in seconds')
    parser.add_argument('--max_audio_prompt_length', type=float, default=5.0, required=False,
                        help='the max audio prompt length in seconds')
    parser.add_argument('--min_generate_audio_seconds', type=float, default=10.0, required=False,
                        help='the minimum generated audio length in seconds')
    parser.add_argument('--max_generate_audio_seconds', type=float, default=30.0, required=False,
                        help='the maximum generated audio length in seconds')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--task',
                        default='text-to-music',
                        choices=['text-to-music', 'continuation', "reconstruct", "super_resolution"],
                        help='choose inference task type. text-to-music: text-to-music task. continuation: music continuation task. reconstruct: reconstruction of original music. super_resolution: convert original 24kHz music into 48kHz music.')
    parser.add_argument('--result_dir', default="./exp/inspiremusic", required=False, help='generate audio folder')
    parser.add_argument('--output_fn', default="output_audio", required=False, help='output file name')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.fast:
        args.output_sample_rate = 24000

    min_generate_audio_length = int(args.output_sample_rate * args.min_generate_audio_seconds)
    max_generate_audio_length = int(args.output_sample_rate * args.max_generate_audio_seconds)
    assert args.min_generate_audio_seconds <= args.max_generate_audio_seconds
    
    # Init inspiremusic models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f)

    model = InspireMusic(args.model_dir, True, False, args.fast, args.fp16)

    if args.llm_model is None:
        model.model.llm = None
    else:
        model.model.llm = model.model.llm.to(torch.float32)

    if args.flow_model is None:
        model.model.flow = None

    del configs
    os.makedirs(args.result_dir, exist_ok=True)

    with torch.no_grad():
        text = args.text

        if args.chorus == "random":
            chorus = torch.randint(1, 5, (1,)).to(torch.int).to(device)
        elif args.chorus == "intro":
            chorus = torch.Tensor([0]).to(torch.int).to(device)
        elif "verse" in args.chorus:
            chorus = torch.Tensor([1]).to(torch.int).to(device)
        elif args.chorus == "chorus":
            chorus = torch.Tensor([2]).to(torch.int).to(device)
        elif args.chorus == "outro":
            chorus = torch.Tensor([4]).to(torch.int).to(device)
        else:
            chorus = torch.Tensor([1]).to(torch.int).to(device)

        time_start = torch.tensor([args.time_start], dtype=torch.float64, device=device)
        time_end = torch.tensor([args.time_end], dtype=torch.float64, device=device)

        text_prompt = f"<|{args.time_start}|><|{args.chorus}|><|{args.text}|><|{args.time_end}|>"
        music_fn = os.path.join(args.result_dir, f'{args.output_fn}.{args.format}')

        bench_start = time.time()

        if args.task == 'text-to-music':
            model_input = {"text": text, "audio_prompt": args.audio_prompt, "time_start": time_start, "time_end": time_end,
                            "chorus": chorus, "task": args.task, "stream": False, "sampling": 50, "duration_to_gen": args.max_generate_audio_seconds, "sr": args.sample_rate}
        elif args.task == 'continuation':
            if args.audio_prompt is not None:
                audio, sample_rate = torchaudio.load(args.audio_prompt)
                if audio.size(0) == 2:
                    audio = audio.mean(dim=0, keepdim=True)
                else:
                    audio = audio
                if sample_rate != args.sample_rate:
                    audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.sample_rate)(audio)
                max_audio_prompt_length = int(args.max_audio_prompt_length * args.sample_rate)
                if audio.size()[1] >= max_audio_prompt_length:
                    audio = audio[:,:max_audio_prompt_length]
                elif audio.size()[1] < int(args.sample_rate):
                    logging.warning(f"Warning: Input prompt audio length of {audio.size()[1]/args.sample_rate}s is shorter than 1s. Please provide an audio prompt of the appropriate length and try again.")
            model_input = {"text": text, "audio_prompt": audio, "time_start": time_start, "time_end": time_end,
                            "chorus": chorus, "task": args.task, "stream": False, "sampling": 50, "duration_to_gen": args.max_generate_audio_seconds, "sr": args.sample_rate}
        
        music_audios = []
        for model_output in model.cli_inference(**model_input):
            music_audios.append(model_output['music_audio'])

        bench_end = time.time()

        if args.trim:
            music_audio = trim_audio(music_audios[0], sample_rate=args.output_sample_rate, threshold=0.05, min_silence_duration=0.8)
        else:
            music_audio = music_audios[0]
        if music_audio.shape[0] != 0:
            if music_audio.shape[1] > max_generate_audio_length:
                music_audio = music_audio[:, :max_generate_audio_length]
            if music_audio.shape[1] >= min_generate_audio_length:
                try:
                    if args.fade_out:
                        music_audio = fade_out(music_audio, args.output_sample_rate, args.fade_out_duration)

                    music_audio = music_audio.repeat(2, 1)
                        
                    if args.format in ["wav", "flac"]:
                        torchaudio.save(music_fn, music_audio, sample_rate=args.output_sample_rate, encoding="PCM_S", bits_per_sample=24)
                    elif args.format in ["mp3", "m4a"]:
                        torchaudio.backend.sox_io_backend.save(filepath=music_fn, src=music_audio, sample_rate=args.output_sample_rate, format=args.format)
                    else:
                        logging.info(f"Format is not supported. Please choose from wav, mp3, m4a, flac.")
                except Exception as e:
                    logging.info(f"Error saving file: {e}")
                    raise
                
            audio_duration = music_audio.shape[1] / args.output_sample_rate
            rtf = (bench_end - bench_start) / audio_duration
            logging.info(f"processing time: {int(bench_end - bench_start)}s, save time: {int(bench_end - time.time())}s audio length: {int(audio_duration)}s, rtf: {rtf}, text prompt: {text_prompt}")
        else:
            logging.info(f"Generate audio length {music_audio.shape[1]} is shorter than min_generate_audio_length.")


if __name__ == '__main__':
    main()
