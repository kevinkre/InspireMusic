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
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from inspiremusic.cli.model import InspireMusicModel
from inspiremusic.dataset.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser(description='inference only with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--prompt_data', required=True, help='prompt data file')
    # parser.add_argument('--prompt_utt2data', required=True, help='prompt data file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--llm_model', default=None,required=False, help='flow model file')

    parser.add_argument('--music_tokenizer', required=True, help='music tokenizer model file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--mode',
                        default='sft',
                        choices=['sft', 'zero_shot'],
                        help='inference mode')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Init inspiremusic models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f)

    model = InspireMusicModel(configs['llm'], configs['flow'], configs['hift'])
    model.load(args.llm_model, args.flow_model, args.music_tokenizer)
    test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False)
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    del configs
    os.makedirs(args.result_dir, exist_ok=True)
    fn = os.path.join(args.result_dir, 'wav.scp')
    f = open(fn, 'w')
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_data_loader)):
            utts = batch["utts"]
            assert len(utts) == 1, "inference mode only support batchsize 1"

            speech_token = batch["speech_token"].to(device)
            speech_token_len = batch["speech_token_len"].to(device)

            if args.mode == 'sft':
                model_input = {"speech_token":speech_token, "speech_token_len":speech_token_len}
            else:
                model_input = {'text': text_token, 'text_len': text_token_len,
                               'prompt_text': text_token, 'prompt_text_len': text_token_len,
                               'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                               'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                               'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                               'llm_embedding': utt_embedding, 'flow_embedding': utt_embedding}
                               
            music_audios = []
            for model_output in model.inference(**model_input):
                music_audios.append(model_output['music_audio'])

            # music_audios = torch.concat(music_audios, dim=1)

            music_key = utts[0]
            music_fn = os.path.join(args.result_dir, '{}.wav'.format(music_key))
            torchaudio.save(music_fn, music_audios[0], sample_rate=24000)
            f.write('{} {}\n'.format(music_key, music_fn))
            f.flush()
    f.close()
    logging.info('Result wav.scp saved in {}'.format(fn))

if __name__ == '__main__':
    main()