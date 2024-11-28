#!/usr/bin/env python3
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
import argparse
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper


def main(args):
    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    utt2speech_token = {}
    for utt in tqdm(utt2wav.keys()):
        audio, sample_rate = torchaudio.load(utt2wav[utt])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        if audio.shape[1] / 16000 > 30:
            logging.warning('do not support extract speech token for audio longer than 30s')
            speech_token = []
        else:
            feat = whisper.log_mel_spectrogram(audio, n_mels=128)
            speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                  ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        utt2speech_token[utt] = speech_token
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--onnx_path',
                        type=str)
    args = parser.parse_args()
    main(args)
