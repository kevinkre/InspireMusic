# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from inspiremusic.utils.mask import make_pad_mask
from inspiremusic.music_tokenizer.vqvae import VQVAE

class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000},
                generator_model_dir: str = "../../pretrained_models/InspireMusic/music_tokenizer"
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss
        self.quantizer = VQVAE( f'{generator_model_dir}/config.json',
                                  f'{generator_model_dir}/model.pt',with_encoder=True).quantizer
        self.quantizer.eval()
        self.num_codebooks  = 4
                                  
    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:

        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        speech_token  = speech_token.view(speech_token.size(0),-1,self.num_codebooks)
        if "semantic_token" not in batch:
            token = speech_token[:,:,0]
            token_len = (speech_token_len/self.num_codebooks).long()
    
        else:
            token = batch['semantic_token'].to(device)
            token_len = batch['speech_token_len'].to(device)
        
        with torch.no_grad():
            feat = self.quantizer.embed(speech_token)
            feat_len = (speech_token_len/self.num_codebooks).long()

        token_x = self.input_embedding(token) 


        h, h_lengths = self.encoder(token_x, token_len)

        h, h_lengths = self.length_regulator(h, feat_len)   
        conds = None

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = feat 
        loss, _ = self.decoder.compute_loss(
            feat,
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            None,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len):
        assert token.shape[0] == 1
        
        token_x = self.input_embedding(torch.clamp(token, min=0)) 

        h, h_lengths = self.encoder(token_x, token_len)

        h, h_lengths = self.length_regulator(h, token_len)  

        # get conditions
        conds = None

        mask = (~make_pad_mask(token_len)).to(h)
        feat = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=None,
            cond=conds,
            n_timesteps=10
        )
        return feat
