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
import torch
import numpy as np
import threading
import time
from contextlib import nullcontext
import uuid
from inspiremusic.music_tokenizer.vqvae import VQVAE
from inspiremusic.wavtokenizer.decoder.pretrained import WavTokenizer
from torch.cuda.amp import autocast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InspireMusicModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 music_tokenizer: torch.nn.Module,
                 wavtokenizer: torch.nn.Module,
                 fast: bool = False,
                 fp16: bool = True,
                 ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.music_tokenizer = music_tokenizer
        self.wavtokenizer = wavtokenizer
        self.fp16 = fp16
        self.token_min_hop_len = 100
        self.token_max_hop_len = 200
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = 34
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.music_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.fast = fast
        self.generator = "hifi"

    def load(self, llm_model, flow_model, hift_model, wavtokenizer_model):
        if llm_model is not None:
            self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
            self.llm.to(self.device).eval()
            if self.fp16 is True:
                self.llm.half()
        if flow_model is not None:
            self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
            self.flow.to(self.device).eval()
        if hift_model is not None:
            self.music_tokenizer = VQVAE( hift_model + '/config.json',
                                  hift_model + '/model.pt', with_encoder=True)                    
            self.music_tokenizer.to(self.device).eval()
        if self.wavtokenizer is not None:
            self.wavtokenizer = WavTokenizer.from_pretrained_feat( wavtokenizer_model + '/config.yaml',
                                  wavtokenizer_model + '/model.pt')                    
            self.wavtokenizer.to(self.device)

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        assert self.fp16 is True, "we only provide fp16 jit model, set fp16=True if you want to use jit model"
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model)
        self.flow.encoder = flow_encoder

    def load_onnx(self, flow_decoder_estimator_model):
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option, providers=providers)

    def llm_job(self, text, prompt_text, llm_prompt_audio_token, embeddings, uuid, duration_to_gen):
        with self.llm_context:
            local_res = []
            with autocast(enabled=self.fp16):
                for i in self.llm.inference(text=text.to(self.device),
                                            text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_text=prompt_text.to(self.device),
                                            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_audio_token=llm_prompt_audio_token.to(self.device),
                                            prompt_audio_token_len=torch.tensor([llm_prompt_audio_token.shape[1]], dtype=torch.int32).to(self.device),
                                            embeddings=embeddings,
                                            sampling=350,
                                            duration_to_gen=duration_to_gen,
                                            ):
                    local_res.append(i)
            self.music_token_dict[uuid] = local_res
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, token_len, uuid, finalize=False):
        codec_embed = self.flow.inference(token=token.to(self.device),
                                          token_len=token_len.to(self.device),
                                         )
        # use music_tokenizer decoder
        wav = self.music_tokenizer.generator(codec_embed)
        wav = wav.squeeze(0).cpu().detach()
        return wav

    def semantictoken2wav(self, token):
        # fast mode, use wavtokenizer decoder
        new_tensor = torch.tensor(token.to(self.device)).unsqueeze(0)
        features = self.wavtokenizer.codes_to_features(new_tensor)
        bandwidth_id = torch.tensor([0]).to(self.device)
        wav = self.wavtokenizer.to(self.device).decode(features, bandwidth_id=bandwidth_id)
        wav = wav.cpu().detach()
        return wav

    def continuation_inference(self, text, audio_token, audio_token_len, text_token, text_token_len, embeddings=None,
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                  llm_prompt_audio_token=torch.zeros(1, 0, dtype=torch.int32),
                  flow_prompt_audio_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_audio_feat=torch.zeros(1, 0, 80), duration_to_gen = 30, trim = True, stream=False, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        # music continuation task
        # require either audio input only or text and audio inputs

        this_uuid = str(uuid.uuid1())

        input_token = torch.cat([t for t in (text_token, audio_token) if t is not None], dim=1) if text_token or audio_token else None

        if self.llm:
            with self.lock:
                self.music_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            p = threading.Thread(target=self.llm_job, args=(input_token, prompt_text, llm_prompt_audio_token, embeddings, this_uuid, duration_to_gen))
            p.start()
        
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.music_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_music_audio = self.token2wav(token=audio_token,
                                                     token_len=audio_token_len,
                                                        uuid=this_uuid,
                                                        finalize=False)
                    yield {'music_audio': this_music_audio.cpu()}
                    with self.lock:
                        self.music_token_dict[this_uuid] = self.music_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better audio quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.music_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_audio is not None
            this_music_token = torch.concat(self.music_token_dict[this_uuid], dim=1)
            with self.flow_hift_context:
                this_music_audio = self.token2wav(token=this_music_token,
                                                 prompt_token=flow_prompt_audio_token,
                                                 prompt_feat=prompt_audio_feat,
                                                 embedding=flow_embedding,
                                                 uuid=this_uuid,
                                                 finalize=True)
            yield {'music_audio': this_music_audio.cpu()}
        else:
            # deal with all tokens
            if self.fast:
                if self.llm:
                    p.join()
                    this_music_token = torch.concat(self.music_token_dict[this_uuid], dim=1)
                else:
                    this_music_token = audio_token

                logging.info("using wavtokenizer generator without flow matching")
                this_music_audio = self.semantictoken2wav(token=this_music_token)

            else:
                if self.llm:
                    p.join()
                    if len(self.music_token_dict[this_uuid]) != 0:
                        this_music_token = torch.concat(self.music_token_dict[this_uuid], dim=1)
                    else:
                        print(f"The list of tensors is empty for UUID: {this_uuid}")
                else:
                    this_music_token = audio_token
                logging.info(f"LLM generated audio token length: {this_music_token.shape[1]}")
                logging.info(f"using flow matching and {self.generator} generator")
                
                if self.generator == "hifi":
                    if (embeddings[1] - embeddings[0]) < duration_to_gen:
                        if trim:
                            trim_length = (int((embeddings[1] - embeddings[0])*75))
                            this_music_token = this_music_token[:, :trim_length]
                            logging.info(f"After trimmed, generated audio token length: {this_music_token.shape[1]}")
                    elif (embeddings[1] - embeddings[0]) < 1:
                        logging.info(f"Given audio length={(embeddings[1] - embeddings[0])}, which is too short, please give a longer audio length.")
                    
                    this_music_audio = self.token2wav(token=this_music_token,
                                                token_len=torch.LongTensor([this_music_token.size(1)]),
                                                uuid=this_uuid,
                                                finalize=True)
                    logging.info(f"Generated audio sequence length: {this_music_audio.shape[1]}")
                elif self.generator == "wavtokenizer":
                    if (embeddings[1] - embeddings[0]) < duration_to_gen:
                        if trim:
                            trim_length = (int((embeddings[1] - embeddings[0])*75))
                            this_music_token = this_music_token[:,:trim_length]
                            logging.info(f"After trimmed, generated audio token length: {this_music_token.shape[1]}")
                    elif (embeddings[1] - embeddings[0]) < 1:
                        logging.info(f"Given audio length={(embeddings[1] - embeddings[0])}, which is too short, please give a longer audio length.")

                    this_music_audio = self.semantictoken2wav(token=this_music_token)

            yield {'music_audio': this_music_audio.cpu()}
            torch.cuda.synchronize()



    def inference(self, text, audio_token, audio_token_len, text_token, text_token_len, embeddings=None,
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                  llm_prompt_audio_token=torch.zeros(1, 0, dtype=torch.int32),
                  flow_prompt_audio_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_audio_feat=torch.zeros(1, 0, 80), duration_to_gen = 30, trim = True, stream=False, **kwargs):
        # text to music task
        # this_uuid is used to track variables related to this inference thread

        this_uuid = str(uuid.uuid1())

        if self.llm:
            with self.lock:
                self.music_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            p = threading.Thread(target=self.llm_job, args=(text_token, prompt_text, llm_prompt_audio_token, embeddings, this_uuid, duration_to_gen))
            p.start()

        
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.music_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_music_audio = self.token2wav(token=text_token,
                                                     token_len=text_token_len,
                                                        uuid=this_uuid,
                                                        finalize=False)
                    yield {'music_audio': this_music_audio.cpu()}
                    with self.lock:
                        self.music_token_dict[this_uuid] = self.music_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better audio quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.music_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_music_token = torch.concat(self.music_token_dict[this_uuid], dim=1)
            with self.flow_hift_context:
                this_music_audio = self.token2wav(token=this_music_token,
                                                 prompt_token=flow_prompt_audio_token,
                                                 prompt_feat=prompt_audio_feat,
                                                 embedding=flow_embedding,
                                                 uuid=this_uuid,
                                                 finalize=True)
            yield {'music_audio': this_music_audio.cpu()}
        else:
            # deal with all tokens
            if self.fast:
                if self.llm:
                    p.join()
                    this_music_token = torch.concat(self.music_token_dict[this_uuid], dim=1)
                else:
                    this_music_token = text_token

                logging.info("using wavtokenizer generator without flow matching")
                this_music_audio = self.semantictoken2wav(token=this_music_token)

            else:
                if self.llm:
                    p.join()
                    if len(self.music_token_dict[this_uuid]) != 0:
                        this_music_token = torch.concat(self.music_token_dict[this_uuid], dim=1)
                    else:
                        print(f"The list of tensors is empty for UUID: {this_uuid}")
                else:
                    this_music_token = text_token
                logging.info(f"LLM generated audio token length: {this_music_token.shape[1]}")
                logging.info(f"using flow matching and {self.generator} generator")
                
                if self.generator == "hifi":
                    if (embeddings[1] - embeddings[0]) < duration_to_gen:
                        if trim:
                            trim_length = (int((embeddings[1] - embeddings[0])*75))
                            this_music_token = this_music_token[:, :trim_length]
                            logging.info(f"After trimmed, generated audio token length: {this_music_token.shape[1]}")
                    elif (embeddings[1] - embeddings[0]) < 1:
                        logging.info(f"Given audio length={(embeddings[1] - embeddings[0])}, which is too short, please give a longer audio length.")
                    
                    this_music_audio = self.token2wav(token=this_music_token,
                                                token_len=torch.LongTensor([this_music_token.size(1)]),
                                                uuid=this_uuid,
                                                finalize=True)
                    logging.info(f"Generated audio sequence length: {this_music_audio.shape[1]}")
                elif self.generator == "wavtokenizer":
                    if (embeddings[1] - embeddings[0]) < duration_to_gen:
                        if trim:
                            trim_length = (int((embeddings[1] - embeddings[0])*75))
                            this_music_token = this_music_token[:,:trim_length]
                            logging.info(f"After trimmed, generated audio token length: {this_music_token.shape[1]}")
                    elif (embeddings[1] - embeddings[0]) < 1:
                        logging.info(f"Given audio length={(embeddings[1] - embeddings[0])}, which is too short, please give a longer audio length.")

                    this_music_audio = self.semantictoken2wav(token=this_music_token)

            yield {'music_audio': this_music_audio.cpu()}
            torch.cuda.synchronize()