import os
import sys
import torchaudio
import time
import logging

from modelscope import snapshot_download
from inspiremusic.cli.inspiremusic import InspireMusic
from inspiremusic.utils.file_utils import logging
import torch
from inspiremusic.utils.audio_utils import trim_audio, fade_out
from transformers import AutoModel

def set_env_variables():
	os.environ['PYTHONIOENCODING'] = 'UTF-8'
	os.environ['TOKENIZERS_PARALLELISM'] = 'False'
	current_working_dir = os.getcwd()
	main_root = os.path.realpath(os.path.join(current_working_dir, '../../'))
	bin_dir = os.path.join(main_root, 'inspiremusic')
	third_party_matcha_tts_path = os.path.join(main_root, 'third_party', 'Matcha-TTS')
	python_path = f"{main_root}:{bin_dir}:{third_party_matcha_tts_path}:{os.environ.get('PYTHONPATH', '')}"
	os.environ['PYTHONPATH'] = python_path
	sys.path.extend([main_root, third_party_matcha_tts_path])

class InspireMusicUnified:
	def __init__(self,
				 model_dir: str = "../../pretrained_models/InspireMusic-1.5B-Long",
				 model_name: str = "InspireMusic-1.5B-Long",
				 min_generate_audio_seconds: float = 10.0,
				 max_generate_audio_seconds: float = 30.0,
				 sample_rate: int = 24000,
				 output_sample_rate: int = 48000,
				 load_jit: bool = True,
				 load_onnx: bool = False,
				 fast: bool = False,
				 fp16: bool = True,
				 gpu: int = 0,
				 result_dir: str = "exp/inspiremusic",
				 ):
		os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
		self.model_dir = model_dir
		self.sample_rate = sample_rate

		if fast:
			self.output_sample_rate = 24000
		else:
			self.output_sample_rate = output_sample_rate

		if not os.path.exists(model_dir):
			self.model_dir = snapshot_download(f"iic/{model_name}", cache_dir=model_dir)

		self.min_generate_audio_seconds = min_generate_audio_seconds
		self.max_generate_audio_seconds = max_generate_audio_seconds
		self.min_generate_audio_length = int(self.output_sample_rate * self.min_generate_audio_seconds)
		self.max_generate_audio_length = int(self.output_sample_rate * self.max_generate_audio_seconds)
		assert self.min_generate_audio_seconds <= self.max_generate_audio_seconds, "Min audio seconds must be less than or equal to max audio seconds"

		use_cuda = gpu >= 0 and torch.cuda.is_available()
		self.device = torch.device('cuda' if use_cuda else 'cpu')
		self.model = InspireMusic(self.model_dir, load_jit=load_jit, load_onnx=load_onnx, fast=fast, fp16=fp16)

		os.makedirs(result_dir, exist_ok=True)
		self.result_dir = result_dir

		logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	@torch.inference_mode()
	def inference(self,
				  text: str = "Create an original electronic music track, incorporating uplifting and captivating melodies.",
				  task: str = 'text-to-music',
				  audio_prompt: str = None, # audio prompt file path
				  chorus: str = "intro",
				  time_start: float = 0.0,
				  time_end: float = 30.0,
				  output_fn: str = "output_audio",
				  max_audio_prompt_length: float = 5.0,
				  fade_out_duration: float = 1.0,
				  output_format: str = "wav",
				  fade_out_mode: bool = True,
				  trim: bool = False,
				  ):

		with torch.no_grad():
			text_prompt = f"<|{time_start}|><|{chorus}|><|{text}|><|{time_end}|>"
			chorus_dict = {"random": torch.randint(1, 5, (1,)).item(), "intro" : 0, "verse": 1, "chorus": 2, "outro": 4}
			chorus = chorus_dict.get(chorus, 1)
			chorus = torch.tensor([chorus], dtype=torch.int).to(self.device)

			time_start_tensor = torch.tensor([time_start], dtype=torch.float64).to(self.device)
			time_end_tensor = torch.tensor([time_end], dtype=torch.float64).to(self.device)

			music_fn = os.path.join(self.result_dir, f'{output_fn}.{output_format}')

			bench_start = time.time()

			if task == 'text-to-music':
				model_input = {
					"text"           : text,
					"audio_prompt"   : audio_prompt,
					"time_start"     : time_start_tensor,
					"time_end"       : time_end_tensor,
					"chorus"         : chorus,
					"task"           : task,
					"stream"         : False,
					"duration_to_gen": self.max_generate_audio_seconds,
					"sr"             : self.sample_rate
				}
			elif task == 'continuation':
				if audio_prompt is not None:
					audio, _ = process_audio(audio_prompt, self.sample_rate)
					if audio.size(1) < self.sample_rate:
						logging.warning("Warning: Input prompt audio length is shorter than 1s. Please provide an appropriate length audio prompt and try again.")
						audio = None
					else:
						max_audio_prompt_length_samples = int(max_audio_prompt_length * self.sample_rate)
						audio = audio[:, :max_audio_prompt_length_samples]  # Trimming prompt audio

				model_input = {
					"text"           : text,
					"audio_prompt"   : audio,
					"time_start"     : time_start_tensor,
					"time_end"       : time_end_tensor,
					"chorus"         : chorus,
					"task"           : task,
					"stream"         : False,
					"duration_to_gen": self.max_generate_audio_seconds,
					"sr"             : self.sample_rate
				}

			music_audios = []
			for model_output in self.model.cli_inference(**model_input):
				music_audios.append(model_output['music_audio'])

			bench_end = time.time()

			if trim:
				music_audio = trim_audio(music_audios[0],
										 sample_rate=self.output_sample_rate,
										 threshold=0.05,
										 min_silence_duration=0.8)
			else:
				music_audio = music_audios[0]

			if music_audio.shape[0] != 0:
				if music_audio.shape[1] > self.max_generate_audio_length:
					music_audio = music_audio[:, :self.max_generate_audio_length]

				if music_audio.shape[1] >= self.min_generate_audio_length:
					try:
						if fade_out_mode:
							music_audio = fade_out(music_audio, self.output_sample_rate, fade_out_duration)

						music_audio = music_audio.repeat(2, 1)

						if output_format in ["wav", "flac"]:
							torchaudio.save(music_fn, music_audio,
											sample_rate=self.output_sample_rate,
											encoding="PCM_S",
											bits_per_sample=24)
						elif output_format in ["mp3", "m4a"]:
							torchaudio.backend.sox_io_backend.save(
								filepath=music_fn, src=music_audio,
								sample_rate=self.output_sample_rate,
								format=output_format)
						else:
							logging.info("Format is not supported. Please choose from wav, mp3, m4a, flac.")

					except Exception as e:
						logging.error(f"Error saving file: {e}")
						raise

				audio_duration = music_audio.shape[1] / self.output_sample_rate
				rtf = (bench_end - bench_start) / audio_duration
				logging.info(f"Processing time: {int(bench_end - bench_start)}s, audio length: {int(audio_duration)}s, rtf: {rtf}, text prompt: {text_prompt}")

			else:
				logging.error(f"Generated audio length is shorter than minimum required audio length.")

if __name__ == "__main__":
	set_env_variables()
	model = InspireMusicUnified("../../pretrained_models/InspireMusic-1.5B-Long")
	model.inference("Create an original electronic music track, incorporating uplifting and captivating melodies.", 'text-to-music')