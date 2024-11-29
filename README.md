# InspireMusic

[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=Inspire%20Music🎶&text2=🤗%20A%20Fundamental%20Music%20Song%20Audio%20Generation%20Toolkit&width=800&height=210)](https://github.com/Akshay090/svg-banners)

InspireMusic is a PyTorch library designed for research on music, song, and audio generation. It provides both inference and training code for AI generative models that produce high-quality music. The library features a unified framework that incorporates autoregressive Transformer and conditional flow-matching modeling, enabling controllable generation of music, songs, and audio with textual and music structural conditioning, along with neural audio tokenizers. Currently, InspireMusic supports tasks like text-to-music generation, with plans to extend capabilities to text-to-song and text-to-audio generation in the future.

[![code](https://img.shields.io/badge/InspireMusic-Code-green.svg)](https://github.com/FunAudioLLM/InspireMusic)
[![demo](https://img.shields.io/badge/InspireMusic-Demo-brightgreen.svg)](https://iris2c.github.io/InspireMusic)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20InspireMusic-Models-orange.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-Base)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-lightgrey.svg)](https://arxiv.org/abs/)

### 👉🏻 [InspireMusic Demos](https://iris2c.github.io/InspireMusic) 👈🏻

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

For `CosyVoice`, visit [CosyVoice repo](https://github.com/FunAudioLLM/CosyVoice) and [CosyVoice space](https://www.modelscope.cn/studios/iic/CosyVoice).

<a name="Highligts"></a>
## Highlights
**InspireMusic** focuses on music generation, song generation, audio generation.
- A unified framework for music, song, audio genereation.
- Controllable with text prompts, music genres, music structures, etc.
- Convenient Finetuning and Inference: Provide convenient finetuning and inference scripts and strategies, allowing users to easily their own music generation models according to different scenarios.
- High audio quality


<a name="What's News"></a>
## What's New 🔥

[//]: # (- 2024/12: The [InspireMusic-Base]&#40;https://huggingface.co/FunAudioLLM/InspireMusic-Base&#41; voice understanding model is open-sourced, which offers high quality, diverse text style, music structure, music genre control capability.  )
- 2024/11: We are excited to open-source the [Code repo](https://github.com/FunAudioLLM/InspireMusic) and [Demos](https://iris2c.github.io/InspireMusic) of **InspireMusic**. The **InspireMusic**, a unified framework of music, song, audio generation, with text-to-music, music structure, music genre and timestamps control. InspireMusic excels in music generation and instruction-following capabilities. 

## Installation

### Clone

- Clone the repo
``` sh
git clone --recursive https://github.com/FunAudioLLM/InspireMusic.git
# If you failed to clone submodule due to network failures, please run following command until success
cd InspireMusic
git submodule update --init --recursive
```

### Install
InspireMusic requires Python 3.8, PyTorch 2.1.0. To install InspireMusic, you can run one of the following:

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
``` sh
conda create -n inspiremusic python=3.8
conda activate inspiremusic
cd InspireMusic
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# install flash attention to speedup training
pip install flash-attn --no-build-isolation
```

- Install within the package:
```sh
cd InspireMusic
# You can run to install the packages
python setup.py install
pip install flash-attn --no-build-isolation
```

We also recommend having `sox` or `ffmpeg` installed, either through your system or Anaconda:
```sh
# # Install sox
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel

# Install ffmpeg
# ubuntu
sudo apt-get install ffmpeg
# centos
sudo yum install ffmpeg
```

## Models
### Download Model

We strongly recommend that you download our pretrained `InspireMusic model`.

If you are expert in this field, and you are only interested in training your own InspireMusic model from scratch, you can skip this step.

``` sh
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/InspireMusic.git pretrained_models/InspireMusic-Base
```

### Available Models
Currently, we open source the music generation models only supporting 24000Hz mono channel audio. 
The table below presents the links to the ModelScope and Huggingface model hub. More models will be available soon.

| Model name            | Model Links                                                                                                                                                                               | Remarks                                        |
|------------------------------|-------------------------------------------------------|-------------|
| InspireMusic-Base     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-Base) [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]() | Pre-trained Music Generation Model, 24kHz mono |
| InspireMusic-1.5B     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Music Generation 1.5B Model, 24kHz mono       |
| InspireSong-Base      | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Song Generation Base Model, 24kHz mono         |
| InspireSong-1.5B      | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Song Generation 1.5B Model, 24kHz mono         |
| InspireAudio-1.5B     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                      | Pre-trained Audio Generation 1.5B Model, 24kHz mono        |

## Basic Usage

At the moment, InspireMusic contains the training code and inference code for Music Generation. More tasks such as song generation and audio generation will be supported in future.

### Quick Start

Here is a quick start running script to do music generation task including data preparation pipeline, model training, inference. 
``` sh
cd InspireMusic/examples/music_generation/
bash run.sh
```

### Training

Here is an example to train LLM model. 
```sh
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=1024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    inspiremusic/bin/train.py \
    --train_engine "torch_ddp" \
    --config conf/inspiremusic.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model llm \
    --model_dir `pwd`/exp/music_generation/llm/ \
    --tensorboard_dir `pwd`/tensorboard/music_generation/llm/ \
    --ddp.dist_backend "nccl" \
    --num_workers 8 \
    --prefetch 100 \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer \
    --fp16
```

Here is an example code to train flow matching model. 
```sh
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=1024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    inspiremusic/bin/train.py \
    --train_engine "torch_ddp" \
    --config conf/inspiremusic.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model flow \
    --model_dir `pwd`/exp/music_generation/flow/ \
    --tensorboard_dir `pwd`/tensorboard/music_generation/flow/ \
    --ddp.dist_backend "nccl" \
    --num_workers 8 \
    --prefetch 100 \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer \
    --fp16
```

### Inference

Here is an example script to quickly do model inference.
``` sh
cd InspireMusic/examples/music_generation/
bash infer.sh
```

Here is an example code to run inference with flow matching model.
```sh
pretrained_model_dir = "./pretrained_models/InspireMusic/"
python inspiremusic/bin/inference.py --mode sft \
      --gpu 0 \
      --config conf/inspiremusic.yaml \
      --prompt_data data/test/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_model_dir/llm.pt \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --result_dir `pwd`/exp/inspiremusic/sft_test \
      --chorus verse \
      --min_generate_audio_seconds 8 \
      --max_generate_audio_seconds 30 
```

Here is an example code to run inference without flow matching model.
```sh
pretrained_model_dir = "./pretrained_models/InspireMusic/"
python inspiremusic/bin/inference.py --mode sft \
      --gpu 0 \
      --config conf/inspiremusic.yaml \
      --prompt_data data/test/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_model_dir/llm.pt \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --no_flow_mode True \
      --result_dir `pwd`/exp/inspiremusic/sft_test \
      --chorus verse \
      --min_generate_audio_seconds 8 \
      --max_generate_audio_seconds 30 
```

## Roadmap

- [ ] 2024/12
  - [ ] 75Hz InspireMusic base model for music generation
  - [ ] Support song generation task
    
- [ ] 2025/01
    - [ ] 75Hz InspireSong model for song generation

- [ ] 2025/02
    - [ ] Support audio generation task 
    - [ ] 75Hz InspireMusic-1.5B model for music generation

- [ ] 2025/03
    - [ ] 75Hz InspireAudio model for music and audio generation

- [ ] TBD

    - [ ] 25Hz InspireMusic model
    - [ ] Support 48kHz stereo audio
    - [ ] Streaming inference mode support
    - [ ] Support more instruction mode, multi-lingual instructions
    - [ ] InspireSong trained with more multi-lingual data
    - [ ] More...


## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/InspireMusic/issues).

You can also scan the QR code to join our official chat group.

<p align="center">
  <img alt="Light" src="./asset/dingding.png" width="30%" />
  <img alt="Light" src="./asset/dingtalk.png" width="30%" /> 
  <img alt="Light" src="./asset/QR.jpg" width="25%" />
</p>

## Acknowledge

1. We borrowed a lot of code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).
3. We borrowed a lot of code from [WavTokenizer](https://github.com/jishengpeng/WavTokenizer).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
6. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
7. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
9. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
