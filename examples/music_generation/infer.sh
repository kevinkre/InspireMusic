#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
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
. ./path.sh || exit 1;

export TOKENIZERS_PARALLELISM=False

model_name="InspireMusic-Base"
pretrained_model_dir=../../pretrained_models/${model_name}
dataset_name=musiccaps_dev

# inference
echo "Run inference."
expr_name="inspiremusic_${dataset_name}"
for mode in sft; do
  python inspiremusic/bin/inference.py --mode $mode \
      --gpu 1 \
      --config conf/inspiremusic_base_fp16.yaml \
      --prompt_data data/${dataset_name}/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_model_dir/llm.pt \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --chorus verse \
      --min_generate_audio_seconds 5 \
      --max_generate_audio_seconds 30 \
      --result_dir `pwd`/exp/${model_name}/${mode}_${expr_name}
  echo `pwd`/exp/${model_name}/${mode}_${expr_name}
done
