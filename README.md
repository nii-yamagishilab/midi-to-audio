# CAN KNOWLEDGE OF END-TO-END TEXT-TO-SPEECH MODELS IMPROVE NEURAL MIDI-TO-AUDIO SYNTHESIS SYSTEMS?

This is an implementation of our paper submitted to ICASSP 2023:  
"CAN KNOWLEDGE OF END-TO-END TEXT-TO-SPEECH MODELS IMPROVE NEURAL MIDI-TO-AUDIO SYNTHESIS SYSTEMS?," by Xuan Shi, Erica Cooper, Xin Wang, Junichi Yamagishi, and Shrikanth Narayanan.  
Please cite this paper if you use this code.

Audio samples can be found here:  https://github.com/nii-yamagishilab-visitors/sample-midi-to-audio (placehoder)

## News:
 * 2022-11-16: Open Source Code for midi-to-audio synthesis.

## Dependencies:  

It is recommended to follow the official [installation](https://espnet.github.io/espnet/installation.html) to set up a complete [ESPnet2](https://github.com/espnet/espnet) environment for model training.

Suggested dependencies:
```
python-version==3.9
pytorch==1.8.0
espnet==0.10
pretty_midi==0.2.9
wandb==0.12.9
protobuf==3.19.3
```

Steps:
1. Setup kaldi
```
$ cd <midi2wav-root>/tools
$ ln -s <kaldi-root> .
```

2. Setup Python environment. There are 4 types of setup method, we strongly suggest the first one.
```
$ cd <espnet-root>/tools
$ ./setup_anaconda.sh [output-dir-name|default=venv] [conda-env-name|default=root] [python-version|default=none]
# e.g.
$ ./setup_anaconda.sh /home/smg/v-xuanshi/anaconda3/ midi2wav_oc 3.9
$ ./setup_anaconda.sh ${anaconda_dir} midi2wav_oc 3.9
```

3. Install ESPnet
```
$ cd <midi2wav-root>/tools
$ make TH_VERSION=1.8 CUDA_VERSION=11.1
```

Next, download models:
work_dir: `egs2/maestro/tts1`
 * MIDI2WAV models: make the directory  `model_zoo`, download well-trained model weights from (placeholder) and put them here. 


## How to use

### Code analysis
The scripts are developed based on kaldi-style ESPnet.

There are 7 main stages:
* 1~4:  Data Preparation
* 5: Stats Collection
* 6: Model Training
* 7: Model Inference

### Scripts to run experiments

Experimental environment setting:
`./run.sh --stage 1 --stop_stage 5 --ngpu ${num_gpu} --tts_task mta --train_config ./conf/train.yaml`

Model training (Acoustic Model):
`./run.sh --stage 6 --stop_stage 6 --ngpu ${num_gpu} --tts_task mta --train_config ./conf/train.yaml`

Model training (Synthesizer or Joint training):
`./run.sh --stage 6 --stop_stage 6 --ngpu ${num_gpu} --tts_task gan_mta --train_config ./conf/tuning/finetune_joint_transformer_hifigan.yaml`

Model inference (Acoustic Model):
`./run.sh --stage 7 --stop_stage 7 --skip_data_prep true --ngpu ${num_gpu} --tts_task mta --train_config ./conf/.yaml`

Model inference (Synthesizer or Joint training):
`./run.sh --stage 7 --stop_stage 7 --skip_data_prep true --ngpu ${num_gpu} --tts_task gan_mta --train_config ./conf/tuning/finetune_joint_transformer_hifigan.yaml`

## Acknowledgments

This study is partially supported by the Japanese-French joint national project called VoicePersonae supported by JST CREST (JPMJCR18A6, JPMJCR20D3), MEXT KAKENHI Grants (21K17775, 21H04906, 21K11951), Japan, and Google AI for Japan program.

## Licence

BSD 3-Clause License

Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
