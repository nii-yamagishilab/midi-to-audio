# CAN KNOWLEDGE OF END-TO-END TEXT-TO-SPEECH MODELS IMPROVE NEURAL MIDI-TO-AUDIO SYNTHESIS SYSTEMS?

This is the pretrained model for our paper submitted to ICASSP 2023:
"CAN KNOWLEDGE OF END-TO-END TEXT-TO-SPEECH MODELS IMPROVE NEURAL MIDI-TO-AUDIO SYNTHESIS SYSTEMS?" by Xuan Shi, Erica Cooper, Xin Wang, Junichi Yamagishi, Shrikanth Narayanan

Please cite this paper if you use this pretrained model.
This pretrained model goes with the code found here:
https://github.com/nii-yamagishilab/midi-to-audio

See the following part of the codebase's README for more information about dependencies etc.

The code for training this model was based on the [ESPnet-TTS project](https://github.com/espnet/espnet):
"ESPnet-TTS: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit," ICASSP 2020
Tomoki Hayashi, Ryuichi Yamamoto, Katsuki Inoue, Takenori Yoshimura, Shinji Watanabe, Tomoki Toda, Kazuya Takeda, Yu Zhang, and Xu Tan

The data used to train this model was trained using the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset:
"Enabling factorized piano music modeling and generation with the MAESTRO dataset," ICLR 2019
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, JesseEngel, and Douglas Eck

This model consists of a MIDI-to-mel component based on **Transformer-TTS**:
"Neural speech synthesis with transformer network," AAAI 2019
Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, and Ming Liu
and a **HiFiGAN**-based mel-to-audio component:
"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis," NeurIPS 2020
Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae
The two components were first separately trained, and then jointly fine-tuned for an additional 200K steps.


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
 * MIDI2WAV models: make the directory  `exp/tts_finetune_joint_transformer_hifigan_raw_proll`, download well-trained model from [Zenodo](https://zenodo.org/record/7370009#.Y4QaQi8Rr0o), rename the model as `train.loss.ave.pth`, and put it under the directory.


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

Model inference (Synthesizer or Joint training):
`./run.sh --stage 7 --stop_stage 7 --skip_data_prep true --ngpu ${num_gpu} --tts_task gan_mta --train_config ./conf/tuning/finetune_joint_transformer_hifigan.yaml `

## ACKNOWLEDGMENTS
This study is supported by the Japanese-French joint national project called
VoicePersonae, JST CREST (JPMJCR18A6, JPMJCR20D3), MEXT KAKENHI Grants
(21K17775, 21H04906, 21K11951), Japan, and Google AI for Japan program.

## COPYING
This pretrained model is licensed under the Creative Commons License:
Attribution 4.0 International
http://creativecommons.org/licenses/by/4.0/legalcode 
Please see `LICENSE.txt` for the terms and conditions of this pretrained model.

