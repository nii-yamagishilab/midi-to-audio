# CAN KNOWLEDGE OF END-TO-END TEXT-TO-SPEECH MODELS IMPROVE NEURAL MIDI-TO-AUDIO SYNTHESIS SYSTEMS?

This is the implementation for our paper submitted to ICASSP 2023:
"CAN KNOWLEDGE OF END-TO-END TEXT-TO-SPEECH MODELS IMPROVE NEURAL MIDI-TO-AUDIO SYNTHESIS SYSTEMS?"

Xuan Shi, Erica Cooper, Xin Wang, Junichi Yamagishi, Shrikanth Narayanan

The audio samples are uploaded to this website: https://nii-yamagishilab.github.io/sample-midi-to-audio/. 

It is appreciated if you can cite this paper when the idea, code, and pretrained model are helpful to your research.

The code for model training was based on the [ESPnet-TTS project](https://github.com/espnet/espnet):
"ESPnet-TTS: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit," ICASSP 2020
Tomoki Hayashi, Ryuichi Yamamoto, Katsuki Inoue, Takenori Yoshimura, Shinji Watanabe, Tomoki Toda, Kazuya Takeda, Yu Zhang, and Xu Tan

The data for all experiments (training and inference) is the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset:
"Enabling factorized piano music modeling and generation with the MAESTRO dataset," ICLR 2019
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, JesseEngel, and Douglas Eck

This work consists of a MIDI-to-mel component based on **Transformer-TTS**:
"Neural speech synthesis with transformer network," AAAI 2019
Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, and Ming Liu
and a **HiFiGAN**-based mel-to-audio component:
"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis," NeurIPS 2020
Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae
The two components were first separately trained, and then jointly fine-tuned for an additional 200K steps.


## How to use

### Installment

#### ESPnet-based Python Environment Setup

It is recommended to follow the official [installation](https://espnet.github.io/espnet/installation.html) to set up a complete [ESPnet2](https://github.com/espnet/espnet) environment for model training.

1. Setup kaldi

```
$ cd <midi2wav-root>/tools
$ ln -s <kaldi-root> .
```

2. Setup Python environment. There are 4 types of setup method, we strongly suggest the first one.
```
$ cd <midi2wav-root>/tools
$ CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
$ ./setup_anaconda.sh ${CONDA_TOOLS_DIR} [conda-env-name] [python-version]
# e.g.
$ ./setup_anaconda.sh ${CONDA_TOOLS_DIR} midi2wav 3.9
```

3. Install ESPnet
```
$ cd <midi2wav-root>/tools
$ make TH_VERSION=1.8 CUDA_VERSION=11.1
```
Make sure the espnet version is `espnet==0.10`.

#### Python Environment Revision

After the `midi2wav` conda environment set up, we will install packages for music processing and re-install some packages to avoid potential package version conflicts. 

Suggested dependencies:
```
pretty_midi==0.2.9
wandb==0.12.9
protobuf==3.19.3
```

#### Pre-trained model preparation

First, make the directory to save the pre-trained model.

```
$ cd <midi2wav-root>/egs2/maestro/tts1
$ mkdir -p exp/tts_finetune_joint_transformer_hifigan_raw_proll
$ cd exp/tts_finetune_joint_transformer_hifigan_raw_proll
```
Then, download the pre-trained model from [Zenodo](https://zenodo.org/record/7439325#.Y5pcAi8Rr0o), rename the model as `train.loss.ave.pth`, and save it under the directory mentioned above.


### Code analysis

The midi-to-wav scripts are developed based on kaldi-style ESPnet.  The main work directory is at `<midi2wav-root>/egs2/maestro/tts1`.

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
The code is licensed under Apache License Version 2.0, following ESPnet.
The pretrained model is licensed under the Creative Commons License:
Attribution 4.0 International
http://creativecommons.org/licenses/by/4.0/legalcode 

