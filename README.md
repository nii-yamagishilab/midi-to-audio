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
python-version 3.9
pytorch: 1.8.0 / 1.9?
espnet: 0.10
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

Next, download project data and models:
work_dir: `egs2/maestro/tts1`
 * MAESTRO data: make the directory `downloads`, download from (maestro)[https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip] and unzip the dataset.
 * MIDI2WAV models: make the directory  `model_zoo`, download well-trained model weights and put them here. 


## How to use

See the scripts `warmup.sh` (warm start training), `train_from_scratch.sh` (train on VCTK data only), and `predictmel.sh` (prediction).  The scripts assume a SLURM-type computing environment.  You will need to change the paths to match your environments and point to your data.  Here are the parameters relevant to multi-speaker TTS:
 * `source-data-root` and `target-data-root`: path to your source and target preprocessed data
 * `selected-list-dir`: train/eval/test set definitions
 * `batch_size`: if you get OOM errors, try reducing the batch size
 * `use_external_speaker_embedding=True`: use speaker embeddings that you provide from a file (see the files in the `speaker_embeddings` directory)
 * `embedding_file`: path to the file containing your speaker embeddings
 * `speaker_embedding_dim`:  dimension should match the dimension in your embedding file <!-- TODO: deprecate this -->
 * `speaker_embedding_projection_out_dim=64`: We found experimentally that projecting the speaker embedding to a lower dimension helped to reduce overfitting.  You can try different values, but to use our pretrained multi-speaker models you will have to use 64.
 * `speaker_embedding_offset`: must match the ID of your first speaker.  <!-- TODO: deprecate this -->

The scripts are set up using `embedding_file="vctk-x-vector.txt",speaker_embedding_dim='200'` which is default x-vectors.  Please change it to `embedding_file="vctk-lde-3.txt",speaker_embedding_dim='512'` to use LDE embeddings from our best system.

<!-- num_speakers does not actually get used with external_embedding so TODO remove this from the scripts. -->

## Acknowledgments

This work was partially supported by a JST CREST Grant (JPMJCR18A6, VoicePersonae project), Japan, and by MEXT KAKENHI Grants (16H06302, 17H04687, 18H04120, 18H04112, 18KT0051, 19K24372), Japan. The numerical calculations were carried out on the TSUBAME 3.0 supercomputer at the Tokyo Institute of Technology.

## Licence

BSD 3-Clause License

Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
