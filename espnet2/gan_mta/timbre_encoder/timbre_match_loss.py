import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.gan_mta.timbre_encoder.timbre_encoder import TimbreEncoder

# TODO (Xuan)
class TimbreMatchLoss(torch.nn.Module):
    """Timbre matching loss module."""

    def __init__(
        self,
        timbre_encoder_params: Dict[str, Any] = {
            "sinc_conv_params": {
                "out_channels": 122,
                "kernel_size": 50,
                "input_shape": None,
                "in_channels": 1,
                "stride": 12,
                "dilation": 1,
                "padding": "same",
                "padding_mode": "reflect",
                "sample_rate": 16000,
                "min_low_hz": 5,
                "min_band_hz": 5,
                "init_type": "midi",
                "requires_grad": True,
            },
            "encoder_type": "resnet34",
            "lde_params": {
                "D": 8,
                "pooling": "mean", 
                "network_type": "lde", 
                "distance_type": "sqr",
            },
            "out_channels": 512,
        },
        timbre_encoder_pretrained: str = None,
    ):
        """Initialize TimbreMatchLoss module.

        Args:
            # TODO (Xuan): add comments on args, for example:
            average_by_layers (bool): Whether to average the loss by the number
                of layers.
            
        """
        super().__init__()
        self.encoder = TimbreEncoder(**timbre_encoder_params)

        if os.path.isfile(timbre_encoder_pretrained):
            checkpoint = torch.load(timbre_encoder_pretrained, 
                map_location=lambda storage, loc: storage)
            new_ckpt = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'fc11' in k or 'fc12' in k:
                    continue
                elif 'res.' in k:
                    k = k.replace('res.', '')
                new_ckpt[k] = v
            self.encoder.load_state_dict(new_ckpt)


    def forward(
        self,
        audio_hat: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate timbre matching loss.

        Args:
            audio_hat (torch.Tensor) [batch, 1, audio_length]: synthesised audio
            audio (torch.Tensor) [batch, 1, audio_length]: groundtruth audio

        Returns:
            Tensor [batch, feat]: Timbre matching loss value.

        """
        self.encoder.eval()
        target = torch.ones(audio.shape[0]).to(audio.device)
        timbre_emb_hat = self.encoder(audio_hat)
        with torch.no_grad():
            timbre_emb = self.encoder(audio)
        timbre_match_loss = F.cosine_embedding_loss(timbre_emb_hat, timbre_emb, target, margin=0.1)
        return timbre_match_loss


    def inference(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate timbre embedding.

        Args:
            audio (torch.Tensor) [batch, 1, audio_length]: input audio
            
        Returns:
            Tensor [batch, feat]: generated timbre embedding.

        """
        audio = audio.transpose(1, 0).unsqueeze(0)
        self.encoder.eval()
        with torch.no_grad():
            timbre_emb = self.encoder(audio)
            
        return timbre_emb
