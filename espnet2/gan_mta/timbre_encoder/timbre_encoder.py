import os
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Any
from typing import Dict

from espnet2.gan_mta.timbre_encoder.sinc_conv import SincConv
from espnet2.gan_mta.timbre_encoder.backbone import se_resnet34, resnet34
from espnet2.gan_mta.timbre_encoder.lde import LDE

class TimbreEncoder(torch.nn.Module):
    """Timbre Encoder Net (Instrument Embedding SincConv Model) 
    """
    def __init__(
        self,
        sinc_conv_params: Dict[str, Any] = {
            "out_channels": 122,
            "kernel_size": 50,
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
        encoder_type: str = "resnet34",
        lde_params: Dict[str, Any] = {
            "D": 8,
            "pooling": "mean", 
            "network_type": "lde", 
            "distance_type": "sqr",
        },
        out_channels: int = 512,
    ):
        super().__init__()

        # [sinc_conv: trans] tranfer raw audio to sinc feature
        sinc_conv_params["kernel_size"] = int(
            sinc_conv_params["kernel_size"] / 1000 * 24000)
        if not (sinc_conv_params["kernel_size"] % 2):
             sinc_conv_params["kernel_size"] += 1 
        sinc_conv_params["stride"] = int(
            sinc_conv_params["stride"] / 1000 * 24000
        )
        self.trans = SincConv(
            **sinc_conv_params
        )

        # [backbone: encoder] encode sinc feature to latent space
        # TODO (Xuan): refine here with importlib
        if encoder_type == 'se_resnet34':
            self.encoder = se_resnet34()
            _feature_dim = 128
        elif encoder_type == 'resnet34':
            self.encoder = resnet34()
            _feature_dim = 128
        else:
            raise NotImplementedError

        # [lde: pool] to aggregate time-variant feature to time-invariant feature
        lde_params["input_dim"] = _feature_dim
        self.pool = LDE(
            **lde_params
        )

        if lde_params['pooling']=='mean':
            in_channels = _feature_dim*lde_params['D']
        if lde_params['pooling']=='mean+std':
            in_channels = _feature_dim*2*lde_params['D']
        
        self.fc0 = nn.Linear(in_channels, out_channels)
        self.bn0  = nn.BatchNorm1d(out_channels)


    def forward(self, x):
        x = x.transpose(1, -1) # batch * time * channel
        x = self.trans(x)
        x = self.encoder(x)
        x = self.pool(x)
        if type(x) is tuple:
            x = x[0]
        feat = self.fc0(x)
        return feat


