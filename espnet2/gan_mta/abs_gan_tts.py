# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based TTS abstrast class."""

from abc import ABC
from abc import abstractmethod

from typing import Dict
from typing import Union

import torch

from espnet2.mta.abs_tts import AbsTTS


class AbsGANTTS(AbsTTS, ABC):
    """GAN-based TTS model abstract class."""

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """Return generator or discriminator loss."""
        raise NotImplementedError
