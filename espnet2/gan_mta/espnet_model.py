# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based midi-to-audio ESPnet model."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Any
from typing import Dict
from typing import Optional

import torch

from typeguard import check_argument_types

from espnet2.gan_mta.abs_gan_tts import AbsGANTTS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet2.mta.feats_extract.abs_feats_extract import AbsFeatsExtract

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch < 1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetGANTTSModel(AbsGANESPnetModel):
    """ESPnet model for GAN-based midi-to-audio task."""

    def __init__(
        self,
        feats_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_extract: Optional[AbsFeatsExtract],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_extract: Optional[AbsFeatsExtract],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        tts: AbsGANTTS,
    ):
        """Initialize ESPnetGANTTSModel module."""
        assert check_argument_types()
        super().__init__()
        self.feats_extract = feats_extract
        self.normalize = normalize
        self.pitch_extract = pitch_extract
        self.pitch_normalize = pitch_normalize
        self.energy_extract = energy_extract
        self.energy_normalize = energy_normalize
        self.tts = tts
        assert hasattr(
            tts, "generator"
        ), "generator module must be registered as tts.generator"
        if self.tts.use_discriminator:
            assert hasattr(
                tts, "discriminator"
            ), "discriminator module must be registered as tts.discriminator"

    def forward(
        self,
        midi: torch.Tensor,
        midi_lengths: torch.Tensor,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
    ) -> Dict[str, Any]:
        """Return generator or discriminator loss with dict format.

        Args:
            midi (Tensor): midi index tensor (B, T_midi).
            midi_lengths (Tensor): midi length tensor (B,).
            audio (Tensor): audio waveform tensor (B, T_wav).
            audio_lengths (Tensor): audio length tensor (B,).
            duration (Optional[Tensor]): Duration tensor.
            duration_lengths (Optional[Tensor]): Duration length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        with autocast(False):
            # Extract features
            feats = None
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(
                    audio,
                    audio_lengths,
                )
            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )
            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Make batch for tts inputs
        # MIDI downsample
        downsample_rate = 4
        midi = midi[:, ::downsample_rate, :]
        midi_lengths = midi_lengths // downsample_rate

        # Make batch for tts inputs
        batch = dict(
            midi=midi,
            midi_lengths=midi_lengths,
            forward_generator=forward_generator,
        )

        # Update batch for additional auxiliary inputs
        if feats is not None:
            batch.update(feats=feats, feats_lengths=feats_lengths)
        if self.tts.require_raw_speech:
            batch.update(audio=audio, audio_lengths=audio_lengths)
        if durations is not None:
            batch.update(durations=durations, durations_lengths=durations_lengths)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy, energy_lengths=energy_lengths)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        return self.tts(**batch)

    def collect_feats(
        self,
        midi: torch.Tensor,
        midi_lengths: torch.Tensor,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Calculate features and return them as a dict.

        Args:
            midi (Tensor): midi index tensor (B, T_midi).
            midi_lengths (Tensor): midi length tensor (B,).
            audio (Tensor): audio waveform tensor (B, T_wav).
            audio_lengths (Tensor): audio length tensor (B, 1).
            durations (Optional[Tensor): Duration tensor.
            durations_lengths (Optional[Tensor): Duration length tensor (B,).
            pitch (Optional[Tensor): Pitch tensor.
            pitch_lengths (Optional[Tensor): Pitch length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker index tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).

        Returns:
            Dict[str, Tensor]: Dict of features.

        """
        feats = None
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(
                audio,
                audio_lengths,
            )
        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                speech,
                speech_lengths,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
            )
        if self.energy_extract is not None:
            energy, energy_lengths = self.energy_extract(
                speech,
                speech_lengths,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
            )

        # store in dict
        feats_dict = {}
        if feats is not None:
            feats_dict.update(feats=feats, feats_lengths=feats_lengths)
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)

        return feats_dict


    def inference(
        self,
        midi: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        **decode_config,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

        Args:
            midi (Tensor): Text index tensor (T_text).
            audio (Tensor): Speech waveform tensor (T_wav).
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            durations (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            energy (Optional[Tensor): Energy tensor.

        Returns:
            Dict[str, Tensor]: Dict of outputs.

        """
        # NOTE (Xuan): hard code here, for spec input
        downsample_rate = 4
        midi = midi[::downsample_rate, :]
        # midi = self.normalize(midi[None])[0][0]
        input_dict = dict(midi=midi)
        if decode_config["use_teacher_forcing"] or getattr(self.tts, "use_gst", False):
            if audio is None:
                raise RuntimeError("missing required argument: 'audio'")
            if self.feats_extract is not None:
                feats = self.feats_extract(audio[None])[0][0]
            else:
                # Use precalculated feats (feats_type != raw case)
                feats = audio
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            input_dict.update(feats=feats)
            # NOTE: (Xuan) hard code here, put them back in other exps
            # if self.tts.require_raw_speech:
            #     input_dict.update(audio=audio)

        if decode_config["use_teacher_forcing"]:
            if durations is not None:
                input_dict.update(durations=durations)

            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    audio[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.pitch_normalize is not None:
                pitch = self.pitch_normalize(pitch[None])[0][0]
            if pitch is not None:
                input_dict.update(pitch=pitch)

            if self.energy_extract is not None:
                energy = self.energy_extract(
                    audio[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.energy_normalize is not None:
                energy = self.energy_normalize(energy[None])[0][0]
            if energy is not None:
                input_dict.update(energy=energy)

        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        output_dict = self.tts.inference(**input_dict, **decode_config)

        if self.normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)
        return output_dict
