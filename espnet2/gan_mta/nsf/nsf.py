#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# import core_scripts.other_tools.debug as nii_debug
import espnet2.gan_mta.nsf.block_nn as nii_nn
import espnet2.gan_mta.nsf.block_nsf as nii_nsf


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

    
#####
## Model definition
#####

## For condition module only provide Spectral feature to Filter block
class CondModule(torch_nn.Module):
    """ Conditiona module

    Upsample and transform input features
    CondModule(input_dimension, output_dimension, up_sample_rate,
               blstm_dimension = 64, cnn_kernel_size = 3)
    
    Spec, F0 = CondModule(features, F0)
    Both input features should be frame-level features

    If x doesn't contain F0, just ignore the returned F0
    """
    def __init__(self, input_dim, output_dim, up_sample, \
                 blstm_s = 64, cnn_kernel_s = 3):
        super(CondModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.blstm_s = blstm_s
        self.cnn_kernel_s = cnn_kernel_s

        # bi-LSTM
        self.l_blstm = nii_nn.BLSTMLayer(input_dim, self.blstm_s)
        self.l_conv1d = nii_nn.Conv1dKeepLength(
            self.blstm_s, output_dim, 1, self.cnn_kernel_s)

        self.l_upsamp = nii_nn.UpSampleLayer(
            self.output_dim, self.up_sample, True)
        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = nii_nn.UpSampleLayer(1, self.up_sample, False)

    def forward(self, feature, f0):
        """ spec, f0 = forward(self, feature, f0)
        feature: (batchsize, length, dim)
        f0: (batchsize, length, dim=1), which should be F0 at frame-level
        
        spec: (batchsize, length, self.output_dim), at wave-level
        f0: (batchsize, length, 1), at wave-level
        """ 
        spec = self.l_upsamp(self.l_conv1d(self.l_blstm(feature)))
        f0 = self.l_upsamp_F0(f0)
        return spec, f0

# For source module
class SourceModuleMusicNSF(torch_nn.Module):
    """ SourceModule for hn-nsf 
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1, 
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)

    Sine_source, noise_source = SourceModuleMusicNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, 
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleMusicNSF, self).__init__()
        
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = nii_nsf.SineGen(
            sampling_rate, harmonic_num, sine_amp, 
            add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch_nn.Linear(harmonic_num+1, 1)
        self.l_tanh = torch_nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleMusicNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        #  sine fundamental component and harmonic overtones
        sine_wavs, uv, _ = self.l_sin_gen(x)
        #  merge into a single excitation
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
        
        
# For Filter module
class FilterModuleMusicNSF(torch_nn.Module):
    """ Filter for Hn-NSF
    FilterModuleMusicNSF(signal_size, hidden_size, fir_coef,
                      block_num = 5,
                      kernel_size = 3, conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    fir_coef: list of FIR filter coeffs,
              (low_pass_1, low_pass_2, high_pass_1, high_pass_2)
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block


    output = FilterModuleMusicNSF(harmonic_source,noise_source,uv,context)
    harmonic_source (batchsize, length, dim=1)
    noise_source  (batchsize, length, dim=1)
    context (batchsize, length, dim)
    uv (batchsize, length, dim)

    output: (batchsize, length, dim=1)    
    """
    def __init__(self, signal_size, hidden_size, \
                 block_num = 5, kernel_size = 3, conv_num_in_block = 10):
        super(FilterModuleMusicNSF, self).__init__()        
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block

        # filter blocks for harmonic branch
        tmp = [nii_nsf.NeuralFilterBlock(
            signal_size, hidden_size, kernel_size, conv_num_in_block) \
               for x in range(self.block_num)]
        self.l_har_blocks = torch_nn.ModuleList(tmp)


    def forward(self, har_component, noi_component, condition_feat, uv):
        """
        """
        # harmonic component
        for l_har_block in self.l_har_blocks:
            har_component = l_har_block(har_component, condition_feat)
        
        output = har_component
        return output        
        

## FOR MODEL
class NSFGenerator(torch_nn.Module):
    """ Model definition
    """
    def __init__(
        self, 
        in_dim: int = 128, 
        out_dim: int = 1, 
        prj_conf_input_reso: int = 288, 
        prj_conf_wav_samp_rate: int = 24000,
        mean_std=None):
        super(NSFGenerator, self).__init__()
        # NOTE: (Xuan) removed args and prj_conf for convinience, be care of any error.
        ######
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        ######

        # configurations
        self.sine_amp = 0.1
        self.noise_std = 0.001
        
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.hidden_dim = 64

        # self.upsamp_rate = prj_conf_input_reso    # NOTE: (Xuan) for segments computation 
        self.upsample_factor = prj_conf_input_reso
        self.sampling_rate = prj_conf_wav_samp_rate

        self.cnn_kernel_size = 3
        self.filter_block_num = 5
        self.cnn_num_in_block = 10
        self.harmonic_num = 16
        
        # the three modules
        self.m_condition = CondModule(self.input_dim, \
                                      self.hidden_dim, \
                                      self.upsample_factor, \
                                      cnn_kernel_s = self.cnn_kernel_size)

        #self.m_source = SourceModuleMusicNSF(self.sampling_rate, 
        #                                     self.harmonic_num, 
        #                                     self.sine_amp, 
        #                                     self.noise_std)
        
        self.m_filter = FilterModuleMusicNSF(self.output_dim, 
                                             self.hidden_dim,\
                                             self.filter_block_num, \
                                             self.cnn_kernel_size, \
                                             self.cnn_num_in_block)
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, data_mean_std=None):
        """
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        """
        return y * self.output_std + self.output_mean
    
    def forward(self, x):
        """ definition of forward method 
        Assume x (batchsize=1, length, dim)
        Return output(batchsize=1, length)
        """
        # NOTE (Xuan): the input x in espnet is x (batchsize, dim, length)
        x = x.permute(0, 2, 1)     # -> (batchsize, length, dim)

        # normalize the data
        feat = self.normalize_input(x)

        # condition module
        #  place_holder is originally the up-sampled F0
        #  it is not used for noise-excitation model
        #  but it has the same shape as the upsampled souce signal
        #  it can help to create the noise_source below
        cond_feat, place_holder = self.m_condition(feat, x[:, :, -1:])

        with torch.no_grad():
            noise_source = torch.randn_like(place_holder) * self.noise_std / 3

        # source module
        #har_source, noi_source, uv = self.m_source(f0_upsamped)

        # filter module (including FIR filtering)
        output = self.m_filter(noise_source, None, cond_feat, None)

        # output
        return output.permute(0, 2, 1)


    def inference(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Perform inference.

        Args:
            x (torch.Tensor): Input tensor (T, in_channels).

        Returns:
            Tensor: Output tensor (T ** upsample_factor, out_channels).

        """
        x = self.forward(x.transpose(1, 0).unsqueeze(0)).squeeze(0)
        return x.transpose(1, 0)


class NSFGeneratorLoss(torch_nn.Module):
    """ Wrapper to define loss function 
    """
    def __init__(
        self,
        frame_hops: list,
        frame_lens: list,
        fft_n: list,
        amp_floor: float):
        super(NSFGeneratorLoss, self).__init__()
        # NOTE: (Xuan) removed args, be care of any error.
        """
        """
        # frame shift (number of points)
        self.frame_hops = frame_hops
        # frame length
        self.frame_lens = frame_lens
        # FFT length
        self.fft_n = fft_n
        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating
        self.amp_floor = amp_floor
        # loss
        self.loss1 = torch_nn.MSELoss()
        self.loss2 = torch_nn.MSELoss()
        self.loss3 = torch_nn.MSELoss()
        self.loss = [self.loss1, self.loss2, self.loss3]
        #self.loss = torch_nn.MSELoss()
        
    def forward(self, output_orig, target_orig):
        """ Loss().compute(output, target) should return
        the Loss in torch.tensor format
        Assume output and target as (batchsize=1, length)
        """
        # convert from (batchsize=1, length, dim=1) to (1, length)
        if output_orig.ndim == 3:
            output = output_orig.squeeze()
        else:
            output = output_orig
        if target_orig.ndim == 3:
            target = target_orig.squeeze()
        else:
            target = target_orig
        
        # compute loss
        loss = 0
        for frame_shift, frame_len, fft_p, loss_f in \
            zip(self.frame_hops, self.frame_lens, self.fft_n, self.loss):
            x_stft = torch.stft(output, fft_p, frame_shift, frame_len, \
                                window=self.win(frame_len, 
                                                device=output.device), 
                                onesided=True,
                                pad_mode="constant")
            y_stft = torch.stft(target, fft_p, frame_shift, frame_len, \
                                window=self.win(frame_len, 
                                                device=output.device), 
                                onesided=True,
                                pad_mode="constant")
            x_sp_amp = torch.log(torch.norm(x_stft, 2, -1).pow(2) + \
                                 self.amp_floor)
            y_sp_amp = torch.log(torch.norm(y_stft, 2, -1).pow(2) + \
                                 self.amp_floor)
            loss += loss_f(x_sp_amp, y_sp_amp)
        return loss


#########
## Model Discriminator definition
#########
def get_padding(kernel_size, dilation=1):
    """Function to compute the padding length for CNN layers
    """
    # L_out = (L_in + 2*pad - dila * (ker - 1) - 1) // stride + 1
    # stride -> 1
    # L_out = L_in + 2*pad - dila * (ker - 1) 
    # L_out == L_in ->
    # 2 * pad = dila * (ker - 1) 
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(torch_nn.Module):
    def __init__(self, period, 
                 kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.leaky_relu_slope = 0.1
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch_nn.ModuleList([
            norm_f(
                torch_nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(1024, 1024, (kernel_size, 1), 1, 
                                padding=(2, 0))),
        ])
        self.conv_post = norm_f(
            torch_nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        return
    
    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = torch_nn_func.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = torch_nn_func.leaky_relu(x, self.leaky_relu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch_nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = torch_nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    
class DiscriminatorS(torch_nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.leaky_relu_slope = 0.1
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch_nn.ModuleList([
            norm_f(
                torch_nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(
                torch_nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(
                torch_nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(torch_nn.Conv1d(1024, 1, 3, 1, padding=1))
        return
    
    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = torch_nn_func.leaky_relu(x, self.leaky_relu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    
class MultiScaleDiscriminator(torch_nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = torch_nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = torch_nn.ModuleList([
            torch_nn.AvgPool1d(4, 2, padding=2),
            torch_nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class NSFDiscriminator(torch_nn.Module):
    """ Model definition
    """
    def __init__(
        self, 
        in_dim: int = 128, 
        out_dim: int = 1, 
        prj_conf_input_reso: int = 288, 
        prj_conf_wav_samp_rate: int = 24000,
        mean_std=None
    ):
        super(NSFDiscriminator, self).__init__()
        self.m_mpd = MultiPeriodDiscriminator()
        self.m_msd = MultiScaleDiscriminator()

    def _feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss*2

    def _discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        return loss, r_losses, g_losses
    
    def _generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l
        return loss, gen_losses
    
    def loss_for_D(self, nat_wav, gen_wav_detached, input_feat):
        # gen_wav has been detached
        nat_wav_tmp = nat_wav.permute(0, 2, 1)
        gen_wav_tmp = gen_wav_detached.permute(0, 2, 1)
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.m_mpd(nat_wav_tmp, gen_wav_tmp)
        
        loss_disc_f, _, _ = self._discriminator_loss(y_df_hat_r, y_df_hat_g)
        
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.m_msd(nat_wav_tmp, gen_wav_tmp)
        loss_disc_s, _, _ = self._discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        return loss_disc_f + loss_disc_s

    def loss_for_G(self, nat_wav, gen_wav, input_feat):
        nat_wav_tmp = nat_wav.permute(0, 2, 1)
        gen_wav_tmp = gen_wav.permute(0, 2, 1)
        # MPD
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.m_mpd(nat_wav_tmp, 
                                                                gen_wav_tmp)
        # MSD
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.m_msd(nat_wav_tmp, 
                                                                gen_wav_tmp)

        loss_fm_f = self._feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self._feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = self._generator_loss(y_df_hat_g)
        loss_gen_s, _ = self._generator_loss(y_ds_hat_g)
        return loss_fm_f + loss_fm_s + loss_gen_f + loss_gen_s


if __name__ == "__main__":
    print("Definition of model")

    
