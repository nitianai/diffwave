# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch.nn as nn


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    lr_decay=0.999875,
    max_grad_norm=None,

    loss_fn=nn.L1Loss(),
    cuda_id=0,
    audiofile='2.wav',
    checkpoint_file='model-ldm-res{epoch:04d}',
    restore_file='checkpoints/model-l1-epoch=0034.ckpt',

    # loss_fn=nn.MSELoss(),
    # cuda_id=1,
    # audiofile='2.wav',
    # checkpoint_file='model-mse-{epoch:04d}',
    # restore_file='checkpoints/model-mse-epoch=0034.ckpt',


    # Data params
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_samples=320,
    crop_mel_frames=64,  # Probably an error in paper.

    # Model params
    residual_layers=30,
    residual_channels=256,
    dilation_cycle_length=10,
    diffusion_step_dim=512,
    latent_dim=256,
    conformer_layers=6,
    unconditional = False,
    noise_schedule=np.linspace(1e-4, 0.09, 50).tolist(),
    # noise_schedule=np.linspace(1e-4, 0.09, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    #vae params
    hidden_channels=192,
    gin_channels=768,
    ssl_channels=768,
    n_flow_layers=4,
    flow_kernel_size=5,
    n_posterior_layers=16,
    posterior_kernel_size=5,
    n_attention_heads=2,
    n_attention_layers=6,
    attention_kernel_size=3,
    p_dropout=0.1,

    # unconditional sample len
    audio_len = 22050*5, # unconditional_synthesis_samples

    #rvq parameters
    weights_filename='model_weights/rvq_model_weights.pth',
    dim=768,
    codebook_size=1024,
    codebook_dim=768,
    num_quantizers=8,
    ema_update = False,
    learnable_codebook = True,

    #soGPT parameters
    phoneme_vocab_size=180,
    soGPT_nhead=16,
    soGPT_nLayer=12,
)
