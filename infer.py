import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm  # Import tqdm for the progress bar
from train_diff import ssl2latent
from module.params import params
from module.vae import VAE
from train_stft import AudioVQVAE_lightning
from train_ldm import LatentDiffusion
from DUMMY2.dataset import AudioFeatureDataset, collate_fn
import numpy as np
import math
from module.vq import VectorQuantizer
from module.modules import MelStyleEncoder

def slice(wav, lengths, crop):
    wavs = []
    for i in range(wav.shape[0]):
        length = lengths[i]
        if length >= crop:
            start = torch.randint(0, length - crop + 1, (1,)).item()
            end = start + crop
            _wav = wav[i, :, start:end]
            wavs.append(_wav)
    wavs = torch.stack(wavs)
    return wavs

def put_noise2latent(latent, t):
    beta = np.array(params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32)).to(latent.device)

    t = torch.full([1], t).to(latent.device)
    noise_scale = noise_level[t].unsqueeze(1)
    latent_scale = noise_scale**0.5
    noise = torch.randn_like(latent)

    noise_scale = noise_scale.unsqueeze(1)
    latent_scale = latent_scale.unsqueeze(1)
    noise_scale = (1.0 - noise_scale)**0.5

    noisy_latent = latent_scale * latent + noise_scale * noise
    return noisy_latent

if __name__=="__main__":

    model = AudioVQVAE_lightning.load_from_checkpoint('stft_check/model-epoch=0019.ckpt')
    wav_a, sr = torchaudio.load('DUMMY2/wav/p227_019.wav')
    wav_b, sr = torchaudio.load('DUMMY2/wav/p230_413.wav')

    wav, _, _ = model.vqvae(wav_b.to(model.device), wav_b.to(model.device))

    torchaudio.save('3.wav', wav.cpu(), sr)
    
    latent = torch.randn(10, 256, 100)
    put_noise2latent(latent, 29)
    model = LatentDiffusion.load_from_checkpoint('ldm_chpt/model-ldm-resepoch=0224.ckpt')
    ssl = torch.load('DUMMY2/hubert/p227_019.ssl').to(model.device)
    latent = model.inference(ssl.transpose(1, 2))
    idx = model.vq.latent2indices(latent)
    wav = model.vqvae.indices2wav(idx).squeeze(1).cpu()

    torchaudio.save('2.wav', wav, 16000)

    wav, sr = torchaudio.load('DUMMY2/wav/p227_019.wav')
    wav = wav.unsqueeze(1).to(model.device)
    idx = model.vqvae.wav2indices(wav)
    latent = model.vq.indices2latent(idx)
    print(torch.max(latent))
    latent = put_noise2latent(latent, 19)
    print(torch.max(latent))
    idx = model.vq.latent2indices(latent)
    wav = model.vqvae.indices2wav(idx).squeeze(1).cpu()

    torchaudio.save('3.wav', wav, sr)
    pass

