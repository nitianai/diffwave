import torch
import torchaudio
from torch import nn
import torch.nn.functional as F

import numpy as np
import pytorch_lightning as pl
from module.params import params
from module.vq import VectorQuantizer

from module.model import Diffusion4latent
from ldm.modules.diffusionmodules.openaimodel import UNetModel

from DUMMY2.dataset import AudioFeatureDataset, collate_fn

START_T = 1
class LatentDiffusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.unet = UNetModel(
        #     image_size=None,
        #     channel_mult=(1, 2, 4),
        #     in_channels=256,
        #     model_channels=64,
        #     out_channels=256,
        #     num_res_blocks=8,
        #     attention_resolutions=[1,2,4],
        #     dropout=0.1,
        #     dims=1,
        #     num_heads=4,
        #     use_spatial_transformer=True,
        #     context_dim=768,
        # )

        self.unet = Diffusion4latent()

        self.vqvae = VQVAE(
            input_dim=1,
            base_dim=32,
            latent_dim=256,
            updown_rate=2,
            res_layers=4,
            codebook_size=2048,
            active=nn.Identity()
        )
        self.vqvae.load_state_dict(torch.load('model_weights/vqvae_weights.pth'))
        for param in self.vqvae.parameters():
            param.requires_grad = False
        
        self.vq = VectorQuantizer(
            num_embeddings=2048,
            embedding_dim=256
        )
        for param in self.vq.parameters():
            param.requires_grad = False

        beta = np.array(params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.loss_fn = params.loss_fn

    def slice(self, ssl, wav, lengths, crop):
        ssls = []
        wavs = []
        for i in range(ssl.shape[0]):
            length = lengths[i]
            if length >= crop:
                start = torch.randint(0, length - crop + 1, (1,)).item()
                end = start + crop
                _ssl = ssl[i, :, start:end]
                start = start * params.hop_samples
                end = end * params.hop_samples
                _wav = wav[i, :, start:end]
                ssls.append(_ssl)
                wavs.append(_wav)
        ssls = torch.stack(ssls)
        wavs = torch.stack(wavs)
        return ssls, wavs

    
    def training_step(self, batch, batch_idx):
        texts, sid, wav, wav_lengths, xbert, xbert_lengths, ssl, lengths, _, _, _ = batch
        ssl, wav = self.slice(ssl.transpose(1, 2), wav, lengths, params.crop_mel_frames)
        idxs = self.vqvae.wav2indices(wav)
        latent = self.vq.indices2latent(idxs)

        N, C, T = latent.shape

        t = torch.randint(START_T, len(params.noise_schedule), [N])
        # t = torch.full((N,), 0)
        noise_scale = self.noise_level[t].unsqueeze(1).to(latent.device)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(latent)

        noise_scale = noise_scale.unsqueeze(1)
        noise_scale_sqrt = noise_scale_sqrt.unsqueeze(1)
        noise_scale_sqrt_one = (1.0 - noise_scale)**0.5

        noisy_latent = noise_scale_sqrt * latent + noise_scale_sqrt_one * noise

        predicted = self.unet(noisy_latent, t, ssl)
        loss = F.mse_loss(noise, predicted)
        # 记录损失
        self.log('l1', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss

    def on_train_epoch_start(self):
        with open('learn_rate/lr_ldm.txt') as file:
            lr = file.readline()
            try:
                lr = float(lr.strip())
                for param in self.optimizers().param_groups:
                    param['lr'] = lr
            except ValueError:
                pass

    def on_train_epoch_end(self):
        pass
        # ssl = torch.load('DUMMY2/hubert/p334_023.ssl').to(self.device)
        # latent = self.inference(ssl.transpose(1, 2))
        # idxs = self.vq.latent2indices(latent)
        # audio = self.vqvae.indices2wav(idxs)
        # audio = audio.squeeze(1)
        # torchaudio.save(params.audiofile, audio.cpu(), params.sample_rate)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            params=list(self.unet.parameters()), 
            lr=params.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, 
            gamma=1, #params.lr_decay, 
            last_epoch=-1
        )
        return [opt], [scheduler]


    def inference(self, ssl):
        with torch.no_grad():
            beta = np.array(params.noise_schedule)
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)
            
            ssl = ssl.to(self.device)
            B,C,L = ssl.shape
            if L % 2 != 0:
                ssl = F.pad(ssl, (0, 1), mode='constant', value=0)            
            latent = torch.randn(ssl.shape[0], 256, params.hop_samples * ssl.shape[-1] // 4)
            latent = latent.to(self.device)
            noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(self.device)
            
            for n in range(len(alpha) - 1, START_T-1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                diff_step = torch.tensor([n]).to(self.device)
                latent = c1 * (latent - c2 * self.unet(latent, diff_step, ssl).squeeze(1))
                if n > START_T:
                    noise = torch.randn_like(latent)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    latent += sigma * noise
                latent = torch.clamp(latent, -1.0, 1.0)
        return latent
        
if __name__ == '__main__':
    model = LatentDiffusion()

    dataset = AudioFeatureDataset('filelist/vctk_audio_sid_text_train_filelist.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=params.batch_size,
        num_workers=12, 
        shuffle=True,
        collate_fn=collate_fn
    )

    torch.set_float32_matmul_precision("medium")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='ldm_chpt/',  # 保存检查点的目录
        filename=params.checkpoint_file,  # 检查点的文件名格式
        save_top_k=-1, 
        every_n_epochs=5, 
    )

    trainer = pl.Trainer(
        max_epochs=10000, 
        accelerator='gpu', 
        devices=[0, 1], 
        precision=32,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=model, 
        train_dataloaders=dataloader, 
        ckpt_path='ldm_chpt/model-ldm-resepoch=0099.ckpt'
    )
       