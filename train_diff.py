import torch
import torchaudio
from torch import nn
import torch.nn.functional as F

import numpy as np
import pytorch_lightning as pl
from module.params import params

from module.model import DiffWave
from DUMMY2.dataset import AudioFeatureDataset, collate_fn


from module import commons

from pytorch_lightning.loggers import TensorBoardLogger

def slice(ssl, latent, lengths, crop):
    # tensor的形状为(B, C, T)
    ssls = []
    latents = []
    # 遍历每个样本
    for i in range(ssl.shape[0]):
        # 获取有效长度
        length = lengths[i]
        # 过滤掉有效长度小于 segment_length 的样本
        if length >= crop:
            # 随机选择起始点，确保不会超出有效长度
            start = torch.randint(0, length - crop + 1, (1,)).item()
            end = start + crop
            # 切割片段
            _ssl = ssl[i, :, start:end]
            start = start * params.hop_samples
            end = end * params.hop_samples
            _latent = latent[i, :, start:end]
            ssls.append(_ssl)
            latents.append(_latent)
    # 将切割的片段转换为张量
    ssls = torch.stack(ssls)
    latents = torch.stack(latents)
    return ssls, latents

class ssl2latent(pl.LightningModule):
    def __init__(self, param):
        super().__init__()

        self.params = param
        self.model = DiffWave(param)

        self.vqvae = VQVAE()
        state = torch.load('model_weights/wavVQVAE_weights.pth')
        self.vqvae.load_state_dict(state)
        for param in self.vqvae.parameters():
            param.requires_grad = False

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.loss_fn = params.loss_fn
        

    def training_step(self, batch, batch_idx):
        texts, sid, wav, wav_lengths, xbert, xbert_lengths, ssl, lengths, _, _, _ = batch
        latent, _, _ = self.vqvae.vector_quantizer(wav.transpose(1, 2)) 

        ssl, latent = slice(ssl.transpose(1, 2), latent.transpose(1, 2), lengths, params.crop_mel_frames)

        N, C, T = latent.shape

        t = torch.randint(0, len(self.params.noise_schedule), [N])
        noise_scale = self.noise_level[t].unsqueeze(1).to(latent.device)
        noise_scale_sqrt = noise_scale**0.5
        noise_scale_sqrt = noise_scale_sqrt
        noise = torch.randn_like(latent)

        noise_scale = noise_scale.unsqueeze(1)
        noise_scale_sqrt = noise_scale_sqrt.unsqueeze(1)

        noisy_latent = noise_scale_sqrt * latent + (1.0 - noise_scale)**0.5 * noise

        predicted = self.model(noisy_latent, t, ssl)
        l1_loss = self.loss_fn(noise, predicted)
        src = torch.log_softmax(predicted, dim=-1)
        tgt = torch.softmax(noise, dim=-1)
        kl_loss = F.kl_div(src, tgt, reduction='batchmean')
        loss = l1_loss + kl_loss

        # 记录损失
        self.log('l1', l1_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('kl', kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss

    def on_train_epoch_start(self):
        with open('filelist/lr.txt') as file:
            lr = file.readline()
            try:
                lr = float(lr.strip())
                for param in self.optimizers().param_groups:
                    param['lr'] = lr
            except ValueError:
                pass

    def on_train_epoch_end(self):
        ssl = torch.load('DUMMY2/hubert/p334_023.ssl').to(self.device)
        
        latent = model.inference(ssl.transpose(1, 2))
        audio, _, _ = self.vqvae(latent)
        audio = audio.squeeze(1)
        
        torchaudio.save(params.audiofile, audio.cpu(), params.sample_rate)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            params=list(self.model.parameters()), 
            lr=params.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, 
            gamma=1, #params.lr_decay, 
            last_epoch=-1
        )
        return [opt], [scheduler]

    def inference(self, ssl, fast_sampling=True):
        with torch.no_grad():
            # Change in notation from the DiffWave paper for fast sampling.
            # DiffWave paper -> Implementation below
            # --------------------------------------
            # alpha -> talpha
            # beta -> training_noise_schedule
            # gamma -> alpha
            # eta -> beta
            training_noise_schedule = np.array(self.params.noise_schedule)
            inference_noise_schedule = np.array(self.params.inference_noise_schedule)

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            ssl = ssl.to(self.device)

            latent = torch.randn(ssl.shape[0], 1, self.params.hop_samples * ssl.shape[-1])
            latent = latent.to(self.device)
            noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(self.device)

            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                diff_step = torch.tensor([T[n]]).to(self.device)
                latent = c1 * (latent - c2 * self.model(latent, diff_step, ssl).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(latent)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    latent += sigma * noise
                latent = torch.clamp(latent, -1.0, 1.0)
        return latent

if __name__ == "__main__":
    dataset = AudioFeatureDataset('filelist/vctk_audio_sid_text_train_filelist.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=params.batch_size,
        num_workers=12, 
        shuffle=True,
        collate_fn=collate_fn
    )
    model = ssl2latent(params)

    torch.set_float32_matmul_precision("medium")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints/',  # 保存检查点的目录
        filename=params.checkpoint_file,  # 检查点的文件名格式
        save_top_k=-1, 
        every_n_epochs=5, 
    )

    trainer = pl.Trainer(
        max_epochs=10000, 
        accelerator='gpu', 
        devices=[params.cuda_id], 
        precision=32,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=model, 
        train_dataloaders=dataloader, 
        # ckpt_path=params.restore_file,
    ) 

    # The Middle Ages brought calligraphy to perfection, and it was natural therefore
