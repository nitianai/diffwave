import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from vector_quantize_pytorch import ResidualVQ

import pytorch_lightning as pl
from DUMMY2.dataset import AudioFeatureDataset, collate_fn
from module.wavVQVAE import VQVAE

class AudioVQVAE_lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vqvae = VQVAE(
            input_dim=1,
            base_dim=32,
            latent_dim=256,
            updown_rate=2,
            res_layers=4,
            codebook_size=2048,
            active=nn.Identity()
        )

    def slice(self, wav, lengths, crop):
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

    def pad_wav(self, wav, pad_multiple):
        _, _, L = wav.shape
        pad_len = (pad_multiple - L % pad_multiple) % pad_multiple
        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), 'constant')
        return wav

    def training_step(self, batch, batch_idx):
        texts, sid, wav, wav_lengths, _, _, ssl, lengths, _, latent, _=batch
        B, C, L = wav.shape
        # mask = torch.arange(L).expand(B, L) < wav_lengths.unsqueeze(1)
        wav = self.slice(wav, wav_lengths, 20000)

        r_wav, vq_loss, ids = self.vqvae(wav, None)
        rec_loss = F.l1_loss(r_wav, wav)

        loss = rec_loss

        unique_ids, counts = torch.unique(ids, return_counts=True)
        unique_ids = unique_ids.to(torch.int32)
        counts = counts.to(torch.int32)
        self.vqvae.codebook_usage.index_add_(0, unique_ids, counts)
        # Log codebook usage
        num = torch.count_nonzero(self.vqvae.codebook_usage).cpu().numpy()
        self.log('use', num.item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('rec', rec_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        self.vqvae.codebook_usage.zero_()
        return loss

    def on_train_epoch_start(self):
        with open('learn_rate/lr_wav_vqvae.txt') as file:
            lr = file.readline()
            try:
                lr = float(lr.strip())
                for param in self.optimizers().param_groups:
                    param['lr'] = lr
            except ValueError:
                pass

    def on_train_epoch_end(self):
        wav, sr = torchaudio.load('DUMMY2/wav/p227_019.wav')
        wav = self.pad_wav(wav.unsqueeze(1), 16)
        wav = wav.to(self.device)

        _wav, _, _ = self.vqvae(wav, None)
        torchaudio.save('1.wav', wav.squeeze(1).cpu(), sr)
        torchaudio.save('2.wav', _wav.squeeze(1).cpu(), sr)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

####################################################################################
if __name__ == "__main__":
    dataset = AudioFeatureDataset('filelist/vctk_audio_sid_text_train_filelist.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=8,
        collate_fn=collate_fn)

    torch.set_float32_matmul_precision("medium")

    model = AudioVQVAE_lightning()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='vqvae_check/',  # 保存检查点的目录
        filename='model-style-relu-{epoch:04d}',  # 检查点的文件名格式
        save_top_k=-1, 
        every_n_epochs=5, 
    )

    trainer = pl.Trainer(
        max_epochs=10000, 
        accelerator='gpu', 
        devices=[0], 
        precision=32,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=model, 
        train_dataloaders=dataloader, 
        #ckpt_path='vqvae_check/model-style-relu-epoch=0059.ckpt'
    )
