import torch
import torchaudio
from torch import nn

import numpy as np
import pytorch_lightning as pl

from module.params import params
from module.model import soGPT

from dataset_vctk_new import AudioFeatureDataset, collate_fn

class soGPTlighting(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.EOS = params.codebook_size
        self.SOS = params.codebook_size + 1
        self.model = soGPT(
            phoneme_vocab_size=params.phoneme_vocab_size,
            ssl_vocab_size=params.codebook_size+2,
            embedding_dim=params.dim,
            nhead=params.soGPT_nhead,
            num_decoder_layers=params.soGPT_nLayer,
            dim_feedforward=params.dim * 4
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def training_step(self, batch, batch_idx):
        texts, sid, wav, wav_lengths, phoneme, phoneme_lengths, ssl, lengths, ssl_idx = batch

        phoneme = phoneme.to(torch.long)
        ssl_idx = ssl_idx[:, :, 0]

        B, ssl_L = ssl_idx.shape
        _, phoneme_L = phoneme.shape

        sos_token = torch.full((B,), self.SOS, dtype=torch.long).unsqueeze(-1).to(self.device)
        ssl_input = torch.cat([sos_token, ssl_idx], dim=1)
        ssl_target = torch.cat([ssl_idx, torch.zeros(B, 1, dtype=torch.long).to(self.device)], dim=1)
        ssl_target[torch.arange(B), lengths] = self.EOS
        lengths += 1
  
        output = self.model(phoneme, phoneme_lengths, ssl_input, lengths)
        x = output.view(-1, output.size(-1))
        y = ssl_target.view(-1)
        
        loss = self.criterion(x, y)
        lr = self.optimizers().param_groups[0]['lr']
        # 记录损失
        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), 
            lr=2e-5,     
        )
            
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, 
            gamma=params.lr_decay, 
            last_epoch=-1
        )
        return [opt], [scheduler]

if __name__=="__main__":
    dataset_train = AudioFeatureDataset('filelist/vctk_audio_sid_text_train_filelist.txt')
    dataloader_train  = torch.utils.data.DataLoader(
        dataset=dataset_train, 
        batch_size=24, #params.batch_size,
        num_workers=16, 
        shuffle=True,
        collate_fn=collate_fn
    )

    model = soGPTlighting()

    torch.set_float32_matmul_precision("medium")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='gpt_checkpoints/',  # 保存检查点的目录
        filename='model-{epoch:04d}',  # 检查点的文件名格式
        save_top_k=-1, 
        every_n_epochs=50, 
    )

    trainer = pl.Trainer(
        default_root_dir='gpt_logs',
        max_epochs=10000, 
        accelerator='gpu', 
        devices=[1], 
        precision=32,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=model, 
        train_dataloaders=dataloader_train, 
        # ckpt_path='gpt_checkpoints/model-epoch=0249.ckpt'
    ) 


