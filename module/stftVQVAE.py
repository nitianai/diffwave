import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from vector_quantize_pytorch import ResidualVQ
from module.model4stft import HEncLayer, HDecLayer, StyleEncLayer
from module.modules import MelStyleEncoder
from module.STFTProcessor import STFTProcessor

def pad_to_multiple(tensor, multiple):
    B, L = tensor.shape
    # 计算需要填充的数量
    pad_length = (multiple - (L % multiple)) % multiple
    # 使用F.pad进行填充
    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), mode='constant', value=0)
    return padded_tensor

class Residual_Conv1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, 1, 1)

    def forward(self, x):
        y = self.conv(x) + x
        return y

class Residual_Conv2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        y = self.conv(x) + x
        return y

class Residual_GRU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gru = nn.GRU(channels, channels, batch_first=True)


    def forward(self, x):
        y, h = self.gru(x.transpose(1, 2))
        return y.transpose(1, 2) + x

class Residual_Atten(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.atten = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
    
    def forward(self, q, k, v):
        y, _ = self.atten(query=q,key=k, value=v)
        return y

class ContentEncoder(nn.Module):
    def __init__(self, out_channels, Activer):
        super().__init__()
        self.downSample = nn.Sequential(
            HEncLayer(in_channels=2, out_channels=48),
            HEncLayer(in_channels=48, out_channels=96)
        )
        self.vector_quantizer = ResidualVQ(
            dim=out_channels,
            num_quantizers=1,
            codebook_size=1024,
            commitment_weight=0.01,
            kmeans_init=True,
            ema_update=True,
        )   
    

    def forward(self, x):
        y = self.downSample(x).contiguous()  
        B, C, Fr, T = y.shape
        beforvq = y

        y = y.view(B, C, Fr*T).transpose(1, 2)
        latent, indices, commit_loss = self.vector_quantizer(y)
        latent = latent.transpose(1, 2).view(B, C, Fr, T)
        return latent, indices, beforvq

class Decoder(nn.Module):
    def __init__(self, in_channels, Activer):
        super().__init__()  
        self.upSample = nn.Sequential(
            HDecLayer(in_channels=96, out_channels=48),
            HDecLayer(in_channels=48, out_channels=2)
        )
        self.atten = Residual_Atten(192)
        self.proj = nn.Linear(192, 96)

    
    def forward(self, latent, style):
        B, C, Fr, T = latent.shape
        style = style.expand(B, C, Fr, T)
        mix = latent
        mix = torch.cat((latent, style), dim=1)
        mix = mix.view(B, 192, Fr*T).transpose(1, 2)

        mix = self.atten(mix, mix, mix)
        mix = self.proj(mix)
        mix = mix.view(B, Fr, T, C)
        mix = mix.permute(0, 3, 1, 2)
    
        x = self.upSample(mix)

        return x, mix

class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.downSample = nn.Sequential(
            HEncLayer(in_channels=2, out_channels=48),
            HEncLayer(in_channels=48, out_channels=96)
        )
        self.gru = nn.GRU(input_size=96 * 40, hidden_size=96 * 40, batch_first=True)
    
    def forward(self, stft):
        style = self.downSample(stft).contiguous()
        B, C, Fr, T = style.shape
        style = style.view(B, C*Fr, T)
        style = style.transpose(1, 2)
        _, style = self.gru(style)
        style = style.permute(1, 2, 0)
        style = style.view(B, C, Fr, 1)
        
        return style

class VQVAE(nn.Module):
    def __init__(self, n_fft):
        super().__init__()

        self.STFT = STFTProcessor(n_fft // 4, nfft=n_fft)

        self.context = ContentEncoder(96, nn.Identity())
        self.style = StyleEncoder()

        self.decoder = Decoder(96, nn.LeakyReLU(0.1))
        self.register_buffer('codebook_usage', torch.zeros(2048, dtype=torch.int32))

    def forward(self, wav_context, wav_style):
        # 处理Style
        wav_style = pad_to_multiple(wav_style, 1280).unsqueeze(1)
        stft = self.STFT._spec(wav_style)
        stft = self.STFT._magnitude(stft)
        x = stft
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        style = self.style(x)

        # 处理Context
        wav_context = pad_to_multiple(wav_context, 1280).unsqueeze(1)
        B, C, L = wav_context.shape
        stft_context = self.STFT._spec(wav_context)
        stft = self.STFT._magnitude(stft_context)
        x = stft
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        latent, indices, beforvq = self.context(x)
        
        # _stft, mix = self.decoder(beforvq, style)
        # loss = F.mse_loss(stft, _stft)
        _stft, mix = self.decoder(latent, style)
        loss = F.mse_loss(beforvq, mix)

        _stft = self.STFT._imagnitude(_stft)
        wav = self.STFT._ispec(_stft, L)
        return wav, loss, indices


if __name__ == '__main__':
    wav, sr = torchaudio.load('/home/kpliang/python/diffwave/DUMMY2/wav/p240_088.wav')
    wav = torch.randn(10, 320 * 64)
    vqvae = VQVAE(n_fft=320 * 4)

    # real, imag = vqvae.wav2stft(wav)
    # wav = vqvae.stft2wav(real, imag, wav.size(1))
    # torchaudio.save('/home/kpliang/python/diffwave/3.wav', wav, sr)
    
    x = vqvae(wav)

    hdmus = torchaudio.models.HDemucs(
        sources=['noise'],
        audio_channels=1,
        nfft=1280, 
        depth=2
    )
    print(hdmus)

    z = hdmus._spec(wav.unsqueeze(1))
    z = hdmus._magnitude(z)
    z = vqvae._imagnitude(z)
    w = hdmus._ispec(z, wav.size(-1))
    
    torchaudio.save('1.wav', w.squeeze(1), sr)
    x = hdmus._magnitude(z)

    total_params = sum(p.numel() for p in hdmus.parameters())
    print(total_params / 1000000)
    pass 
