import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from vector_quantize_pytorch import ResidualVQ

def pad_to_multiple(tensor, multiple):
    B, L = tensor.shape
    # 计算需要填充的数量
    pad_length = (multiple - (L % multiple)) % multiple
    # 使用F.pad进行填充
    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), mode='constant', value=0)
    return padded_tensor[:, :-1]

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
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

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
            batch_first=True
        )
    
    def forward(self, x):
        y = x.transpose(1, 2)
        y, _ = self.atten(y, y, y)
        return x + y.transpose(1, 2)

class ContentEncoder(nn.Module):
    def __init__(self, channels, Activer):
        super().__init__()
        self.channels = channels
        self.downSample = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(4),
            Activer,
            Residual_Conv2D(4),
            nn.BatchNorm2d(4),
        )

        self.vector_quantizer = ResidualVQ(
            dim=self.channels // 2,
            num_quantizers=1,
            codebook_size=1024,
            commitment_weight=0.01,
            kmeans_init=True,
            ema_update=True,
        )

    def forward(self, x):
        y = self.downSample(x)
        y = y.transpose(2, 3)
        B, C, L, F = y.shape
        y = y.reshape(B, C*L, F)
        latent, indices, commit_loss = self.vector_quantizer(y)
        latent =latent.transpose(1, 2)
        return latent, indices, commit_loss

class StyleEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.proj = nn.Conv1d(self.channels + 1, self.channels, 1, 1)
        self.seq = nn.Sequential(
            nn.Conv1d(self.channels, self.channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(self.channels // 2),
            Residual_Conv1D(self.channels // 2),
            nn.BatchNorm1d(self.channels // 2),
        )
        self.gru = nn.GRU(self.channels // 2, self.channels // 2, batch_first=True)
        self.bn = nn.BatchNorm1d(self.channels // 2)

    def forward(self, x):
        x = self.proj(x)
        x = self.seq(x)
        _, h_n = self.gru(x.transpose(1, 2))
        h_n = h_n.squeeze(0).unsqueeze(-1)
        return h_n

class Decoder(nn.Module):
    def __init__(self, channels, Activer):
        super().__init__()  
        self.channels = channels 
        self.proj = nn.Conv1d(self.channels, self.channels + 1, 1, 1)
        self.recover = nn.Conv2d(1, 2, (1, 1), (1, 1))
        self.seq1 = nn.Sequential(
            nn.ConvTranspose1d(self.channels // 2, self.channels, 4, 2, 1),
            nn.BatchNorm1d(self.channels),
            Activer,
            Residual_Atten(self.channels),
            nn.BatchNorm1d(self.channels),
            Activer,
            nn.ConvTranspose1d(self.channels, self.channels, 3, 1, 1),
            nn.BatchNorm1d(self.channels),
            Activer,
        )
        # self.seq2 = nn.Sequential(
        #     nn.ConvTranspose1d(self.channels, self.channels, 1, 1),
        #     nn.BatchNorm1d(self.channels),
        #     Activer,
        #     Residual_Atten(self.channels),
        #     nn.BatchNorm1d(self.channels),
        #     Activer,
        #     nn.ConvTranspose1d(self.channels, self.channels, 4, 2, 1),
        #     nn.BatchNorm1d(self.channels),
        #     Activer,
        # )
        self.seq3 = nn.Sequential(
            nn.ConvTranspose1d(self.channels, self.channels, 1, 1),
            nn.BatchNorm1d(self.channels),
            Residual_Atten(self.channels),
            nn.BatchNorm1d(self.channels),
            nn.Conv1d(self.channels, self.channels, 3, 1, 1),
        )

    def forward(self, latent, style):
        x = self.seq1(latent + style)
        # x = self.seq2(x + style)
        x = self.seq3(x)
        x = self.proj(x).unsqueeze(1)
        x = self.recover(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.channels = n_fft // 2

        self.context = ContentEncoder(self.channels, nn.Identity())
        self.style = StyleEncoder(self.channels)

        self.vector_quantizer = ResidualVQ(
            dim=self.channels,
            num_quantizers=1,
            codebook_size=1024,
            commitment_weight=0.01,
            kmeans_init=False,
            ema_update=True,
        )

        self.decoder = Decoder(self.channels, nn.LeakyReLU(0.1))
        self.register_buffer('codebook_usage', torch.zeros(2048, dtype=torch.int32))

        
    def forward(self, wav):
        wav = pad_to_multiple(wav, 1280)
        B, L = wav.shape
        stft = self.wav2stft(wav)
        real = stft[:, 1, :, :]

        latent, indices, commit_loss = self.context(stft)
        style = self.style(real)
        _stft = self.decoder(latent, style)

        _wav = self.stft2wav(_real, imag, L)
        loss = F.l1_loss(real, _real)
        return _wav, loss, indices

    def wav2stft(self, wav):
        # wav->(B, T)
        window = torch.hann_window(self.n_fft).to(wav.device)
        stft = torch.stft(
            input=wav, 
            n_fft=self.n_fft, 
            window=window, 
            return_complex=True, 
            normalized=True, 
        )
        # 分离实部和虚部
        real = stft.real
        imag = stft.imag
        stft = torch.stack((real, imag), dim=1)    
        # stft->(B, 1282, L)
        return stft

    def stft2wav(self, real, imag, length):
        # stft->(B, 641 * 2, L)
        # real = stft[:, :stft.size(1) // 2, :]
        # imag = stft[:, stft.size(1) // 2:, :]

        stft = torch.complex(real, imag)
        window = torch.hann_window(self.n_fft).to(stft.device)
        wav = torch.istft(
            input=stft, 
            n_fft=self.n_fft, 
            window=window, 
            normalized=True,
            length=length
        )

        # wav->(B, T)
        return wav


if __name__ == '__main__':
    wav, sr = torchaudio.load('/home/kpliang/python/diffwave/DUMMY2/wav/p240_088.wav')
    wav = torch.randn(10, 320 * 64-1)
    vqvae = VQVAE(n_fft=320 * 4)
    # real, imag = vqvae.wav2stft(wav)
    # wav = vqvae.stft2wav(real, imag, wav.size(1))
    # torchaudio.save('/home/kpliang/python/diffwave/3.wav', wav, sr)
    
    x = vqvae(wav)

    hdmus = torchaudio.models.HDemucs(
        sources=['noise', 'voise'],
        audio_channels=1,
        nfft=2048, 
        depth=6
    )

    print(hdmus)
    total_params = sum(p.numel() for p in hdmus.parameters())
    print(total_params / 1000000)
    pass 
