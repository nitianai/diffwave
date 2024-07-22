import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.distributions import Beta, Normal, kl_divergence

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, base_dim, latent_dim, downsample_layers=4, res_layers=1, active=nn.ReLU()):
        super(Encoder, self).__init__()
        layers = []
        channels = base_dim

        # First layer
        layers.append(nn.Conv1d(input_dim, channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm1d(channels))
        layers.append(active)
        for _ in range(downsample_layers - 1):
            next_channels = channels * 2
            layers.append(nn.Conv1d(channels, next_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(next_channels))
            layers.append(active)
            channels = next_channels

        self.convs = nn.Sequential(*layers)
        self.res_layers = nn.ModuleList([ResidualBlock(channels) for _ in range(res_layers)])
        self.conv_mu = nn.Conv1d(channels, latent_dim, kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv1d(channels, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.convs(x)
        for res_layer in self.res_layers:
            x = res_layer(x)
        mu = self.conv_mu(x).squeeze(-1)  # Remove the extra dimension
        logvar = self.conv_logvar(x).squeeze(-1)  # Remove the extra dimension
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, base_dim, output_dim, latent_dim, upsample_layers=4, res_layers=1, active=nn.ReLU()):
        super(Decoder, self).__init__()
        layers = []
        channels = base_dim * (2 ** (upsample_layers - 1))

        self.res_layers = nn.ModuleList([ResidualBlock(channels) for _ in range(res_layers)])
        self.proj_latent = nn.Sequential(
           nn.Conv1d(latent_dim, channels, kernel_size=3, stride=1, padding=1), 
           nn.BatchNorm1d(channels),
           active
        )

        for _ in range(upsample_layers - 1):
            next_channels = channels // 2
            layers.append(nn.ConvTranspose1d(channels, next_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(next_channels))
            layers.append(active)
            channels = next_channels

        layers.append(nn.ConvTranspose1d(channels, output_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(active)

        self.down_convs = nn.Sequential(*layers)
        self.proj_out = nn.Conv1d(output_dim, output_dim, 3, 1, 1)

    def forward(self, latent):
        latent = self.proj_latent(latent)
        for res_layer in self.res_layers:
            latent = res_layer(latent)
        latent = self.down_convs(latent)
        latent = F.silu(latent)
        latent = self.proj_out(latent)
        return latent

class styleEncoder(nn.Module):
    def __init__(self, base_dim, latent_dim, downsample_layers):
        super().__init__()
        self.wav2mel = torchaudio.transforms.MelSpectrogram(
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            normalized=True
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB()
        self.encoder = Encoder(
            input_dim=80,
            base_dim=base_dim,
            latent_dim=latent_dim,
            downsample_layers=2,
            res_layers=0,
            active=nn.Identity()
        )
        self.glu = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            batch_first=True
        )

    def forward(self, wav):
        mel = self.wav2mel(wav)
        x = self.am2db(mel).squeeze(1)
        x, _ = self.encoder(x)
        y = self.glu(x.transpose(1, 2))
        style = y[1].permute(1, 2, 0)
        return style

class VAE(nn.Module):
    def __init__(self, latent_dim, loss_type='L1', res_layers=4):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, res_layers=res_layers, active=nn.SiLU())
        self.decoder = Decoder(latent_dim, res_layers=res_layers, active=nn.SiLU())
        self.loss_type = loss_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode2latent(self, wav):
        mu, logvar = self.encoder(wav)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar, z

    def loss_function(self, x_recon, x, mu, logvar):
        if self.loss_type == 'L1':
            rec_loss = F.l1_loss(x_recon, x, reduction='mean')
        elif self.loss_type == 'MSE':
            rec_loss = F.mse_loss(x_recon, x, reduction='mean')
        elif self.loss_type == 'Huber':
            rec_loss = F.smooth_l1_loss(x_recon, x, reduction='mean', beta=0.01)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return rec_loss, kl_loss


if __name__ == "__main__":
    enc = Encoder(
        input_dim=80,
        base_dim=128,
        latent_dim=256,
        downsample_layers=3,
        res_layers=4
    )
    dec = Decoder(
        base_dim=128,
        output_dim=80,
        latent_dim=256,
        upsample_layers=3,
        res_layers=4
    )
    style = styleEncoder(
        base_dim=64,
        latent_dim=256,
        downsample_layers=2
    )
    print(style)
    wav = torch.randn(10, 1, 16000)
    y = style(wav)
    y = enc(wav)
    z = dec(y[0])

    # 参数设置
    latent_dim = 10
    B = 8
    L = 16000  # 例如，1秒钟的音频采样率为16kHz
    res_layers = 8  # 设置残差块的数量
    loss_type = 'L1'  # 可以是 'L1', 'MSE' 或 'Huber'
    wav = torch.randn(B, 1, L)  # 修正输入的维度，应为 (B, 1, L)

    # 模型
    model = VAE(latent_dim, loss_type=loss_type, res_layers=res_layers)
    print(model)
    x_rec, mu, logvar, z = model(wav)
    print(z.shape)
    
