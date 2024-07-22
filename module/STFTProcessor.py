import math
import torch
import torchaudio
from torch.nn import functional as F

# 定义一个类，用于处理短时傅里叶变换（STFT）相关的操作
class STFTProcessor:
    """
    STFT处理器类，提供音频信号的短时傅里叶变换和反变换功能。
    
    参数:
    hop_length: 窗口的步长，即每个窗口之间重叠的样本数。
    nfft: 傅里叶变换的点数。
    """
    def __init__(self, hop_length, nfft):
        self.hop_length = hop_length
        self.nfft = nfft

    def _pad1d(self, x: torch.Tensor, padding_left: int, padding_right: int, mode: str = "zero", value: float = 0.0):
        """
        对一维张量进行填充。

        参数:
        x: 需要填充的张量。
        padding_left: 左侧填充的长度。
        padding_right: 右侧填充的长度。
        mode: 填充的模式，默认为"zero"。
        value: 填充的值，默认为0.0。

        返回:
        填充后的张量。
        """
        """Wrapper around F.pad, in order for reflect padding when num_frames is shorter than max_pad.
        Add extra zero padding around in order for padding to not break."""
        length = x.shape[-1]
        if mode == "reflect":
            max_pad = max(padding_left, padding_right)
            if length <= max_pad:
                x = F.pad(x, (0, max_pad - length + 1))
        return F.pad(x, (padding_left, padding_right), mode, value)

    def _spectro(self, x: torch.Tensor, n_fft: int = 512, hop_length: int = 0, pad: int = 0) -> torch.Tensor:
        """
        对音频信号进行短时傅里叶变换（STFT）。

        参数:
        x: 输入的音频张量。
        n_fft: 傅里叶变换的点数，默认为512。
        hop_length: 窗口的步长，默认使用类初始化时的值。
        pad: 填充的长度，默认为0。

        返回:
        STFT结果的张量。
        """
        other = list(x.shape[:-1])
        length = int(x.shape[-1])
        x = x.reshape(-1, length)
        z = torch.stft(
            x,
            n_fft * (1 + pad),
            hop_length,
            window=torch.hann_window(n_fft).to(x),
            win_length=n_fft,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        _, freqs, frame = z.shape
        other.extend([freqs, frame])
        return z.view(other)

    def _ispectro(self, z: torch.Tensor, hop_length: int = 0, length: int = 0, pad: int = 0) -> torch.Tensor:
        """
        对STFT结果进行反变换，得到原始音频信号。

        参数:
        z: 输入的STFT张量。
        hop_length: 窗口的步长，默认使用类初始化时的值。
        length: 输出音频的长度，默认为0，表示使用z的长度。
        pad: 填充的长度，默认为0。

        返回:
        反变换后的音频张量。
        """
        other = list(z.shape[:-2])
        freqs = int(z.shape[-2])
        frames = int(z.shape[-1])

        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        win_length = n_fft // (1 + pad)
        x = torch.istft(
            z,
            n_fft,
            hop_length,
            window=torch.hann_window(win_length).to(z.real),
            win_length=win_length,
            normalized=True,
            length=length,
            center=True,
        )
        _, length = x.shape
        other.append(length)
        return x.view(other)

    def _spec(self, x):
        """
        对音频信号进行STFT，并返回 magnitude spectrogram。

        参数:
        x: 输入的音频张量。

        返回:
        magnitude spectrogram的张量。
        """
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa
        if hl != nfft // 4:
            raise ValueError("Hop length must be nfft // 4")
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = self._pad1d(x, pad, pad + le * hl - x.shape[-1], mode="reflect")

        z = self._spectro(x, nfft, hl)[..., :-1, :]
        if z.shape[-1] != le + 4:
            raise ValueError("Spectrogram's last dimension must be 4 + input size divided by stride")
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z, length=None):
        """
        对magnitude spectrogram进行反变换，得到音频信号。

        参数:
        z: 输入的magnitude spectrogram张量。
        length: 输出音频的长度。

        返回:
        反变换后的音频张量。
        """
        hl = self.hop_length
        z = F.pad(z, [0, 0, 0, 1])
        z = F.pad(z, [2, 2])
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = self._ispectro(z, hl, length=le)
        x = x[..., pad : pad + length]
        return x

    def _magnitude(self, z):
        """
        将复数维度移至通道维度
        """
        # move the complex dimension to the channel one.
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
        return m

    def _imagnitude(self, m):

        # move the channel dimension back to the complex one.
        B, C, Fr, T = m.shape
        C = C // 2
        z = m.reshape(B, C, 2, Fr, T)
        z = z.permute(0, 1, 3, 4, 2).contiguous()
        z = torch.view_as_complex(z)
        z = z.squeeze(1)
        
        return z


if __name__ == '__main__':

    audio="DUMMY2/wav48/p233/p233_006.wav"
    waveform,sr = torchaudio.load(audio)
    torchaudio.save("111.wav",waveform,sr)
    processor = STFTProcessor(hop_length=320, nfft=1280)
    
    spec = processor._spec(waveform)
   
    print(spec.shape)
    # real,img=spec
    spec = spec.unsqueeze(1)
    x = processor._magnitude(spec)
    y = processor._imagnitude(x)
    wav = processor._ispec(y,length=waveform.shape[-1])

    print(x.shape,x.dtype)
    print(y.shape,y.dtype)
    torchaudio.save("222.wav",wav,sr)
    pass

