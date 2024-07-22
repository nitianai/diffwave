import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1, 1)

    def forward(self, x):
        B, D, L = x.shape
        quantized = torch.zeros_like(x)
        indices = torch.zeros(B, L, dtype=torch.long, device=x.device)
        # 分块处理
        chunk_size = 2048
        for i in range(0, L, chunk_size):
            end = min(i + chunk_size, L)
            x_chunk = x[:, :, i:end]  # 形状 (B, D, chunk_size)
            x_flatten = x_chunk.permute(0, 2, 1).contiguous().view(-1, D)
            distances = torch.cdist(x_flatten.unsqueeze(1), self.embeddings.weight.unsqueeze(0))
            chunk_indices = distances.argmin(dim=-1)
            chunk_quantized = self.embeddings(chunk_indices).view(B, end-i, D).permute(0, 2, 1)
            quantized[:, :, i:end] = chunk_quantized
            indices[:, i:end] = chunk_indices.view(B, end-i)
        
        return quantized, indices


    def latent2indices(self, x):
        B, D, L = x.shape
        indices = torch.zeros(B, L, dtype=torch.long, device=x.device)
        # 分块处理
        chunk_size = 2048
        for i in range(0, L, chunk_size):
            end = min(i + chunk_size, L)
            x_chunk = x[:, :, i:end]  # 形状 (B, D, chunk_size)
            x_flatten = x_chunk.permute(0, 2, 1).contiguous().view(-1, D)
            distances = torch.cdist(x_flatten.unsqueeze(1), self.embeddings.weight.unsqueeze(0))
            chunk_indices = distances.argmin(dim=-1)
            chunk_quantized = self.embeddings(chunk_indices).view(B, end-i, D).permute(0, 2, 1)
            indices[:, i:end] = chunk_indices.view(B, end-i)
        
        return indices

    def indices2latent(self, idx):
        latent = self.embeddings(idx)
        return latent.transpose(1, 2)


if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 256, 10)  # 假设输入形状为 (B, 256, L)
    vq_layer = VectorQuantizer(num_embeddings=512, embedding_dim=256)
    indices = vq_layer.latent2indices(x)

    latent = vq_layer.indices2latent(indices)

    print("Quantized shape:", quantized.shape)  # (B, 256, L)
    print("Indices shape:", indices.shape)      # (B, L)
