import torch
import torchaudio
import torch.nn as nn

class ConditionalConformerLayer(nn.Module):
    def __init__(self, d_model, latent_dim=64, ssl_dim=768, diffusion_step_dim=512):
        super().__init__()
        self.conformer = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=8,
            ffn_dim=1024,
            num_layers=1,
            depthwise_conv_kernel_size=5,
            dropout=0.1,
            use_group_norm=True,
        )
        self.latent_proj = nn.Conv1d(latent_dim, d_model, 1)
        self.ssl_proj = nn.Conv1d(ssl_dim, d_model, 1)
        self.diffusion_step_proj = nn.Linear(diffusion_step_dim, d_model)  

    def forward(self, latent, ssl, diffusion_step, lengths):
        B, D, L = latent.shape
        
        # Project ssl and diffusion_step
        latent = self.latent_proj(latent)
        ssl = self.ssl_proj(ssl)  
        diffusion_step = self.diffusion_step_proj(diffusion_step).unsqueeze(-1)

        # Combine x, ssl_proj, and diffusion_step_proj
        x = latent + ssl + diffusion_step

        # Pass through conformer layer
        x, lengths = self.conformer(x.transpose(1, 2), lengths)
        x = x.transpose(1, 2)
        return x, lengths

class ConformerModel(nn.Module):
    def __init__(self, d_model, conformer_layers, latent_dim=64, ssl_dim=768, diffusion_step_dim=512):
        super(ConformerModel, self).__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList()
        for i in range(conformer_layers):
            self.layers.append(ConditionalConformerLayer(d_model, latent_dim, ssl_dim, diffusion_step_dim)) 
            latent_dim=d_model

        self.output = nn.Conv1d(d_model, self.latent_dim, 1)

    def forward(self, x, ssl, diffusion_step, lengths):
        for layer in self.layers:
            x, _ = layer(x, ssl, diffusion_step, lengths)
        x = self.output(x)
        return x


if __name__ == "__main__":

    # Create the model
    model = ConformerModel(d_model=512, conformer_layers=8)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Example inputs
    B, L, D = 8, 100, 64
    x = torch.randn(B, D, L)
    ssl = torch.randn(B, 768, L)
    diffusion_step = torch.randn(B, 512)
    lengths = torch.randint(1, 101, (B,))
    lengths[0] = 100

    # Forward pass
    output = model.forward(x, ssl, diffusion_step, lengths)
    print(output.shape)  # Should output: torch.Size([8, 100, 512])
