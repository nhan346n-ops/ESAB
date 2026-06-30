import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledSigmoid(nn.Module):
    def __init__(self, margin=0.005):
        """
        改进: 使用更小的 margin (0.5%) 以覆盖更宽参数范围。
        原来 margin=0.05 导致 10% 的参数范围无法到达，
        新值使有效输出 [0.005, 0.995]，经济覆盖 [0.5%, 99.5%]。
        """
        super().__init__()
        self.margin = margin
        self.scale = 1.0 - 2 * margin

    def forward(self, x):
        # Maps (-inf, inf) to (margin, 1-margin)
        return self.margin + self.scale * torch.sigmoid(x)

class PINNESABNet(nn.Module):
    def __init__(self, seq_len=71):
        super().__init__()
        
        # Input: (Batch, 3, seq_len)
        # Channels: 0=BS_norm, 1=Theta_norm, 2=Freq_norm (expanded)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            # Dilated conv to capture global shape (e.g. nadir peak vs oblique tail)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling -> (Batch, 128, 1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4) # 4 physical parameters
        )
        
        self.scaled_sigmoid = ScaledSigmoid(margin=0.05)
        
    def forward(self, x, freq_norm):
        """
        x: (Batch, 2, seq_len)
        freq_norm: (Batch, 1)
        """
        batch_size, _, seq_len = x.shape
        
        # Expand freq_norm to match sequence length
        freq_expanded = freq_norm.unsqueeze(-1).expand(batch_size, 1, seq_len)
        
        # Concat features: (Batch, 3, seq_len)
        x_concat = torch.cat([x, freq_expanded], dim=1)
        
        # Extract features
        features = self.conv_block(x_concat).squeeze(-1) # (Batch, 128)
        
        # Regression
        logits = self.mlp(features) # (Batch, 4)
        
        # Constrain output to (0.05, 0.95)
        y_pred = self.scaled_sigmoid(logits)
        
        return y_pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, SeqLen, d_model)
        return x + self.pe[:, :x.size(1), :]


class PINNESABTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, seq_len=71):
        super().__init__()
        
        # 1. Feature Projection (3 channels -> d_model)
        self.embedding = nn.Linear(3, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len + 10)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Regression MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4)
        )
        
        self.scaled_sigmoid = ScaledSigmoid(margin=0.05)
        
    def forward(self, x, freq_norm):
        """
        x: (Batch, 2, seq_len)
        freq_norm: (Batch, 1)
        """
        batch_size, _, seq_len = x.shape
        
        # Expand freq_norm to match sequence length
        freq_expanded = freq_norm.unsqueeze(-1).expand(batch_size, 1, seq_len)
        
        # Concat features: (Batch, 3, seq_len)
        x_concat = torch.cat([x, freq_expanded], dim=1)
        
        # Transpose for Transformer: (Batch, SeqLen, 3)
        x_transposed = x_concat.transpose(1, 2)
        
        # Project to d_model: (Batch, SeqLen, d_model)
        x_emb = self.embedding(x_transposed)
        
        # Add Positional Encoding
        x_pe = self.pos_encoder(x_emb)
        
        # Transformer Pass
        transformer_out = self.transformer_encoder(x_pe) # (Batch, SeqLen, d_model)
        
        # Global Average Pooling over SeqLen -> (Batch, d_model)
        features = transformer_out.mean(dim=1)
        
        # Regression -> (Batch, 4)
        logits = self.mlp(features)
        
        # Scale to bounds -> (Batch, 4)
        y_pred = self.scaled_sigmoid(logits)
        
        return y_pred


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # Test model
    print("Testing CNN...")
    model_cnn = PINNESABNet()
    dummy_x = torch.randn(16, 2, 71)
    dummy_f = torch.randn(16, 1)
    
    out_cnn = model_cnn(dummy_x, dummy_f)
    print(f"CNN Output shape: {out_cnn.shape}")
    
    print("\nTesting Transformer (PI-Former)...")
    model_transformer = PINNESABTransformer()
    out_trans = model_transformer(dummy_x, dummy_f)
    print(f"Transformer Output shape: {out_trans.shape}")
    print(f"Transformer Min value: {out_trans.min().item():.4f}, Max value: {out_trans.max().item():.4f}")
    
    # Test params reconstruction
    z_pred = out_trans[:, 0] * 13.0 + 1.0
    print(f"Predicted Z range: {z_pred.min().item():.2f} to {z_pred.max().item():.2f}")
