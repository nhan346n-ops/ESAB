import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pinn_esab_forward import DifferentiableESAB
from pinn_model import PINNESABTransformer
from data_generator import ESABDataset


class PINNLoss(nn.Module):
    """
    PINN 损失函数: 数据损失 + 物理损失 (Physics-informed).

    数据损失: 监督预测参数与真实参数一致性
    物理损失: 预测参数重建的 BS 曲线与输入 BS 曲线一致性

    改进:
    - 自适应损失归一化 (EMA)，防止量级失衡
    - 扩展参数范围与 data_generator 一致
    - 数值稳定性保护
    """
    def __init__(self, lambda_physics=0.1, ema_decay=0.9):
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_model = DifferentiableESAB()
        self.lambda_physics = lambda_physics
        # 损失尺度 EMA (用于自适应归一化)
        self.register_buffer('phys_ema', torch.tensor(1.0))
        self.register_buffer('data_ema', torch.tensor(1.0))
        self.ema_decay = ema_decay

    def forward(self, y_pred, y_true, x_input, freq_norm):
        # 1. 数据损失 (参数监督)
        loss_data = self.mse(y_pred, y_true)

        # 2. 物理损失 (曲线重建自监督)
        # 反归一化到物理域 (范围与 data_generator.py 一致)
        z_pred = y_pred[:, 0:1] * 13.0 + 1.0
        mu_pred = y_pred[:, 1:2] * 30.0 - 15.0
        s1_pred = y_pred[:, 2:3] * 54.5 + 0.5    # s1 在 [0.5, 55.0] 范围
        d2_pred = y_pred[:, 3:4] * 74.0 + 1.0    # delta2 在 [1, 75] 范围

        freq_hz = freq_norm * 415000.0 + 35000.0
        theta_deg = x_input[:, 1, :] * 70.0      # 反归一化角度

        # 通过物理层前向传播
        bs_reconstructed = self.physics_model(theta_deg, freq_hz, z_pred,
                                               s1_pred, d2_pred, mu_pred)

        # 反归一化输入 BS
        bs_input_db = x_input[:, 0, :] * 20.0 - 20.0

        loss_physics = self.mse(bs_reconstructed, bs_input_db)

        # 3. 自适应损失归一化 (EMA 平衡)
        with torch.no_grad():
            self.phys_ema = self.ema_decay * self.phys_ema + \
                           (1 - self.ema_decay) * loss_physics.detach()
            self.data_ema = self.ema_decay * self.data_ema + \
                           (1 - self.ema_decay) * loss_data.detach()

        # 归一化后两个损失都在 ~1 量级，lambda_physics 控制物理约束强度
        loss_data_norm = loss_data / (self.data_ema + 1e-8)
        loss_phys_norm = loss_physics / (self.phys_ema + 1e-8)

        # 总损失
        loss = loss_data_norm + self.lambda_physics * loss_phys_norm
        return loss, loss_data, loss_physics


def train_pinn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 数据集: 训练集 + 验证集
    print("Generating Synthetic Datasets (50K train + 5K validation)...")
    train_dataset = ESABDataset(num_samples=50000)
    val_dataset = ESABDataset(num_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

    # 2. 模型与损失
    model = PINNESABTransformer().to(device)
    criterion = PINNLoss(lambda_physics=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 3. 验证用的独立损失 (无EMA，使用原始物理损失做早停指标)
    criterion_val = PINNLoss(lambda_physics=1.0)
    # 固定EMA为1.0，使验证时总损失 = data_loss + phys_loss (原始值)
    criterion_val.data_ema.fill_(1.0)
    criterion_val.phys_ema.fill_(1.0)

    # 早停设置
    best_val_phys = float('inf')
    patience = 6
    patience_counter = 0
    best_model_path = "pinn_esab_pretrained_best.pth"

    # 4. 训练循环
    epochs = 30
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0
        total_data = 0
        total_phys = 0

        criterion.lambda_physics = min(1.0, 0.01 + epoch * 0.05)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, freq, y in pbar:
            x, freq, y = x.to(device), freq.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x, freq)
            loss, l_data, l_phys = criterion(y_pred, y, x, freq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            total_data += l_data.item()
            total_phys += l_phys.item()

            pbar.set_postfix({
                'L': f"{loss.item():.4f}",
                'Data': f"{l_data.item():.4f}",
                'Phys': f"{l_phys.item():.4f}",
                'lam': f"{criterion.lambda_physics:.3f}"
            })

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train - Loss: {avg_train_loss:.4f}, Data: {total_data/len(train_loader):.4f}, Phys: {total_phys/len(train_loader):.4f}")

        # --- 验证阶段 ---
        model.eval()
        val_data_loss = 0
        val_phys_loss = 0
        with torch.no_grad():
            for x, freq, y in val_loader:
                x, freq, y = x.to(device), freq.to(device), y.to(device)
                y_pred = model(x, freq)
                # 使用独立验证损失 (无EMA, lambda=1.0, 原始量级)
                _, l_data, l_phys = criterion_val(y_pred, y, x, freq)
                val_data_loss += l_data.item()
                val_phys_loss += l_phys.item()
        avg_val_data = val_data_loss / len(val_loader)
        avg_val_phys = val_phys_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val   - Data: {avg_val_data:.4f}, Phys: {avg_val_phys:.2f}")

        # --- 早停检查 (使用原始物理损失，不受EMA/λ影响) ---
        if avg_val_phys < best_val_phys:
            best_val_phys = avg_val_phys
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  -> New best model saved (val_phys={avg_val_phys:.2f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    # 5. 最终模型保存 (加载最佳权重覆盖最终模型)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model (val_phys={best_val_phys:.2f}) as final.")
    torch.save(model.state_dict(), "pinn_esab_pretrained.pth")
    print(f"Best model: {best_model_path}")
    print(f"Final model: pinn_esab_pretrained.pth")


if __name__ == "__main__":
    train_pinn()
