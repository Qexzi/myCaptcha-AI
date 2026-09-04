```txt
        输入图像 [B, 3, 64, W]
            ↓
        ============ CNN特征提取 ============
        Conv1 + ReLU + MaxPool(2,1) → [B, 64, 32, W]
            ↓
        Conv2 + ReLU + MaxPool(2,1) → [B, 128, 16, W]
            ↓
        Conv3 + BN + ReLU → [B, 256, 16, W]
            ↓
        Conv4 + ReLU + MaxPool(2,1) → [B, 256, 8, W]
            ↓
        Conv5 + BN + ReLU → [B, 512, 8, W]
            ↓
        Conv6 + ReLU + MaxPool(2,1) → [B, 512, 4, W]
            ↓
        Conv7 + BN + ReLU + Conv(k=2,s=2,w=2) + MaxPool(2,1) → [B, 512, 2, W/2]
            ↓
        Squeeze高度 → [B, 512, W/2]
            ↓
        ============ RNN序列建模 ============
        BiLSTM(2层, hidden=256, dropout=0.4)
            ↓
        [B, 512, W/2] → 双向LSTM → [B, 512, W/2]
            ↓
        ============ 分类输出 ============
        FC(512 → 63) → [B, W/2, 63]
            ↓
        Permute → [W/2, B, 63]  (CTC要求的格式)
            ↓
        Log Softmax
            ↓
        CTC Loss + 解码
```