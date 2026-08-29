"""
噪声救援模块
"""
import math
import torch
import torch.nn as nn


def _randn_f32(x):
    # autocast 下 randn_like 会继承 x 的 fp16,低幅度噪声精度差,统一 fp32
    return torch.randn(x.shape, dtype=torch.float32, device=x.device)


class TrainingState:
    """共享训练状态中心:各模块通过它交换指标,协同决策"""
    def __init__(self):
        self.grad_norm_ema = 0.5          # 梯度健康度
        self.loss_gap_ema = 0.0           # 过拟合程度
        self.attn_entropy_ema = 4.0       # 对齐分布熵(越低越尖)
        self.neuron_death_ratio = 0.0     # 神经元死亡率
        self.training_progress = 0.0      # 训练进度 [0,1]

    def update_ema(self, key, value, decay=0.9):
        current = getattr(self, key)
        setattr(self, key, decay * current + (1 - decay) * value)


def make_2d_input_noise(x, sigma, valid_mask=None):
    """图像 [B,C,H,W] 加 2D 感知噪声:按每样本像素 std 缩放,避免淹没字符笔画"""
    if sigma <= 0:
        return x
    base = _randn_f32(x)
    rel = sigma * (x.float().std(dim=[1, 2, 3], keepdim=True) + 1e-6)
    noise = base * rel
    if valid_mask is not None:
        noise = noise * valid_mask
    return (x.float() + noise).to(x.dtype)


class ADNR(nn.Module):
    """深度自适应噪声残差:深层噪声大,随训练进度衰减,神经元死亡率高时增强"""
    def __init__(self, dim, layer_idx, total_layers, base_noise=0.01):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.total_layers = max(total_layers, 1)
        self.noise_scale = nn.Parameter(torch.tensor(base_noise))
        self.depth_factor = (layer_idx / max(self.total_layers - 1, 1)) ** 2

    def forward(self, x, state=None):
        if not self.training or self.noise_scale.item() <= 0:
            return x

        noise = _randn_f32(x) * torch.abs(self.noise_scale).float() * self.depth_factor

        progress_scale = 1.0
        if state is not None:
            progress_scale = 1.0 - 0.5 * state.training_progress   # 1.0 → 0.5
            death = state.neuron_death_ratio
            if death > 0.01:                                       # 死亡率 > 1% 时放大
                progress_scale *= (1.0 + min(death * 10, 0.5))
        return (x.float() + noise * progress_scale).to(x.dtype)

    def extra_repr(self):
        return f"layer={self.layer_idx}, depth_factor={self.depth_factor:.3f}"


class GRNG(nn.Module):
    """梯度急救噪声门:对低激活区域注入结构化噪声,救援死亡神经元"""
    def __init__(self, dim, noise_std=0.05, use_lightweight=False):
        super().__init__()
        self.dim = dim
        self.noise_std = noise_std
        if use_lightweight:
            self.noise_proj = nn.Linear(dim, dim)
        else:
            self.noise_proj = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, dim),
            )
        self.gate = nn.Linear(dim, 1)
        self.rescue_strength = nn.Parameter(torch.tensor(0.1))
        self.grad_norm_threshold = 0.8

    def forward(self, x, state=None):
        if not self.training or self.noise_std <= 0:
            return x

        grad_norm = state.grad_norm_ema if state is not None else 0.5
        orig_dtype = x.dtype
        # 投影/归一化固定 fp32,避免 fp16 数值不稳
        with torch.amp.autocast('cuda', enabled=False):
            xf = x.float()
            base_noise = torch.randn_like(xf) * self.noise_std
            if grad_norm > self.grad_norm_threshold:
                structured_noise = base_noise                  # 梯度健康,跳过投影省算力
            else:
                structured_noise = self.noise_proj(base_noise)
                structured_noise = structured_noise / (structured_noise.norm(dim=-1, keepdim=True) + 1e-6)

            activation_score = torch.sigmoid(self.gate(xf))
            rescue_gate = 1.0 - activation_score               # 激活越低救援越猛
            x = xf + structured_noise * rescue_gate * torch.abs(self.rescue_strength).float()

            if state is not None:
                state.neuron_death_ratio = (x.abs() < 1e-6).float().mean().item()
        return x.to(orig_dtype)


class ACAN2D(nn.Module):
    """反塌陷噪声的序列版:分布过尖(对齐塌缩)时注入负偏置噪声打散"""
    def __init__(self, dim, noise_scale=0.1, sharpness_threshold=1.5):
        super().__init__()
        self.dim = dim
        self.base_noise_scale = noise_scale
        self.sharpness_threshold = sharpness_threshold
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale))

    def forward(self, x, state=None, sharpness=None):
        if not self.training or self.base_noise_scale <= 0:
            return x

        effective_noise = torch.abs(self.noise_scale).float()
        if state is not None and state.grad_norm_ema < 0.1:
            effective_noise *= 0.3                             # 梯度已崩时少添乱

        if sharpness is None and state is not None:
            # 熵越低 → 分布越尖 → sharpness 越大
            sharp = max(0.0, 4.0 - state.attn_entropy_ema)
            sharpness = torch.tensor(sharp, device=x.device).view(1, 1, 1)

        rescue_mask = torch.ones_like(x)
        if sharpness is not None:
            if sharpness.numel() == 1:
                rescue_mask = (sharpness > self.sharpness_threshold).float().expand_as(x)
            else:
                rescue_mask = (sharpness > self.sharpness_threshold).float()

        orig_dtype = x.dtype
        adaptive_noise = _randn_f32(x) * effective_noise
        spread_noise = -torch.abs(adaptive_noise)
        # 一半随机打散 + 一半负向抑制独大位置
        x = x.float() + (adaptive_noise * 0.5 + spread_noise * 0.5) * rescue_mask
        return x.to(orig_dtype)


class CTCMonitor:
    """CTC 对齐塌缩监控:把字符分布熵写入 state.attn_entropy_ema(伪熵,越低越尖)"""
    def __init__(self, entropy_decay=0.95):
        self.entropy_decay = entropy_decay

    def update(self, state, ctc_probs):
        with torch.no_grad():
            p = ctc_probs.float().clamp(1e-8, 1.0)
            ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()
        if state is not None:
            state.update_ema('attn_entropy_ema', ent, decay=self.entropy_decay)


class AONModule(nn.Module):
    """
    自适应过拟合噪声:输入 + 特征两层
    噪声强度由 train-val 损失差距(EMA)驱动,λ 受梯度范数动态调节
    (原版 L3 标签噪声输出 one-hot 软标签,CTCLoss 不兼容,已移除)
    """
    def __init__(self, d_model, sigma_base=0.1, lambda_gap=2.0, noise_schedule='adaptive',
                 use_input_noise=True, use_feature_noise=True):
        super().__init__()
        self.lambda_gap = lambda_gap
        self.noise_schedule = noise_schedule
        self.use_input_noise = use_input_noise
        self.use_feature_noise = use_feature_noise

        self.sigma_input = nn.Parameter(torch.tensor(sigma_base))
        self.sigma_feature = nn.Parameter(torch.tensor(sigma_base))

        self.feature_noise_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )

        self.register_buffer('train_loss_ema', torch.tensor(0.0))
        self.register_buffer('val_loss_ema', torch.tensor(0.0))
        self.register_buffer('loss_gap_ema', torch.tensor(0.0))
        self.register_buffer('grad_norm_ema', torch.tensor(0.5))
        self.ema_decay = 0.9
        self.gap_ema_decay = 0.95

    def compute_noise_scale(self, train_loss=None, val_loss=None):
        """根据过拟合程度计算噪声缩放 [0.05, 1];无损失信息时返回 1.0"""
        if self.noise_schedule != 'adaptive':
            return 1.0
        if train_loss is None or val_loss is None or val_loss < 1e-8:
            return 1.0
        gap = abs(train_loss - val_loss) / val_loss
        self.loss_gap_ema = self.gap_ema_decay * self.loss_gap_ema + (1 - self.gap_ema_decay) * gap
        smoothed_gap = self.loss_gap_ema.item()

        grad_norm = self.grad_norm_ema.item()
        if grad_norm < 0.5:
            lambda_scale = 1.0 + (0.5 - grad_norm) * 2.0             # 梯度消失 → 加噪
        elif grad_norm > 1.5:
            lambda_scale = max(0.3, 1.0 - (grad_norm - 1.5) * 0.5)   # 梯度健康 → 减噪
        else:
            lambda_scale = 1.0
        scale = math.tanh(self.lambda_gap * lambda_scale * smoothed_gap)
        return max(scale, 0.05)

    def update_grad_norm(self, grad_norm):
        self.grad_norm_ema = self.ema_decay * self.grad_norm_ema + (1 - self.ema_decay) * grad_norm

    def update_loss_ema(self, train_loss, val_loss):
        self.train_loss_ema = self.ema_decay * self.train_loss_ema + (1 - self.ema_decay) * train_loss
        self.val_loss_ema = self.ema_decay * self.val_loss_ema + (1 - self.ema_decay) * val_loss

    def input_noise(self, x, scale=1.0, valid_mask=None):
        """L1 输入噪声:文本 embedding 走 padding mask,图像走 2D 相对强度"""
        if not self.training or not self.use_input_noise:
            return x
        if x.dim() == 3:
            noise = _randn_f32(x) * torch.abs(self.sigma_input).float() * scale
            mask = (x.float().abs().sum(dim=-1, keepdim=True) > 1e-6).float()
            return (x.float() + noise * mask).to(x.dtype)
        return make_2d_input_noise(x, torch.abs(self.sigma_input).item() * scale, valid_mask)

    def feature_noise(self, x, scale=1.0):
        """L2 特征噪声:结构化投影 + 范数门控(特征越'死'噪声越大)"""
        if not self.training or not self.use_feature_noise:
            return x
        orig_dtype = x.dtype
        with torch.amp.autocast('cuda', enabled=False):
            xf = x.float()
            base_noise = torch.randn_like(xf) * torch.abs(self.sigma_feature).float() * scale
            structured = self.feature_noise_proj(base_noise)
            structured = structured / (structured.norm(dim=-1, keepdim=True) + 1e-6)
            feature_norm = xf.norm(dim=-1, keepdim=True)
            rescue_gate = torch.exp(-feature_norm * 2)         # 范数小 → gate 大
            return (xf + structured * rescue_gate * 0.1).to(orig_dtype)


class NoisyRNN(nn.Module):
    """BiLSTM 序列的噪声版:两层 LSTM 之间插 GRNG/ADNR/AON 特征噪声
    eval 时各噪声模块直通,输出与非噪声路径一致"""
    def __init__(self, blstm1, grng, adnr, dropout, blstm2, state):
        super().__init__()
        self.blstm1 = blstm1
        self.grng = grng
        self.adnr = adnr
        self.dropout = dropout
        self.blstm2 = blstm2
        self.state = state    # 纯 Python 对象,不会被注册为子模块

    def forward(self, x, aon=None, aon_scale=1.0):
        x = self.blstm1(x)
        if self.grng is not None:
            x = self.grng(x, self.state)
        if self.adnr is not None:
            x = self.adnr(x, self.state)
        if aon is not None:
            x = aon.feature_noise(x, aon_scale)
        return self.blstm2(self.dropout(x))


DEFAULT_NOISE_CONFIG = {
    'enabled': True,
    'use_adnr': True,
    'use_grng': True,
    'use_acan2d': True,
    'use_aon': True,
    'adnr_base_noise': 0.01,
    'grng_noise_std': 0.05,
    'grng_lightweight': False,
    'acan2d_noise_scale': 0.1,
    'acan2d_sharpness_threshold': 1.5,
    'ctc_entropy_decay': 0.95,
    'aon_sigma_base': 0.1,
    'aon_lambda_gap': 2.0,
    'aon_use_input_noise': True,
    'aon_use_feature_noise': True,
    'ctc_output_smoothing': 0.0,
}


def build_noise(cfg, nh, nclass, cnn_dim=512):
    """按配置构造组件;cfg 未启用时返回 None"""
    if not cfg or not cfg.get('enabled'):
        return None
    comps = {
        'state': TrainingState(),
        'ctc_monitor': None, 'aon': None,
        'adnr_cnn': None, 'grng': None, 'adnr_rnn': None, 'acan2d': None,
    }
    if cfg.get('use_aon'):
        comps['aon'] = AONModule(
            nh,
            sigma_base=cfg.get('aon_sigma_base', 0.1),
            lambda_gap=cfg.get('aon_lambda_gap', 2.0),
            use_input_noise=cfg.get('aon_use_input_noise', True),
            use_feature_noise=cfg.get('aon_use_feature_noise', True),
        )
    if cfg.get('use_adnr'):
        comps['adnr_cnn'] = ADNR(cnn_dim, layer_idx=1, total_layers=3,
                                 base_noise=cfg.get('adnr_base_noise', 0.01))
        comps['adnr_rnn'] = ADNR(nh, layer_idx=2, total_layers=3,
                                 base_noise=cfg.get('adnr_base_noise', 0.01))
    if cfg.get('use_grng'):
        comps['grng'] = GRNG(nh, noise_std=cfg.get('grng_noise_std', 0.05),
                             use_lightweight=cfg.get('grng_lightweight', False))
    if cfg.get('use_acan2d'):
        comps['ctc_monitor'] = CTCMonitor(entropy_decay=cfg.get('ctc_entropy_decay', 0.95))
        comps['acan2d'] = ACAN2D(nclass, noise_scale=cfg.get('acan2d_noise_scale', 0.1),
                                 sharpness_threshold=cfg.get('acan2d_sharpness_threshold', 1.5))
    return comps
