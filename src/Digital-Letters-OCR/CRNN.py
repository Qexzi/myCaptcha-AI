# 神经网络搭建

import torch
import torch.nn as nn
from noise_modules import NoisyRNN, build_noise

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut) # 2倍隐藏层节点,因为是使用双向LSTM，两个方向的隐藏单元拼在一起

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, B, H = recurrent.size()      # T: 时间步长，实际上是宽度 B：训练批次大小 H：是输入特征维度，实际上是通道数
        t_rec = recurrent.view(T*B, H)  # 将时间步长和批次大小合并，得到[T*B, H]的张量

        output = self.embedding(t_rec)  # 将[T*B, H]的张量映射到[T*B, nOut]的张量
        output = output.view(T, B, -1)  # 将[T*B, nOut]的张量重新形状为[T, B, nOut]
        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh=256, leakyReLU=False, noise_config=None):


        """
        :param imgH: 输入图片高度
        :param nc : 输入图片通道数
        :param nclass: 输出类别数
        :param nh: RNN的隐藏层神经元节点个数
        :param leakyReLU: 是否使用leakyReLU激活函数
        :param noise_config: 噪声正则配置(dict),None/未启用时行为与旧版一致
        """


        super(CRNN, self).__init__()

        # ----------------CNN----------------
        
        # 卷积层的参数
        conv_ks = [(3,3), (3,3), (3,3), (3,3), (3,3), (3,3), (2,2)]  # 卷积核大小 (层6高度2下采样,宽度2下采样)
        conv_ps = [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (0,0)]  # padding大小
        conv_ss = [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (2,2)]  # stride大小 (层6宽度2下采样)
        conv_nm = [64, 128, 256, 256, 512, 512, 512]                 # 卷积核个数/输出通道数

        # 池化层的参数
        pool_ks = [(2,1), (2,1), (2,1), (2,1), (1,1), (1,1), (2,1)]  # 池化核大小 (层6高度2下采样,宽度1不下采样)
        pool_ss = [(2,1), (2,1), (2,1), (2,1), (1,1), (1,1), (2,1)]  # 池化 stride (层6高度2下采样)
        pool_ps = [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]  # 池化 padding


        
        cnn = nn.Sequential() # 卷积层序列化

        def convRelu(i, batchNormalization):

           nIn = nc if i == 0 else conv_nm[i-1]  # 为输入通道数
           nOut = conv_nm[i]  # 输出通带数
           cnn.add_module(f"convReluBlock_{i}", nn.Conv2d(
               nIn,
               nOut,
               kernel_size=conv_ks[i],
               stride=conv_ss[i],
               padding=conv_ps[i],
               bias=False,
           ))  # 创建卷积层处理块，命名为convReluBlock_i

           # BN层
           if batchNormalization:
               cnn.add_module(f"batchNorm_{i}", nn.BatchNorm2d(nOut))

           # Relu激活层的使用类型
           if leakyReLU:
               cnn.add_module(f"leakyReLU_{i}", nn.LeakyReLU(0.2, inplace=True))
           else:
               cnn.add_module(f"relu_{i}", nn.ReLU(inplace=True))


        def maxPool(i):
            cnn.add_module(f"maxPool_{i}", nn.MaxPool2d(kernel_size=pool_ks[i], stride=pool_ss[i],padding=pool_ps[i]))


        def convBlock(i, batchNormalization):
            convRelu(i, batchNormalization)
            maxPool(i)
        
        # 处理块0-6
        convBlock(0, True)
        convBlock(1, True)
        convBlock(2, True)
        convBlock(3, True)
        convBlock(4, True)
        convBlock(5, True)
        convBlock(6, True)

        # 待议
        # （0，1）浅层不使用BN层：浅层通道数少（64、128），梯度相对稳定；且 batch size 较小时 BN 统计量（均值/方差）波动大，反而引入噪声，影响训练稳定性
        # （2，4）通道翻倍时使用BN层：通道数从 128→256、256→512 时，参数空间骤增，BN 能加速收敛、缓解内部协变量偏移（Internal Covariate Shift）
        # （3，5）同通道第二块不加 BN：同一通道数下连续堆叠时，避免过度正则化。每两层只加一次 BN，保持一定的正则效果同时不抑制模型表达能力
        # （6）最后一块加 BN：输出层之前做最后一次归一化，让送入 RNN 的 feature map 分布更稳定，有利于序列建模

        self.cnn = cnn
        self.dropout = nn.Dropout(0.4) # 随机失活40%,防止过拟合
        
        # ----------------RNN----------------
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            nn.Dropout(0.4),
            BidirectionalLSTM(nh, nh, nclass),
        )

        # ----------------噪声正则(可选)----------------
        self.noise_state = None
        self.ctc_monitor = None
        self.aon = None
        self.adnr_cnn = None
        self.acan2d = None
        self.ctc_output_smoothing = 0.0
        comps = build_noise(noise_config, nh, nclass)
        if comps is not None:
            self.noise_state = comps['state']
            self.ctc_monitor = comps['ctc_monitor']
            self.aon = comps['aon']
            self.adnr_cnn = comps['adnr_cnn']
            self.acan2d = comps['acan2d']
            self.ctc_output_smoothing = noise_config.get('ctc_output_smoothing', 0.0)
            # 需要在两层 LSTM 之间插噪声时,换用 NoisyRNN
            if comps['grng'] is not None or comps['adnr_rnn'] is not None or comps['aon'] is not None:
                self.rnn = NoisyRNN(
                    BidirectionalLSTM(512, nh, nh),
                    comps['grng'],
                    comps['adnr_rnn'],
                    nn.Dropout(0.4),
                    BidirectionalLSTM(nh, nh, nclass),
                    comps['state'],
                )

    def forward(self, x):
        noise_on = self.training and self.noise_state is not None
        scale = 1.0
        if noise_on and self.aon is not None:
            scale = self.aon.compute_noise_scale()
            x = self.aon.input_noise(x, scale)  # AON L1: 输入图像噪声

        conv = self.cnn(x)
        B, C, H, W = conv.size()
        assert conv.size(2) == 1, f"Height must be 1, but got {conv.size(2)}"
        conv = conv.squeeze(2)  # 去掉高度维度，得到[B, C, W]
        conv = self.dropout(conv) # 随机失活,防止过拟合
        conv = conv.permute(2, 0, 1)  # 即[B, C, W] -> [W, B, C] = [T, B, H]

        if not noise_on:
            return self.rnn(conv)

        if self.adnr_cnn is not None:
            conv = self.adnr_cnn(conv, self.noise_state)  # ADNR 浅层(CNN 特征)

        # AON 存在时 self.rnn 必为 NoisyRNN,可传 aon 参数;否则(仅 GRNG/ADNR/ACAN2D)直接调用,
        # 避免 self.rnn 仍是 nn.Sequential 时传入多余关键字参数导致 TypeError
        if self.aon is not None:
            output = self.rnn(conv, aon=self.aon, aon_scale=scale)  # 内含 AON 特征噪声
        else:
            output = self.rnn(conv)  # 内含 GRNG/ADNR 深层噪声

        if self.acan2d is not None:
            out_bt = output.permute(1, 0, 2)  # [B, T, nclass]
            self.ctc_monitor.update(self.noise_state, torch.softmax(out_bt, dim=-1))
            output = self.acan2d(out_bt, self.noise_state).permute(1, 0, 2)  # 过尖则打散
        return output

        




