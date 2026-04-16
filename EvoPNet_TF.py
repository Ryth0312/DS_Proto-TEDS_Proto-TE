import torch
import torch.nn as nn
from Encoders import TFEncoder


class EvoPModel(nn.Module):
    def __init__(self,
                 n_cls,  # 分类数，例如二分类传入2
                 device,  # 运行设备
                 Demo_embed_size,  # 人口统计特征嵌入大小
                 EMR_embed_size,  # 电子病历嵌入大小
                 num_layers,  # Transformer Encoder 的层数
                 p,  # Dropout 比例
                 skip_connect,  # 是否使用跳跃连接
                 TempEnc,  # 时间编码器类型
                 nhead,  # Transformer Encoder 的头数
                 nhid):  # 前馈网络的隐藏维度
        super(EvoPModel, self).__init__()

        self.device = device
        self.skip_connect = skip_connect

        # TFEncoder 定义
        self.encoder = TFEncoder(
            feat_size=Demo_embed_size + EMR_embed_size,  # 输入特征维度
            embed_size=EMR_embed_size,  # 嵌入维度
            nhead=nhead,  # 多头注意力头数
            nhid=nhid,  # 前馈网络隐藏层大小
            num_layers=num_layers,  # Transformer Encoder 层数
            p=p  # Dropout 比例
        )

        # 分类器头部
        self.classifier = nn.Linear(EMR_embed_size, n_cls)


        # 初始化
        self._init_weights()

    def _init_weights(self):
        """对模型的权重进行初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, model_type, demo_data, seq_data, seq_lens, src_mask=None):
        """
        前向传播逻辑
        :param model_type: 当前模型类型
        :param demo_data: 人口统计特征，维度 (N, D_demo)
        :param seq_data: 序列特征，维度 (N, T, D_seq)
        :param seq_lens: 序列长度，维度 (N,)
        :param src_mask: Transformer 的掩码，可选
        :return: 根据 model_type 返回不同内容
        """
        # 拼接人口统计数据和序列数据
        inputs = torch.cat([demo_data.unsqueeze(1).repeat(1, seq_data.size(1), 1), seq_data], dim=-1)
        embeddings = self.encoder(inputs, src_mask)  # 输出维度 (N, T, D_emb)

        # 对时间维度进行池化，例如取均值
        #pooled = embeddings.mean(dim=1)
        pooled = embeddings.max(dim=1).values  # 维度变为 (N, D_emb)

        if model_type in ['Base0', 'Base1']:
            logits = self.classifier(pooled)  # 分类输出
            # softmax
            log_probs = torch.log_softmax(logits, dim=1)
            return log_probs

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


