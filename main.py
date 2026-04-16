import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# 导入之前的代码模块
from Configs import *
from Utils_embed import *
from Utils_eval import Evaluate
from Utils_train import ModelInit, Train
from Encoders import ChannelwiseTemporalEncoder
from Encoders import TemporalEncoder
from Utils_basic import metrics
from Encoders import PredictionModel
def main():
    # 加载数据
    train_data = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Train_data.npy'), dtype=torch.float32)
    train_demo = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Train_demo.npy'), dtype=torch.float32)
    train_label = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Train_label.npy'), dtype=torch.long)

    val_data = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Val_data.npy'), dtype=torch.float32)
    val_demo = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Val_demo.npy'), dtype=torch.float32)
    val_label = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Val_label.npy'), dtype=torch.long)

    test_data = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Test_data.npy'), dtype=torch.float32)
    test_demo = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Test_demo.npy'), dtype=torch.float32)
    test_label = torch.tensor(np.load('D:/DS_Proto(TE)/MIMIC_dataset/Test_label.npy'), dtype=torch.long)

    train_lens = torch.tensor([train_data.shape[1]] * len(train_data), dtype=torch.long)
    val_lens = torch.tensor([val_data.shape[1]] * len(val_data), dtype=torch.long)
    test_lens = torch.tensor([test_data.shape[1]] * len(test_data), dtype=torch.long)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义模型
    model_type = 'Proto0'  # 假设我们使用 Proto0 的 Encoder

    #model = ChannelwiseTemporalEncoder(input_size=76, hidden_size=8)

    # 初始化原始的 ChannelwiseTemporalEncoder
    encoder = ChannelwiseTemporalEncoder(input_size=76, hidden_size=8, num_layers=2, p=0.3)
    # 使用 PredictionModel 包装 encoder
    model = PredictionModel(encoder, input_size=76, hidden_size=8, num_classes=1).to(device)
    #model = TemporalEncoder(input_size=76)
    model = model.to(device)

    # 确保所有参数默认可训练
    for name, param in model.named_parameters():
        param.requires_grad = True  # 强制设置为可训练


    # 定义优化器、调度器和损失函数
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = torch.nn.MSELoss()  # 你可以选择合适的损失函数，例如 MSE 或对比损失

    # 定义训练参数
    epochs = 20
    batch_size = 64
    early_stop = 5
    model_file = 'encoder_only.pth'

    # 自定义训练逻辑，只计算 Encoder 的损失
    train_Loader = MIMICLoader(train_data, train_data, train_lens, train_label, batch_size=batch_size, shuffle=True)
    val_Loader = MIMICLoader(val_data, val_data, val_lens, val_label, batch_size=batch_size, shuffle=False)
    test_Loader = MIMICLoader(test_data, test_data, test_lens, test_label, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, _, batch_lens, _ in train_Loader:
            batch_data, batch_lens = batch_data.to(device), batch_lens.to(device)

            optimizer.zero_grad()

            # 获取模型输出
            all_hts, embeds = model(batch_data, batch_lens)
            # 选择重构损失
            loss = 0
            for i in range(model.input_size):
                channel_all_ht = all_hts[:, :, i].unsqueeze(-1)  # (batch_size, seq_length, 1)
                channel_data = batch_data[:, :, i].unsqueeze(-1)  # (batch_size, seq_length, 1)
                loss += criterion(channel_all_ht, channel_data)


            #loss = criterion(all_hts, batch_data)  # 这里选择重构任务，优化 all_hts

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data, _, batch_lens, _ in val_Loader:
                batch_data, batch_lens = batch_data.to(device), batch_lens.to(device)
                all_hts, embeds = model(batch_data, batch_lens)
                loss = 0
                for i in range(model.input_size):
                    channel_all_ht = all_hts[:, :, i].unsqueeze(-1)  # (batch_size, seq_length, 1)
                    channel_data = batch_data[:, :, i].unsqueeze(-1)  # (batch_size, seq_length, 1)
                    loss += criterion(channel_all_ht, channel_data)
                #loss = criterion(embeds, batch_data)  # 验证时也使用 all_hts
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if epoch == 0 or val_loss < opt_loss:
            opt_loss = val_loss
            torch.save(model.state_dict(), model_file)
            print(f"Saved Encoder model at epoch {epoch + 1}")
        else:
            early_stop -= 1
            if early_stop == 0:
                print("Early stopping triggered.")
                break

    # 定义模型评估函数
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

    def evaluate_metrics(y_true, y_pred_prob):
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_pred_prob)
        precisions, recalls, _ = precision_recall_curve(y_true, y_pred_prob)
        auprc = auc(recalls, precisions)
        return acc, auroc, auprc

    def evaluate_model(model, dataloader, device):
        model.eval()
        y_true, y_pred_prob = [], []
        with torch.no_grad():
            for batch_data, _, batch_lens, batch_labels in dataloader:
                batch_data = batch_data.to(device)
                batch_lens = batch_lens.to(device)

                # 解包模型输出，取 all_hts
                all_hts, _ = model(batch_data, batch_lens)

                # 假设任务为逐样本分类，取最后一个时间步的平均值作为预测概率
                batch_pred_prob = torch.sigmoid(all_hts[:, -1, :]).mean(dim=-1).cpu().numpy()
                batch_labels = batch_labels.cpu().numpy()  # 转为 NumPy

                # 记录真实标签和预测概率
                y_true.extend(batch_labels.flatten())  # 展平为 1D
                y_pred_prob.extend(batch_pred_prob.flatten())  # 展平为 1D

        print(f"y_true shape: {np.array(y_true).shape}")
        print(f"y_pred_prob shape: {np.array(y_pred_prob).shape}")
        return np.array(y_true), np.array(y_pred_prob)

    # 在训练完成后评估模型
    print("Evaluating on Validation Set...")
    val_y_true, val_y_pred_prob = evaluate_model(model, val_Loader, device)
    val_acc, val_auroc, val_auprc = evaluate_metrics(val_y_true, val_y_pred_prob)
    print(f"Validation - ACC: {val_acc:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

    print("Evaluating on Test Set...")
    test_y_true, test_y_pred_prob = evaluate_model(model, test_Loader, device)
    test_acc, test_auroc, test_auprc = evaluate_metrics(test_y_true, test_y_pred_prob)
    print(f"Test - ACC: {test_acc:.4f}, AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")


if __name__ == '__main__':
    main()
