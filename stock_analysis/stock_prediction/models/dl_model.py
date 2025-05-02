import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class DLModel:
    def __init__(self, model_type='lstm'):
        """
        初始化深度学习模型
        model_type: 'lstm', 'gru', 'bilstm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_sequences(self, X, y, time_steps=10):
        """创建时间序列数据"""
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    def build_model(self, input_shape):
        """构建PyTorch深度学习模型"""
        if self.model_type == 'lstm':
            model = LSTMModel(input_size=input_shape[2],
                              hidden_size=50,
                              num_layers=3,
                              output_size=1)
        else:
            # 可以在这里添加其他类型的模型
            model = LSTMModel(input_size=input_shape[2],
                              hidden_size=50,
                              num_layers=3,
                              output_size=1)

        return model.to(self.device)

    def train(self, X, y, time_steps=10, epochs=100, batch_size=32, validation_split=0.2):
        """训练模型"""
        # 标准化数据
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))

        # 创建时间序列
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, time_steps)
        print(f"X_seq 形状: {X_seq.shape}, y_seq 形状: {y_seq.shape}")  # 添加调试信息

        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_seq)
        y_train = torch.FloatTensor(y_seq)

        # 分割训练和验证集
        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size

        X_train, X_val = X_train[:train_size], X_train[train_size:]
        y_train, y_val = y_train[:train_size], y_train[train_size:]

        # 构建模型
        self.model = self.build_model(input_shape=X_seq.shape)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10

        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0

            # 小批量训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size].to(self.device)
                batch_y = y_train[i:i + batch_size].to(self.device)

                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            # 计算训练损失
            train_loss /= len(X_train)
            train_losses.append(train_loss)

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss = criterion(val_outputs, y_val.to(self.device))
                val_losses.append(val_loss.item())

            # 学习率调整
            scheduler.step(val_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss.item():.6f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 加载最佳模型
        self.model.load_state_dict(best_model_state)

        # 预测和评估
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(torch.FloatTensor(X_seq).to(self.device)).cpu().numpy()

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_seq.reshape(-1, 1))

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"模型评估 - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        return {
            'model': self.model,
            'history': {'train_loss': train_losses, 'val_loss': val_losses},
            'y_true': y_true,
            'y_pred': y_pred,
            'X_seq': X_seq
        }

    def predict(self, X, time_steps=10):
        """使用训练好的模型进行预测"""
        if self.model is None:
            raise Exception("请先训练模型")

        # 标准化输入数据
        X_scaled = self.scaler_X.transform(X)

        # 创建时间序列
        X_seq = []
        for i in range(len(X_scaled) - time_steps + 1):
            X_seq.append(X_scaled[i:(i + time_steps)])
        X_seq = np.array(X_seq)

        print(f"预测输入序列形状: {X_seq.shape}")  # 添加调试信息

        # 转换为PyTorch张量并预测
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()

        # 反标准化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        print(f"预测输出形状: {y_pred.shape}")  # 添加调试信息

        return y_pred

    def predict_future(self, X, steps=5, time_steps=10):
        """预测未来多步"""
        if self.model is None:
            raise Exception("请先训练模型")

        # 标准化最后time_steps的数据
        last_sequence = X.tail(time_steps).values
        last_sequence_scaled = self.scaler_X.transform(last_sequence)

        # 初始化预测序列
        curr_sequence = last_sequence_scaled.copy()
        future_predictions = []

        # 逐步预测
        self.model.eval()
        for _ in range(steps):
            # 重塑为LSTM所需的输入形状
            curr_sequence_tensor = torch.FloatTensor(
                curr_sequence.reshape(1, time_steps, X.shape[1])
            ).to(self.device)

            # 预测
            with torch.no_grad():
                predicted_scaled = self.model(curr_sequence_tensor).cpu().numpy()[0]

            future_predictions.append(predicted_scaled)

            # 更新序列（移除第一个值，添加预测值）
            # 假设预测的是下一个时刻的所有特征
            curr_sequence = np.roll(curr_sequence, -1, axis=0)
            curr_sequence[-1] = predicted_scaled

        # 反标准化预测结果
        future_predictions = np.array(future_predictions)
        future_predictions = self.scaler_y.inverse_transform(future_predictions)

        return future_predictions

    def plot_learning_curve(self, history):
        """绘制训练历史"""
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('模型训练历史')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_prediction(self, y_true, y_pred, title='深度学习模型预测'):
        """绘制预测结果"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='实际值')
        plt.plot(y_pred, label='预测值', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()