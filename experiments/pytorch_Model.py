import torch
import torch.nn as nn
import timm


class ModifiedXception(nn.Module):
    def __init__(self):
        super(ModifiedXception, self).__init__()
        
        # 加載 Xception 預訓練模型，去掉最後一層 (fc 層)
        self.base_model = timm.create_model('xception', pretrained=True)
        self.base_model.fc = nn.Identity()  # 移除原來的 fully connected 層
        
        # 新增全局平均池化層、隱藏層和輸出層
        self.global_avg_pool = nn.AdaptiveAvgPool1d(2048)  # 全局平均池化
        self.hidden_layer = nn.Linear(2048, 1370)  # 隱藏層，輸入大小取決於 Xception 的輸出大小
        self.output_layer = nn.Linear(1370, 2)  # 輸出層，依據分類數目設定
        
        # 激活函數與 dropout
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.base_model(x)               # Xception 主體
        x = self.global_avg_pool(x)           # 全局平均池化
        x = self.relu(self.hidden_layer(x))   # 隱藏層 + ReLU
        x = self.dropout(x)                   # Dropout
        x = self.output_layer(x)              # 輸出層
        x = self.softmax(x)
        return x
