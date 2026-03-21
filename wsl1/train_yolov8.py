from ultralytics import YOLO
import torch
import os
import gc

# 訓練前清理
gc.collect()
torch.cuda.empty_cache()

# 设置工作目录
os.chdir(os.path.expanduser('~'))

# 加载模型
model = YOLO('yolov8n.pt')  # 使用预训练模型

# 训练配置
results = model.train(
    data='/home/jewer/drone1/data.yaml',
    epochs=10,
    imgsz=640,
    batch=-1,  # 根据 GPU 内存调整
    device=0,  # 或 'cpu'
    workers=0,  # 减少 workers 避免内存问题
    patience=30,  # 早停耐心值
    save=True,
    save_period=10,
    pretrained=True,
    optimizer='AdamW',  # 或 'SGD'
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 最终学习率
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    plots=True,
    name='drone1_ubuntu18'
)

print("train completed！")