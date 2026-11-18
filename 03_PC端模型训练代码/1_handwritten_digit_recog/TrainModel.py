"""
================================================================================
File Name: TrainModel.py
Description: 模型训练脚本
--------------------------------------------------------------------------------
功能说明：
    1. 加载本地 MNIST 数据集。
    2. 构建专为嵌入式端部署设计的轻量级 CNN 模型。
    3. 采用移除 Softmax 层的设计（输出 Logits），以便后续适配 INT8 纯整型量化。
    4. 训练完成后，将 FP32 原始模型保存为 .keras 格式。
输入：本地 mnist.npz 数据集
输出：mnist_model.keras (原始浮点模型)
================================================================================
"""

import tensorflow as tf
import numpy as np
import os

# ================= 配置路径 =================
DATA_PATH = r'G:\project\project_ForMasterOne\Pycharm\DataSet\MNIST\mnist.npz'
MODEL_DIR = r'G:\project\project_ForMasterOne\Pycharm\1_handwritten_digit_recog\model'
MODEL_PATH = os.path.join(MODEL_DIR, 'mnist_model.keras')

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= 1. 加载数据 =================
print("正在加载本地 MNIST 数据集...")
data = np.load(DATA_PATH)
x_train, y_train = data['x_train'], data['y_train']

# 预处理
x_train = np.expand_dims(x_train / 255.0, -1).astype(np.float32)

# ================= 2. 构建模型 =================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)  # <=== 修改点 1：删除了 activation='softmax'
])

# ================= 3. 编译并训练 =================
# 修改点 2：因为去掉了 Softmax，所以必须加上 from_logits=True
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("开始训练...")
model.fit(x_train, y_train, epochs=5, batch_size=64)

# ================= 4. 保存原始模型 =================
model.save(MODEL_PATH)
print(f"\n✅ 训练完成！原始模型已保存至: {MODEL_PATH}")