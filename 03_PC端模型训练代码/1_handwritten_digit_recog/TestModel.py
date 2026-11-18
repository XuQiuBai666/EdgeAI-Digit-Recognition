"""
================================================================================
File Name: TestModel.py
Description: 原始模型（PC端）精度验证脚本
--------------------------------------------------------------------------------
功能说明：
    1. 加载 TrainModel.py 训练生成的 .keras 格式模型。
    2. 在完整的 10000 张测试集上进行整体评估，计算 Baseline（基准）准确率。
    3. 抽取前 5 张图片进行直观预测，打印真实标签、预测标签和置信度。
    *(注：由于模型去除了 Softmax 层，这里手动使用了 tf.nn.softmax 计算置信度)
输入：mnist_model.keras, 本地 mnist.npz 数据集
输出：终端打印测试集准确率与抽样预测结果
================================================================================
"""

import tensorflow as tf
import numpy as np
import os

# ================= 配置路径 =================
DATA_PATH = r'G:\project\project_ForMasterOne\Pycharm\DataSet\MNIST\mnist.npz'
MODEL_DIR = r'G:\project\project_ForMasterOne\Pycharm\1_handwritten_digit_recog\model'
MODEL_PATH = os.path.join(MODEL_DIR, 'mnist_model.keras')

# ================= 1. 加载模型和数据 =================
print("正在加载模型...")
model = tf.keras.models.load_model(MODEL_PATH)

print("正在加载测试集...")
data = np.load(DATA_PATH)
x_test, y_test = data['x_test'], data['y_test']
x_test = np.expand_dims(x_test / 255.0, -1).astype(np.float32)

# ================= 2. 整体评估 =================
print("\n开始在整个测试集上评估...")
# 注意：这里的 evaluate 会自动处理 from_logits
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ 测试集整体准确率: {acc * 100:.2f}%")

# ================= 3. 抽样测试 =================
print("\n--- 抽取前 5 张图片进行直观测试 ---")
# 修改点：predict 出来的是原始分数 (Logits)
predictions = model.predict(x_test[:5])

# 因为模型没有 Softmax 了，为了算出百分比，我们需要手动加上 Softmax
probs = tf.nn.softmax(predictions).numpy()

for i in range(5):
    pred_digit = np.argmax(probs[i])
    true_digit = y_test[i]
    confidence = np.max(probs[i]) * 100 # 用转换后的概率计算置信度
    print(f"图片 {i+1}: 真实标签={true_digit}, 模型预测={pred_digit}, 置信度={confidence:.2f}%")