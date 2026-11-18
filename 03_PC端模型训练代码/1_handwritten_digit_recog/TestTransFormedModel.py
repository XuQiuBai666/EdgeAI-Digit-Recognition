"""
================================================================================
File Name: TestTransformedModel.py
Description: TFLite 量化模型（PC端）精度验证脚本
--------------------------------------------------------------------------------
功能说明：
    1. 使用 tf.lite.Interpreter 加载 .tflite 格式的量化模型。
    2. 模拟单片机的数据输入方式：将测试集图片从 (0~255) 映射为 (-128~127) 的 int8 格式。
    3. 循环输入 1000 张测试图片，统计预测准确的次数。
    4. 对比 FP32 原始模型，评估 INT8 量化带来的精度损失。
输入：mnist_model_int8.tflite, 本地 mnist.npz 数据集
输出：终端打印量化模型的测试准确率 (如 98.20%)
================================================================================
"""

import tensorflow as tf
import numpy as np

# 1. 加载数据集
DATA_PATH = r'G:\project\project_ForMasterOne\Pycharm\DataSet\MNIST\mnist.npz'
data = np.load(DATA_PATH)
x_test, y_test = data['x_test'], data['y_test']

# 2. 加载 TFLite 模型
TFLITE_PATH = r'G:\project\project_ForMasterOne\Pycharm\1_handwritten_digit_recog\model\mnist_model_int8.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. 循环测试（为了快，这里只取前1000张测试）
correct_count = 0
test_num = 1000

for i in range(test_num):
    # 预处理：将 0~255 转为 -128~127 的 int8
    img = (x_test[i].astype(np.int32) - 128).astype(np.int8)
    img = np.expand_dims(img, axis=(0, -1))  # 增加 batch 和 channel 维度

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    pred_digit = np.argmax(output)

    if pred_digit == y_test[i]:
        correct_count += 1

print(f"✅ TFLite 量化模型在 {test_num} 张图片上的准确率: {(correct_count / test_num) * 100:.2f}%")