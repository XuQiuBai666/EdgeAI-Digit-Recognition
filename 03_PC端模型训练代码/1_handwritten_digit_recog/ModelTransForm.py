"""
================================================================================
File Name: ModelTransForm.py
Description: 模型量化与格式转换脚本
--------------------------------------------------------------------------------
功能说明：
    1. 加载原始的 .keras 模型。
    2. 准备 calibration (校准) 数据集（提取 100 张训练集图片）。
    3. 利用 TensorFlow Lite Converter 进行全整型量化 (Full Integer Quantization)。
    4. 将权重和激活值全部压缩至 INT8 格式。
    5. 导出能够在 STM32 端通过 STM32Cube.AI 部署的 .tflite 文件。
输入：mnist_model.keras, 本地 mnist.npz 数据集
输出：mnist_model_int8.tflite (专用于单片机的量化模型)
================================================================================
"""

import tensorflow as tf
import numpy as np
import os

# ================= 配置路径 =================
DATA_PATH = r'G:\project\project_ForMasterOne\Pycharm\DataSet\MNIST\mnist.npz'
MODEL_DIR = r'G:\project\project_ForMasterOne\Pycharm\1_handwritten_digit_recog\model'
MODEL_PATH = os.path.join(MODEL_DIR, 'mnist_model.keras')
TFLITE_PATH = os.path.join(MODEL_DIR, 'mnist_model_int8.tflite')

# ================= 1. 加载原始模型 =================
print("正在加载原始 Keras 模型...")
model = tf.keras.models.load_model(MODEL_PATH)

# ================= 2. 准备校准数据集 =================
print("正在加载校准数据集...")
data = np.load(DATA_PATH)
x_train = data['x_train']
x_train = np.expand_dims(x_train / 255.0, -1).astype(np.float32)

# ================= 3. 核心转换代码 =================
def representative_dataset():
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]

print("\n正在将模型转换为 INT8 量化格式 (为了给单片机用)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 5. 执行转换
tflite_quant_model = converter.convert()

# ================= 4. 保存文件 =================
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_quant_model)

print(f"\n✅ 转换流程执行完毕！")
print(f"✅ 文件已保存至: {TFLITE_PATH}")
print(f"✅ 模型大小: {os.path.getsize(TFLITE_PATH) / 1024:.2f} KB")
print("\n⚠️ 请注意：请仔细在终端日志中搜索 'fully_quantize'。")
print("如果是 1，恭喜你完美成功；如果是 0，说明遇到了算子兼容性问题。")