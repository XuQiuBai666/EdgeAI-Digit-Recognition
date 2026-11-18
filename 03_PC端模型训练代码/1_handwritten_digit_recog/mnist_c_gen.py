"""
@FileName: mnist_c_gen.py
@Desc: 从MNIST测试集提取单张图片，转换为模型需要的int8格式，输出C语言常量数组
@Function:
    1. 读取mnist.npz数据集
    2. 取一张28×28手写数字灰度图
    3. 像素映射：原始0~255 uint8  --> -128~127 int8
    4. 打印可直接粘贴到STM32工程的 const int8_t 数组
@Note:
    适配全整数量化MNIST模型，输入预处理规则：pixel - 128
    输出数组尺寸：784 = 28 * 28，HWC排布
@Author:
"""
import numpy as np

# 换成你的路径
data = np.load(r'G:\project\project_ForMasterOne\Pycharm\DataSet\MNIST\mnist.npz')
x_test, y_test = data['x_test'], data['y_test']

# 取第一张图
img = x_test[20]
print(f"// 真实标签是: {y_test[20]}")

# 将 0~255 的像素转成 -128~127 的 int8 (因为模型输入是 int8)
img_int8 = (img.astype(np.int32) - 128).astype(np.int8)

# 打印为 C 语言数组
print("const int8_t test_image[784] = {")
for i in range(28):
    row = ", ".join([f"{val:4d}" for val in img_int8[i]])
    print(f"    {row},")
print("};")
