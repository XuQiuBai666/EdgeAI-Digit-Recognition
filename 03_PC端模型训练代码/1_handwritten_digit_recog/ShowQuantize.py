"""
================================================================================
File Name: ShowQuantize.py
Description: TFLite 模型底层算子检查脚本
--------------------------------------------------------------------------------
功能说明：
    1. 加载量化后的 mnist_model_int8.tflite 模型。
    2. 深入解析模型底层的 Operator (算子) 列表。
    3. 检查是否存在带有 'FLOAT' 字眼的浮点算子。
    4. 验证模型是否达成 fully_quantize: 1 (纯 INT8 运算)，排除 TF 日志的误导。
输入：mnist_model_int8.tflite
输出：终端打印所有算子及量化纯度结论
================================================================================
"""

import tensorflow as tf

TFLITE_PATH = r'G:\project\project_ForMasterOne\Pycharm\1_handwritten_digit_recog\model\mnist_model_int8.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

print("\n====== 算子侦查报告 ======")
ops = interpreter._get_ops_details()
op_type_set = set()
float_op_found = False

for op in ops:
    op_name = op['op_name']
    op_type_set.add(op_name)
    # 简单判断：如果是FLOAT版本算子，就是降级了
    if "FLOAT" in op_name:
        print(f"⚠️ 浮点算子：{op_name}")
        float_op_found = True

print("\n全部算子列表：")
print(sorted(op_type_set))

if float_op_found:
    print("\n❌ 结论：存在浮点算子 → fully_quantize:0，接口量化，内部浮点计算")
else:
    print("\n✅ 结论：没有FLOAT算子，全部整数算子 → fully_quantize:1，纯INT8模型！")
