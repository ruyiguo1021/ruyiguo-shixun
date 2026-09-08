import torch
import ultralytics

print("--- 环境验证报告 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
print(f"Ultralytics YOLO 版本: {ultralabelmelytics.__version__}")