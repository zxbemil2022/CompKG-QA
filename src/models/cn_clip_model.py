#import torch  # PyTorch 深度学习框架
#from PIL import Image  # 图像处理库
#import cn_clip.clip as clip  # 中文 CLIP 库
#from cn_clip.clip import load_from_name, available_models  # CLIP 相关方法
import torch
import os
from pathlib import Path
from src.utils import logger
from cn_clip.clip.utils import create_model
from cn_clip.clip import load_from_name

# 设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"

# 路径设置
pre_train_model_path = Path("saves/model/clip_finetune/checkpoints/best.pt")

model = None

try:
    if pre_train_model_path.exists():
        # 如果文件存在，尝试加载
        pretrained = torch.load(str(pre_train_model_path), map_location='cpu')
        model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=pretrained)
        logger.info(f"成功加载本地微调模型: {pre_train_model_path}")
    else:
        allow_download = os.getenv("ALLOW_CN_CLIP_DOWNLOAD", "false").lower() == "true"
        if allow_download:
            logger.warning(f"未找到本地模型 {pre_train_model_path}，尝试加载官方基础模型...")
            model, _ = load_from_name("ViT-B-16", device=device, download_root="./models")
        else:
            logger.info("未找到本地CN-CLIP模型，且未开启 ALLOW_CN_CLIP_DOWNLOAD，跳过初始化。")
            model = None

    if model:
        model = model.to(device)
        model.eval()
        logger.info("CN-CLIP 模型初始化成功")
except Exception as e:
    logger.warning(f"CN-CLIP 初始化失败 (已跳过，不影响主流程): {e}")
    model = None  # 确保程序不会因为模型缺失而彻底崩溃

__all__ = ["model"]
