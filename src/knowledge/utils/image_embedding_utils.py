from PIL import Image  # 图像处理库
import cn_clip.clip as clip  # 中文 CLIP 库
from cn_clip.clip import load_from_name, available_models  # CLIP 相关方法
import torch  # 再次导入 torch（可省略）
import os  # 操作系统相关
import json  # 处理 JSON 文件
from cn_clip.clip.utils import  image_transform  # CLIP 工具方法
from src.models.cn_clip_model import model
import requests
from io import BytesIO
from pathlib import Path
from src.models.vl_model_client import vl_client
from src.utils.logging_config import logger
from src.plugins._ocr import OCRPlugin


IMAGE_PROMPT_TEMPLATES = {
    "screenshot": """你是计算机领域讲师。请针对这张截图输出：
1) 识别出界面/代码/报错的核心元素；
2) 给出潜在问题与定位线索；
3) 用“结论+证据”格式给出3条要点。
输出格式：
描述：...
关键字：k1, k2, k3, k4, k5
""",
    "flowchart": """你是计算机领域讲师。请分析这张流程图/时序图：
1) 核心节点与流转顺序；
2) 每一步的输入输出；
3) 潜在瓶颈或易错点。
输出格式：
描述：...
关键字：k1, k2, k3, k4, k5
""",
    "architecture": """你是计算机领域讲师。请分析这张架构图/拓扑图：
1) 模块组成；
2) 模块关系（调用/依赖/数据流）；
3) 关键边界与可扩展点。
输出格式：
描述：...
关键字：k1, k2, k3, k4, k5
""",
    "formula": """你是计算机领域讲师。请分析这张公式/数学推导图：
1) 核心符号与变量含义；
2) 推导链路；
3) 与常见算法/复杂度关系。
输出格式：
描述：...
关键字：k1, k2, k3, k4, k5
""",
    "natural": """请描述这张图片中的主要对象与关系，聚焦与计算机学习相关的信息，避免泛化描述。
输出格式：
描述：...
关键字：k1, k2, k3, k4, k5
""",
}

_OCR_PLUGIN: OCRPlugin | None = None


def _get_ocr_plugin() -> OCRPlugin:
    global _OCR_PLUGIN
    if _OCR_PLUGIN is None:
        _OCR_PLUGIN = OCRPlugin()
    return _OCR_PLUGIN


def _detect_image_type(image_path: str, ocr_text: str = "") -> str:
    lower_path = (image_path or "").lower()
    lower_ocr = (ocr_text or "").lower()
    if any(k in lower_path for k in ["screenshot", "截屏", "screen", "error", "log", "trace"]) or "exception" in lower_ocr:
        return "screenshot"
    if any(k in lower_path for k in ["flow", "流程", "时序", "sequence"]):
        return "flowchart"
    if any(k in lower_path for k in ["arch", "架构", "topology", "拓扑", "uml"]):
        return "architecture"
    if any(k in lower_path for k in ["formula", "equation", "公式", "推导", "latex"]):
        return "formula"
    return "natural"


def _resolve_localhost_image_path(image_path: str) -> str:
    """将本地服务图片 URL 映射为本地文件路径，便于 OCR 直接读取。"""
    if not isinstance(image_path, str):
        return image_path
    if image_path.startswith('http://localhost:5050/api/system/images/'):
        filename = image_path.split('/')[-1]
        local_path = Path('saves/chat_images') / filename
        if local_path.exists():
            return str(local_path)
    return image_path

def _extract_ocr_text(image_path: str) -> str:
    try:
        plugin = _get_ocr_plugin()
        resolved_path = _resolve_localhost_image_path(image_path)
        return plugin.process_image(resolved_path) or ""
    except Exception as e:
        err_text = str(e)
        if (
                "RapidOCR" in err_text
                or "模型目录不存在" in err_text
                or "MODEL_DIR" in err_text
                or "model_not_found" in err_text
                or "model_incomplete" in err_text
        ):
            logger.info(f"OCR模型未就绪（{err_text}），已跳过OCR并继续VL/基础视觉回退")
            return ""
        logger.warning(f"OCR提取失败: {e}")
        return ""

def get_image_description(image_path, ocr_text: str | None = None):
    """
    从图片中提取描述信息，优先使用VL模型，失败时回退到基础分析

    Args:
        image_path (str): 图片路径

    Returns:
        str: 图片描述信息
    """
    ocr_text = ocr_text if ocr_text is not None else _extract_ocr_text(image_path)
    image_type = _detect_image_type(image_path, ocr_text)
    prompt = IMAGE_PROMPT_TEMPLATES.get(image_type, IMAGE_PROMPT_TEMPLATES["natural"])

    # 优先使用VL模型生成描述
    if vl_client.is_available():
        try:
            description = vl_client.get_image_description(image_path, prompt)
            logger.info("使用VL模型成功生成图片描述")
            if ocr_text:
                return f"{description}\nOCR摘要：{ocr_text[:600]}"
            return description
            
        except Exception as e:
            logger.warning(f"VL模型生成描述失败，回退到基础分析: {str(e)}")
    
    # VL模型不可用或失败时，回退到基础分析
    return _get_basic_image_description(image_path)


def _extract_knowledge_points(description: str, ocr_text: str = "", max_points: int = 8) -> list[str]:
    """从图像描述/OCR中提取知识点关键词（轻量规则）。"""
    text = f"{description}\n{ocr_text}".lower()
    candidates = [
        ("链表", ["链表", "linked list"]),
        ("栈", ["栈", "stack"]),
        ("队列", ["队列", "queue"]),
        ("二叉树", ["二叉树", "binary tree"]),
        ("图", ["图", "graph", "邻接表", "邻接矩阵"]),
        ("BFS", ["bfs", "广度优先"]),
        ("DFS", ["dfs", "深度优先"]),
        ("TCP", ["tcp"]),
        ("UDP", ["udp"]),
        ("HTTP", ["http"]),
        ("DNS", ["dns"]),
        ("路由", ["路由", "routing"]),
        ("进程/线程", ["进程", "线程", "process", "thread"]),
        ("时间复杂度", ["o(", "复杂度", "time complexity"]),
    ]
    points: list[str] = []
    for label, kws in candidates:
        if any(k in text for k in kws):
            points.append(label)
        if len(points) >= max_points:
            break
    return points


def build_image_evidence_bundle(
    image_urls: list[str],
    subject: str = "",
    mode: str = "temp_chat_image",
) -> list[dict]:
    bundle = []
    for idx, image_url in enumerate(image_urls or [], start=1):
        ocr_text = _extract_ocr_text(image_url)
        image_type = _detect_image_type(image_url, ocr_text)
        description = get_image_description(image_url, ocr_text=ocr_text)
        evidence_id = f"IMG{idx:03d}"
        bundle.append(
            {
                "evidence_id": evidence_id,
                "source_kb": "multimodal_image",
                "mode": mode,
                "subject": subject or "",
                "image_url": image_url,
                "image_type": image_type,
                "description": description[:1200],
                "ocr_text": ocr_text[:1200],
                "knowledge_points": _extract_knowledge_points(description, ocr_text=ocr_text),
            }
        )
    return bundle


def _get_basic_image_description(image_path):
    """
    基础图片描述分析（VL模型不可用时的回退方案）
    
    Args:
        image_path (str): 图片路径
        
    Returns:
        str: 基础图片描述
    """
    try:
        # 加载图片
        if image_path.startswith(('http://', 'https://')):
            # 特殊处理：如果是本地服务器图片，尝试直接读取本地文件
            if image_path.startswith('http://localhost:5050/api/system/images/'):
                # 提取文件名
                filename = image_path.split('/')[-1]
                # 构建本地文件路径
                local_path = Path("saves/chat_images") / filename
                if local_path.exists():
                    # 直接从本地文件读取，避免网络请求
                    image = Image.open(local_path)
                else:
                    # 如果本地文件不存在，回退到网络下载
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
            else:
                # 其他网络URL，正常下载
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
        else:
            # 从本地文件读取图片
            image = Image.open(image_path)
        
        # 确保图片是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 分析图片基本信息
        width, height = image.size
        format_info = image.format or "未知格式"
        mode_info = image.mode
        
        # 计算图片亮度特征
        grayscale = image.convert('L')
        brightness = sum(grayscale.getdata()) / (width * height)
        
        # 判断图片方向
        orientation = "横向" if width > height else "纵向" if height > width else "正方形"
        
        # 判断图片分辨率
        resolution = "高分辨率" if width * height > 2000000 else "中等分辨率" if width * height > 500000 else "低分辨率"
        
        # 判断亮度水平
        brightness_level = "明亮" if brightness > 200 else "中等亮度" if brightness > 100 else "较暗"
        
        # 构建基础描述
        description = f"图片基本信息：尺寸{width}x{height}像素（{orientation}，{resolution}），格式为{format_info}，颜色模式为{mode_info}，亮度为{brightness_level}。"

        # 如果检测到可能为技术文档图片，添加提示
        if any(keyword in str(image_path).lower() for keyword in
               ['network', 'topology', 'arch', 'uml', 'diagram', 'code', '算法', '网络', '系统', '数据库']):
            description += " 检测到可能是计算机知识图，建议使用VL模型获取更详细的关系描述。"
        
        logger.info("使用基础分析生成图片描述")
        return description
        
    except Exception as e:
        logger.error(f"基础图片分析失败: {str(e)}")
        return f"无法分析图片: {str(e)}"


def get_image_embedding(image_path, clip_model=None, preprocess=None):
    """
    提取图片的特征嵌入，支持本地文件路径和网络URL
    
    Args:
        image_path (str): 本地图片路径或网络图片URL
        clip_model: CLIP模型，如果为None则使用全局model
        preprocess: 图像预处理函数，如果为None则使用默认预处理
    
    Returns:
        numpy.ndarray: 归一化后的图像特征向量
    """
    if preprocess is None:
        preprocess = image_transform()
    if clip_model is None:
        clip_model = model
    try:
        # 判断是否为URL
        if image_path.startswith(('http://', 'https://')):
            # 特殊处理：如果是本地服务器图片，尝试直接读取本地文件
            if image_path.startswith('http://localhost:5050/api/system/images/'):
                # 提取文件名
                filename = image_path.split('/')[-1]
                # 构建本地文件路径
                local_path = Path("saves/chat_images") / filename
                if local_path.exists():
                    # 直接从本地文件读取，避免网络请求
                    image = Image.open(local_path)
                else:
                    # 如果本地文件不存在，回退到网络下载
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
            else:
                # 其他网络URL，正常下载
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()  # 检查请求是否成功
                image = Image.open(BytesIO(response.content))
        else:
            # 从本地文件读取图片
            image = Image.open(image_path)
        
        # 确保图片是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
    except Exception as e:
        raise ValueError(f"无法加载图片: {image_path}，错误: {str(e)}")
    
    # 预处理图片并提取特征
    image_tensor = preprocess(image).unsqueeze(0).to(clip_model.logit_scale.device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # 特征归一化
    
    return image_features.cpu().numpy().astype('float32').flatten()

def get_text_embedding(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = clip.tokenize([text]).to(device)  # 对文本进行分词并转为张量，移动到设备
    with torch.no_grad():  # 关闭梯度计算
        text_features = model.encode_text(text)  # 提取文本特征
        text_features /= text_features.norm(dim=-1, keepdim=True)  # 特征归一化
    return text_features.cpu().numpy().astype('float32').flatten()  # 转为 numpy 数组并展平成一维

def get_img(image_path):
    try:
        # 判断是否为URL
        if image_path.startswith(('http://', 'https://')):
            # 特殊处理：如果是本地服务器图片，尝试直接读取本地文件
            if image_path.startswith('http://localhost:8000/api/system/images/'):
                # 提取文件名
                filename = image_path.split('/')[-1]
                # 构建本地文件路径
                local_path = Path("saves/chat_images") / filename
                if local_path.exists():
                    # 直接从本地文件读取，避免网络请求
                    image = Image.open(local_path)
                else:
                    # 如果本地文件不存在，回退到网络下载
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
            else:
                # 其他网络URL，正常下载
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()  # 检查请求是否成功
                image = Image.open(BytesIO(response.content))
        else:
            # 从本地文件读取图片
            image = Image.open(image_path)
        print(image)
    except Exception as e:
        raise ValueError(f"无法加载图片: {image_path}，错误: {str(e)}")

if __name__ == "__main__":
    get_img("http://localhost:8000/api/system/images/c8e4197acf3f4de2b8614497d75fc032.png")