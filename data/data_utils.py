import cv2
import numpy as np
import os
import time
import io
# 可选依赖（打包精简时不随包分发，与本项目推理链无关，见改造方案 §4.3 ②）
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from skimage import exposure
except Exception:
    exposure = None
from PIL import Image, ImageCms
REF_IMAGE_ADOBE_RGB_PROFILE_PATH = "./data_samples/_6eGpt4bZOU.png"
# 模块级文件读取改为容错：该文件不存在时 ADOBE_PROFILE 置 None，仅在使用处报错
try:
    ref_img = Image.open(REF_IMAGE_ADOBE_RGB_PROFILE_PATH)
    ref_icc_profile = ref_img.info.get('icc_profile')
    ADOBE_PROFILE = ImageCms.ImageCmsProfile(io.BytesIO(ref_icc_profile))
except Exception:
    ADOBE_PROFILE = None

def display_parameters(data):
    if data is None:
        return
    
    print("参数解析结果：")
    for param_name, param_info in data.items():
        print(f"\n参数：{param_name}")
        print(f"  当前值：{param_info['value']}")
        print(f"  范围：{param_info['min']} 到 {param_info['max']}")
        print(f"  初始值：{param_info['initial']}")
        print(f"  标准差：{param_info['sigma']}")


def convert_adobe_to_srgb(image_path):
    pil_img = Image.open(image_path)
    
    icc_profile = pil_img.info.get('icc_profile')
    
    if icc_profile:
        adobe_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
    else:
        adobe_profile = ADOBE_PROFILE

    srgb_profile = ImageCms.createProfile("sRGB")
    
    transform = ImageCms.buildTransformFromOpenProfiles(
        adobe_profile, srgb_profile, "RGB", "RGB"
    )
    
    converted_img = ImageCms.applyTransform(pil_img, transform)
    
    opencv_img = cv2.cvtColor(np.array(converted_img), cv2.COLOR_RGB2BGR)
    
    return opencv_img

def open_img(img_path):
    img = Image.open(image_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

