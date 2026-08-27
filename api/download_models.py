#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinerU 模型下载脚本
==================
下载 MinerU 所需的全部 OCR、版面分析、表格识别模型到 ~/magic-pdf-models/。

来源：HuggingFace opendatalab/PDF-Extract-Kit-1.0（约 3GB）

使用方式:
    python download_models.py              # 默认：HuggingFace 下载
    python download_models.py --force      # 强制重新下载

注意：
    - 首次下载约 3GB，请耐心等待
    - 国内用户可能需要 VPN 访问 huggingface.co
    - 模型会放在 %USERPROFILE%/magic-pdf-models/
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path.home() / "magic-pdf-models"

# MinerU 需要的所有 OCR 模型（从 pytorchocr/utils/resources/models_config.yml 中提取）
REQUIRED_MODELS = {
    # Detection (检测)
    "OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth": "det_ch",
    "OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth": "det_en",
    "OCR/paddleocr_torch/Multilingual_PP-OCRv3_det_infer.pth": "det_ml",

    # Recognition (识别)
    "OCR/paddleocr_torch/ch_PP-OCRv5_rec_infer.pth": "rec_ch_v5",
    "OCR/paddleocr_torch/ch_PP-OCRv4_rec_infer.pth": "rec_ch_v4",
    "OCR/paddleocr_torch/ch_PP-OCRv5_rec_server_infer.pth": "rec_ch_v5_server",
    "OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_infer.pth": "rec_ch_v4_server",
    "OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth": "rec_ch_v4_doc",
    "OCR/paddleocr_torch/en_PP-OCRv4_rec_infer.pth": "rec_en",
    "OCR/paddleocr_torch/arabic_PP-OCRv5_rec_infer.pth": "rec_arabic",
    "OCR/paddleocr_torch/korean_PP-OCRv3_rec_infer.pth": "rec_korean",
    "OCR/paddleocr_torch/japan_PP-OCRv3_rec_infer.pth": "rec_japan",
    "OCR/paddleocr_torch/chinese_cht_PP-OCRv3_rec_infer.pth": "rec_cht",
    "OCR/paddleocr_torch/ta_PP-OCRv3_rec_infer.pth": "rec_ta",
    "OCR/paddleocr_torch/te_PP-OCRv3_rec_infer.pth": "rec_te",
    "OCR/paddleocr_torch/ka_PP-OCRv3_rec_infer.pth": "rec_ka",
    "OCR/paddleocr_torch/latin_PP-OCRv3_rec_infer.pth": "rec_latin",
    "OCR/paddleocr_torch/cyrillic_PP-OCRv3_rec_infer.pth": "rec_cyrillic",
    "OCR/paddleocr_torch/devanagari_PP-OCRv3_rec_infer.pth": "rec_devanagari",

    # Dictionary files
    "OCR/paddleocr_torch/ppocrv5_dict.txt": "dict_v5",
    "OCR/paddleocr_torch/ppocr_keys_v1.txt": "dict_v1",
    "OCR/paddleocr_torch/ppocrv4_doc_dict.txt": "dict_v4_doc",
    "OCR/paddleocr_torch/en_dict.txt": "dict_en",
    "OCR/paddleocr_torch/korean_dict.txt": "dict_korean",
    "OCR/paddleocr_torch/japan_dict.txt": "dict_japan",
    "OCR/paddleocr_torch/chinese_cht_dict.txt": "dict_cht",
    "OCR/paddleocr_torch/ta_dict.txt": "dict_ta",
    "OCR/paddleocr_torch/te_dict.txt": "dict_te",
    "OCR/paddleocr_torch/ka_dict.txt": "dict_ka",
    "OCR/paddleocr_torch/latin_dict.txt": "dict_latin",
    "OCR/paddleocr_torch/arabic_dict.txt": "dict_arabic",
    "OCR/paddleocr_torch/cyrillic_dict.txt": "dict_cyrillic",
    "OCR/paddleocr_torch/devanagari_dict.txt": "dict_devanagari",
}


def check_model(model_path: str) -> bool:
    """检查单个模型文件是否存在。"""
    return (MODELS_DIR / model_path).exists()


def check_all_models() -> dict[str, bool]:
    """检查所有必需模型的状态。返回 {path: exists}。"""
    return {path: check_model(path) for path in REQUIRED_MODELS}


def download_from_hf(force: bool = False):
    """从 HuggingFace 下载模型。"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("[ERROR] 请先安装 huggingface_hub: pip install huggingface_hub")
        return False

    repo_id = "opendatalab/PDF-Extract-Kit-1.0"

    if force:
        print(f"[INFO] 强制重新下载全部模型到 {MODELS_DIR}...")
        try:
            snapshot_download(
                repo_id,
                local_dir=str(MODELS_DIR),
                resume_download=True,
                max_workers=4,
            )
        except Exception as e:
            print(f"[WARN] 全量下载失败: {e}")
            print("[INFO] 尝试逐个文件下载...")

    # 检查缺失的模型并逐个下载
    model_status = check_all_models()
    missing = [path for path, exists in model_status.items() if not exists]

    if not missing:
        print("[OK] 所有模型已就绪！")
        return True

    print(f"[INFO] 缺失 {len(missing)} 个模型文件，开始下载...")

    # 首先尝试从 HuggingFace 仓库下载
    for i, model_path in enumerate(missing, 1):
        hf_path = f"models/{model_path}"  # HF repo 中的路径有 models/ 前缀
        local_path = MODELS_DIR / model_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  [{i}/{len(missing)}] {model_path} ...", end=" ")
        try:
            hf_hub_download(
                repo_id,
                hf_path,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download 可能下载到不同位置，检查结果
            if check_model(model_path):
                print("OK")
            else:
                # 尝试查找下载的文件
                downloaded = list(MODELS_DIR.rglob(Path(model_path).name))
                if downloaded:
                    downloaded[0].rename(local_path)
                    print("OK (relocated)")
                else:
                    print("MISSING (file not in HF repo)")
        except Exception as e:
            print(f"FAILED: {e}")

    # 最终检查
    model_status = check_all_models()
    missing_after = [p for p, e in model_status.items() if not e]
    if missing_after:
        print(f"\n[WARN] 仍有 {len(missing_after)} 个模型缺失:")
        for p in missing_after:
            print(f"  - {p}")
        print("\n请手动下载这些模型文件并放入对应目录。")
        return False

    print(f"\n[OK] 全部 {len(REQUIRED_MODELS)} 个模型已就绪！")
    return True


def download_from_paddleocr_fallback():
    """
    从 PaddleOCR 官方源下载缺失的中文模型（PyTorch 转换版）。

    HuggingFace 仓库可能不包含所有模型，此函数从 PaddleOCR
    官方下载 Paddle 格式然后需要用户手动转换。
    """
    import urllib.request
    import tarfile
    import tempfile

    # PaddleOCR 官方模型 URL
    PADDLEOCR_MODELS = {
        # 中文检测 V3
        "ch_PP-OCRv3_det_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
        # 中文识别 V4 (ch_lite_v4 使用)
        "ch_PP-OCRv4_rec_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
        # 中文识别 V5 Server
        "ch_PP-OCRv5_rec_server_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_server_infer.tar",
    }

    # 注意：PaddleOCR 官方模型是 PaddlePaddle 格式，MinerU 需要 PyTorch 格式。
    # 如果 HuggingFace 下载失败，需要手动转换或从其他源获取 PyTorch 版本。
    print("\n[INFO] PaddleOCR 官方模型是 PaddlePaddle 格式，MinerU 需要 PyTorch 格式。")
    print("[INFO] 建议使用 HuggingFace 下载（默认方式），或确保网络可访问 huggingface.co。")
    return False


def main():
    force = "--force" in sys.argv

    print("=" * 60)
    print("  MinerU (magic-pdf) 模型下载")
    print("=" * 60)
    print(f"\n模型存放路径: {MODELS_DIR}")
    print(f"模型文件数: {len(REQUIRED_MODELS)}")
    print()

    if not force:
        model_status = check_all_models()
        existing = sum(1 for v in model_status.values() if v)
        print(f"已有模型: {existing}/{len(REQUIRED_MODELS)}")
        if existing == len(REQUIRED_MODELS):
            print("[OK] 所有模型已就绪，无需下载。")
            return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if download_from_hf(force=force):
        return

    # HuggingFace 失败，尝试 PaddleOCR fallback
    print("\n[WARN] HuggingFace 下载不完整，尝试 PaddleOCR 备选方案...")
    download_from_paddleocr_fallback()


if __name__ == "__main__":
    main()
