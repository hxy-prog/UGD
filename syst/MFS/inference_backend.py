# inference_backend.py
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import glob
import os
from pathlib import Path

# 全局路径（根据你的环境调整）
DDP_ROOT = Path("DDP")
CONFIG_IML =  Path("configs/iml/ddp_swin_b_2x8_512x512_epochbase_iml.py")
CKPT_IML =  Path("ckpt/ckpt13/epoch_50.pth")
SCRIPT_IML =  Path("image_demo.py")
DDIMVIS_DIR =DDP_ROOT / "ddimvis1"

CONFIG_CIML =  Path("configs/iml/ddp_swin_l_2x8_512x512_epochbase_iml_cons_shuffle_resize_two.py")
CKPT_CIML =  Path("finalresult/ciml/epoch_50.pth")
SCRIPT_CIML =  Path("image_demo_c.py")
OUTPUT_CIML =  DDP_ROOT / "output/out.png"


def run_iml_inference(input_image_path: str, output_dir: str, unique_id: str):
    """
    执行单图 IML 推理，返回 (result_path, heatmap_path)
    """
    # 清空 ddimvis1
    for f in DDIMVIS_DIR.glob("*.jpg"):
        f.unlink()
    cmd = ["python", str(SCRIPT_IML), input_image_path, str(CONFIG_IML), str(CKPT_IML)]
    with open('cmd.txt', 'w', encoding='utf-8') as f:  # 👈 加 encoding='utf-8'
        f.write(f"{cmd}")
    result = subprocess.run(
        cmd,
        cwd=str(DDP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=120
    )
    if result.returncode != 0:
        with open('error_iml.txt', 'w', encoding='utf-8') as f:  # 👈 加 encoding='utf-8'
            f.write(f"IML 推理失败:\n{result.stderr}")
        # raise RuntimeError(f"IML 推理失败:\n{result.stderr}")
        # print(f"IML 推理失败:\n{result.stderr}")

    mask_paths = sorted(glob.glob(str(DDIMVIS_DIR / "*.jpg")))
    if not mask_paths:
        raise FileNotFoundError("未生成掩码")

    gt_path = max(mask_paths, key=os.path.getmtime)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise ValueError("无法读取最终掩码")
    h, w = gt.shape

    masks_list = []
    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks_list.append(m.astype(np.float32) / 255.0)

    masks = np.stack(masks_list, axis=0)
    uncertainty = masks.std(axis=0)
    u_min, u_max = uncertainty.min(), uncertainty.max()
    u_norm = (uncertainty - u_min) / (u_max - u_min) if u_max > u_min else np.zeros_like(uncertainty)
    u_img = (u_norm * 255).astype(np.uint8)
    u_color = cv2.applyColorMap(u_img, cv2.COLORMAP_JET)

    result_path = Path(output_dir) / f"iml_result_{unique_id}.png"
    heatmap_path = Path(output_dir) / f"iml_heatmap_{unique_id}.png"

    shutil.copy(gt_path, result_path)
    cv2.imwrite(str(heatmap_path), u_color)

    return str(result_path), str(heatmap_path)


def run_ciml_inference(img1_path: str, img2_path: str, output_dir: str, unique_id: str):
    """
    执行双图 CIML 推理，返回 mask_path
    """

    cmd = ["python", str(SCRIPT_CIML), img1_path, img2_path, str(CONFIG_CIML), str(CKPT_CIML)]

    with open('ccmd.txt', 'w', encoding='utf-8') as f:  # 👈 加 encoding='utf-8'
        f.write(f"{cmd}")

    result = subprocess.run(
        cmd,
        cwd=str(DDP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=120
    )
    if result.returncode != 0:
        with open('error_ciml.txt', 'w', encoding='utf-8') as f:  # 👈 加 encoding='utf-8'
            f.write(f"CIML 推理失败:\n{result.stderr}")
        # raise RuntimeError(f"CIML 推理失败:\n{result.stderr}")
        # print(f"CIML 推理失败:\n{result.stderr}")

    if not OUTPUT_CIML.exists():
        raise FileNotFoundError("CIML 未生成 out.png")

    mask_path = Path(output_dir) / f"ciml_mask_{unique_id}.png"
    shutil.copy(str(OUTPUT_CIML), mask_path)
    return str(mask_path)