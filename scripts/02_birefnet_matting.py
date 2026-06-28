#!/usr/bin/env python3
"""
02_birefnet_matting.py — 用 BiRefNet 模型批量抠图（ClipPet）

自动检测 CUDA → fp16 加速，CPU → fp32 兜底。
支持 CUDA OOM 时自动回退到 CPU。
支持模型下载失败时提供替代方案指引。

用法:
  python 02_birefnet_matting.py <输入目录> [--output ./matted] [--size 1024] [--model ZhengPeng7/BiRefNet]

示例:
  python scripts/02_birefnet_matting.py frames_raw --output frames_matted
"""

import argparse
import os
import sys
import time
from pathlib import Path


# ============================================================
# 依赖预检
# ============================================================
def check_dependencies():
    """在导入重量级库之前先检查依赖是否可用，给出友好提示。"""
    missing = []

    try:
        import torch  # noqa
    except ImportError:
        missing.append("torch")

    try:
        import transformers  # noqa
    except ImportError:
        missing.append("transformers")

    try:
        import PIL  # noqa
    except ImportError:
        missing.append("pillow")

    try:
        import torchvision  # noqa
    except ImportError:
        missing.append("torchvision")

    if missing:
        print("❌ 缺少依赖: " + ", ".join(missing))
        print()
        print("请安装：")
        print()
        if "torch" in missing or "torchvision" in missing:
            print("  # CUDA 版（推荐，NVIDIA 显卡）:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            print()
            print("  # CPU 版（无 NVIDIA 显卡）:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print()
        if "transformers" in missing:
            print("  pip install transformers")
        if "pillow" in missing:
            print("  pip install pillow")
        sys.exit(1)

    print("✅ 依赖检查通过")


# ============================================================
# 模型加载（含缓存检测和下载兜底）
# ============================================================
FALLBACK_MODELS = [
    "ZhengPeng7/BiRefNet",
    "briaai/RMBG-2.0",
    "briaai/RMBG-1.4",
]


def resolve_model(model_id: str) -> str:
    """检查模型是否在本地缓存；如果不在，先尝试下载，失败则依次尝试备用模型。"""
    import huggingface_hub

    try:
        # 先检查是否已缓存
        cached = huggingface_hub.try_to_load_from_cache(
            model_id, "model.safetensors"
        )
        if cached is not None and not isinstance(cached, str):
            # try_to_load_from_cache 可能返回 _CACHED_NO_EXIST
            # 保险一点：直接尝试 list_repo_files
            pass
        else:
            print(f"📦 模型已缓存: {model_id}")
            return model_id
    except Exception:
        pass

    # 尝试检查 repo 是否存在且可访问
    for attempt_model in [model_id] + [m for m in FALLBACK_MODELS if m != model_id]:
        if attempt_model != model_id:
            print(f"\n⚠️  尝试备用模型: {attempt_model}")

        try:
            # 快速检查 repo 是否可访问
            from huggingface_hub import list_repo_files
            _ = list_repo_files(attempt_model, timeout=10)
            print(f"📡 模型仓库可访问: {attempt_model}")
            return attempt_model
        except Exception as e:
            print(f"   ❌ 无法访问 {attempt_model}: {e}")
            continue

    # 所有模型都不可用
    print()
    print("=" * 60)
    print("❌ 所有模型都无法下载。可能的原因：")
    print("   1. 网络无法访问 HuggingFace（需要科学上网）")
    print("   2. HuggingFace 服务暂时不可用")
    print()
    print("   解决方法：")
    print("   a) 设置代理环境变量后再试：")
    print("      export HF_ENDPOINT=https://hf-mirror.com")
    print("      python 02_birefnet_matting.py ...")
    print()
    print("   b) 手动下载模型放到缓存目录：")
    print("      ~/.cache/huggingface/hub/")
    print("=" * 60)
    sys.exit(1)


# ============================================================
# 清理代理环境变量
# ============================================================
def clean_proxy():
    """清理可能导致 httpx/requests 报错的代理环境变量（保留系统代理配置）。"""
    # 有些代理变量格式不对会导致 urllib3/httpx 报错
    for key in list(os.environ.keys()):
        if key.lower().endswith("_proxy") or key.lower().endswith("_PROXY"):
            val = os.environ[key]
            # 如果代理值是空的或者格式明显不对，清除它
            if not val or val.strip() in ("", ":", "''", '""'):
                os.environ.pop(key, None)


# ============================================================
# 核心抠图
# ============================================================
def matting(
    input_dir: str,
    output_dir: str = "./matted",
    model_id: str = "ZhengPeng7/BiRefNet",
    image_size: int = 1024,
    frame_pattern: str = "frame_*.jpg",
    batch_size: int = 0,  # 0=逐帧处理（最省显存）
):
    clean_proxy()

    # 需要在 resolve_model 之前确认 huggingface_hub 可用
    try:
        import huggingface_hub  # noqa
    except ImportError:
        print("⚠️  huggingface_hub 未安装，跳过模型缓存检测")
        # 继续使用原始 model_id

    src = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 设备检测 ----------
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"🖥️  GPU: {gpu_name}  ({gpu_mem:.1f} GB)")
        if gpu_mem < 4:
            print("⚠️  显存较小 (< 4GB)，建议降低 --size 或使用 CPU 模式")
    else:
        print("🖥️  设备: CPU（未检测到 NVIDIA 显卡，使用 fp32 推理，速度较慢）")
        print("   提示: 如果后续报 OOM，可降低 --size 到 512 或 768")

    # ---------- 模型加载 ----------
    resolved_model = resolve_model(model_id)
    print(f"📦 加载模型: {resolved_model}")
    print(f"   推理尺寸: {image_size}×{image_size}")

    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    from PIL import Image

    try:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            resolved_model, trust_remote_code=True,
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("可能的原因：")
        print("  - HuggingFace 无法访问（尝试设置 HF_ENDPOINT=https://hf-mirror.com）")
        print("  - 磁盘空间不足")
        print("  - 模型文件损坏（尝试删除缓存后重试）")
        print(f"    缓存目录: ~/.cache/huggingface/hub/")
        sys.exit(1)

    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        birefnet.to(device)
        if device == "cuda":
            birefnet.half()
        else:
            birefnet.float()
    except Exception as e:
        print(f"❌ 模型设备迁移失败: {e}")
        sys.exit(1)

    birefnet.eval()

    # ---------- 预处理 ----------
    transform_fn = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ---------- 扫描帧文件 ----------
    frame_files = sorted(src.glob(frame_pattern))
    if not frame_files:
        print(f"❌ 未找到匹配 {frame_pattern} 的文件")
        print(f"   搜索目录: {src}/")
        # 列出目录内容帮助排查
        all_files = list(src.iterdir())
        if all_files:
            print(f"   目录中共 {len(all_files)} 个文件，前 10 个：")
            for f in all_files[:10]:
                print(f"     - {f.name}")
        else:
            print(f"   目录为空")
        sys.exit(1)

    print(f"🖼️  共 {len(frame_files)} 帧待处理\n")

    # ---------- 逐帧处理 ----------
    success = 0
    skipped = 0
    fallback_to_cpu = False

    for i, frame_path in enumerate(frame_files, 1):
        out_path = out / f"matted_{i:03d}.png"
        if out_path.exists():
            skipped += 1
            continue

        try:
            t0 = time.time()
            image = Image.open(frame_path).convert("RGB")
            orig_size = image.size

            input_tensor = transform_fn(image).unsqueeze(0).to(device, dtype=dtype)

            with torch.no_grad():
                preds = birefnet(input_tensor)[-1].sigmoid().cpu()

            pred = preds[0].squeeze()
            mask = transforms.ToPILImage()(pred).resize(orig_size, Image.LANCZOS)

            result = Image.new("RGBA", orig_size, (0, 0, 0, 0))
            result.paste(image, (0, 0), mask)
            result.save(out_path)
            image.close()
            result.close()

            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(frame_files)}] {out_path.name}  ✅  ({elapsed:.1f}s)")
            success += 1

        except torch.cuda.OutOfMemoryError:
            if not fallback_to_cpu:
                print(f"\n  ⚠️  CUDA OOM! 自动回退到 CPU 推理（fp32，会更慢）")
                print(f"   建议下次降低 --size（如 512）或增大 --batch-size")
                # 将模型移到 CPU
                device = "cpu"
                dtype = torch.float32
                birefnet = birefnet.float().to("cpu")
                fallback_to_cpu = True
                # 清理 GPU 缓存
                torch.cuda.empty_cache()
                # 重试当前帧
                try:
                    t0 = time.time()
                    image = Image.open(frame_path).convert("RGB")
                    orig_size = image.size
                    input_tensor = transform_fn(image).unsqueeze(0).to(device, dtype=dtype)
                    with torch.no_grad():
                        preds = birefnet(input_tensor)[-1].sigmoid().cpu()
                    pred = preds[0].squeeze()
                    mask = transforms.ToPILImage()(pred).resize(orig_size, Image.LANCZOS)
                    result = Image.new("RGBA", orig_size, (0, 0, 0, 0))
                    result.paste(image, (0, 0), mask)
                    result.save(out_path)
                    image.close()
                    result.close()
                    elapsed = time.time() - t0
                    print(f"  [{i:3d}/{len(frame_files)}] {out_path.name}  ✅  (CPU, {elapsed:.1f}s)")
                    success += 1
                    continue
                except Exception as e2:
                    print(f"  ❌ CPU 重试也失败: {e2}")
                    continue
            else:
                print(f"  [{i:3d}/{len(frame_files)}] ❌ OOM，跳过")
                continue

        except Exception as e:
            print(f"  [{i:3d}/{len(frame_files)}] ❌ 失败: {e}")
            continue

    summary = f"\n🎉 抠图完成: 成功 {success} 帧"
    if skipped:
        summary += f", 跳过 {skipped} 帧（已存在）"
    if success + skipped < len(frame_files):
        summary += f", 失败 {len(frame_files) - success - skipped} 帧"
    print(summary)
    if fallback_to_cpu:
        print("⚠️  过程中因显存不足回退到了 CPU 模式")
    print(f"   输出目录: {out}/")

    if success == 0:
        print("\n❌ 没有成功处理任何帧，请检查：")
        print("   - 输入图片是否正常（是否损坏的 JPG？）")
        print("   - 模型是否正常加载")
        print("   - 显存是否溢出（试试 --size 512）")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiRefNet 批量抠图")
    parser.add_argument("input_dir", help="输入帧目录")
    parser.add_argument("--output", type=str, default="./matted", help="输出目录")
    parser.add_argument("--model", type=str, default="ZhengPeng7/BiRefNet", help="模型 HuggingFace ID")
    parser.add_argument("--size", type=int, default=1024, help="模型推理尺寸 (默认: 1024，降低可省显存)")
    parser.add_argument("--pattern", type=str, default="frame_*.jpg", help="帧文件名模式")
    parser.add_argument("--no-check", action="store_true", help="跳过依赖预检（用于已确认环境正常时）")
    args = parser.parse_args()

    if not args.no_check:
        check_dependencies()

    matting(
        args.input_dir,
        args.output,
        args.model,
        args.size,
        args.pattern,
    )
