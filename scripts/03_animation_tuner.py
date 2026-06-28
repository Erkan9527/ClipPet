#!/usr/bin/env python3
"""
03_animation_tuner.py — 以画布尺寸为锚点，将抠图帧居中对齐 + 合成 GIF（ClipPet）

每帧通过 alpha 通道 bbox 计算内容区域，将内容中心对齐到画布中心，
消除因原始视频运动导致的画面抖动。

用法:
  python 03_animation_tuner.py <输入目录> [--output ./tuned] [--gif ./out.gif] [--padding 30] [--duration 100]

示例:
  python scripts/03_animation_tuner.py frames_matted --output frames_tuned --gif anim.gif
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# 依赖检查
# ============================================================
def check_deps():
    try:
        import PIL  # noqa
        import numpy  # noqa
    except ImportError as e:
        print(f"❌ 缺少依赖: {e.name}")
        print("   pip install pillow numpy")
        sys.exit(1)


# ============================================================
# 帧加载（含校验）
# ============================================================
def load_frame(path: Path):
    """加载一帧 PNG，校验文件完整性。返回 (img, arr) 或抛出异常。"""
    # 先校验文件大小
    if path.stat().st_size == 0:
        raise ValueError(f"空文件: {path}")

    img = Image.open(path)
    # 校验图片完整性：强制加载像素数据
    img.load()

    # 检查是否为 RGBA
    if img.mode != "RGBA":
        # 尝试转换
        img = img.convert("RGBA")

    arr = np.array(img)
    return img, arr


# ============================================================
# 核心处理
# ============================================================
def tune(
    input_dir: str,
    output_dir: str = "./tuned",
    gif_path: str = "./output.gif",
    padding: int = 30,
    canvas_w: int = 0,
    canvas_h: int = 0,
    gif_duration: int = 100,
    frame_pattern: str = "matted_*.png",
):
    src = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 收集帧 ----------
    frames = sorted(src.glob(frame_pattern))
    if not frames:
        print(f"❌ 未找到匹配 {frame_pattern} 的文件")
        print(f"   搜索目录: {src}/")
        all_files = list(src.iterdir())
        if all_files:
            print(f"   目录中共 {len(all_files)} 个文件，前 10 个：")
            for f in all_files[:10]:
                print(f"     - {f.name}")
        sys.exit(1)

    print(f"📐 分析 {len(frames)} 帧...")

    # ---------- Step 1: 分析所有帧 bbox（含损坏检测） ----------
    data = []
    corrupt_count = 0
    empty_count = 0

    for f in frames:
        try:
            img, arr = load_frame(f)
        except Exception as e:
            print(f"   ⚠️  跳过损坏帧 {f.name}: {e}")
            corrupt_count += 1
            continue

        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 10)
        if len(xs) == 0:
            # 完全透明帧：跳过（不能提供有效内容）
            print(f"   ⚠️  跳过完全透明帧 {f.name}")
            img.close()
            empty_count += 1
            continue

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        data.append((f, img, arr, bbox))

    if not data:
        print("❌ 没有有效的帧可供处理")
        if corrupt_count:
            print(f"   损坏帧: {corrupt_count}")
        if empty_count:
            print(f"   空帧: {empty_count}")
        sys.exit(1)

    if corrupt_count or empty_count:
        print(f"   （跳过损坏 {corrupt_count} 帧，空帧 {empty_count} 帧）")

    # ---------- 全局最大内容尺寸 ----------
    max_cw = max(b[2] - b[0] + 1 for _, _, _, b in data)
    max_ch = max(b[3] - b[1] + 1 for _, _, _, b in data)

    cw = canvas_w if canvas_w > 0 else max_cw + padding * 2
    ch = canvas_h if canvas_h > 0 else max_ch + padding * 2

    print(f"   有效帧:       {len(data)}")
    print(f"   内容最大宽高: {max_cw}×{max_ch}")
    print(f"   锚定画布:     {cw}×{ch}")

    # ---------- Step 2: 居中锚定 ----------
    print(f"\n🎯 逐帧居中锚定...")
    tuned_frames = []
    failed_count = 0

    for i, (fpath, orig_img, _, (x1, y1, x2, y2)) in enumerate(data, 1):
        try:
            cw_frame = x2 - x1 + 1
            ch_frame = y2 - y1 + 1

            paste_x = cw // 2 - cw_frame // 2 - x1
            paste_y = ch // 2 - ch_frame // 2 - y1

            canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            canvas.paste(orig_img, (paste_x, paste_y), orig_img)
            orig_img.close()

            out_path = out / f"tuned_{i:03d}.png"
            canvas.save(out_path)
            canvas.close()
            tuned_frames.append(out_path)

            if i % 11 == 0 or i == len(data):
                print(f"  [{i:3d}/{len(data)}]  ✅")

        except Exception as e:
            print(f"  [{i:3d}/{len(data)}] ❌ 处理失败: {e}")
            orig_img.close()
            failed_count += 1
            continue

    if not tuned_frames:
        print("❌ 没有成功生成任何对齐帧")
        sys.exit(1)

    if failed_count:
        print(f"   （{failed_count} 帧处理失败）")

    # ---------- Step 3: 合成 GIF ----------
    print(f"\n🎬 合成 GIF ({len(tuned_frames)} 帧, {gif_duration}ms/帧)...")

    try:
        imgs = [Image.open(f) for f in tuned_frames]

        # disposal=2 表示每帧恢复背景（透明帧正确显示的关键）
        # loop=0 表示无限循环
        imgs[0].save(
            Path(gif_path),
            save_all=True,
            append_images=imgs[1:],
            duration=gif_duration,
            loop=0,
            disposal=2,
            optimize=False,
            transparency=0,
        )
        for img in imgs:
            img.close()

    except Exception as e:
        # 如果 GIF 合成失败，PNG 序列仍然有效
        print(f"⚠️  GIF 合成失败: {e}")
        print(f"   对齐帧仍然可用: {out}/")
        # 尝试使用较老的方式（单帧或减少帧数）
        if len(tuned_frames) > 1:
            print("   尝试减半帧数重试...")
            reduced = tuned_frames[::2]
            try:
                imgs = [Image.open(f) for f in reduced]
                imgs[0].save(
                    Path(gif_path),
                    save_all=True,
                    append_images=imgs[1:],
                    duration=gif_duration * 2,
                    loop=0,
                    disposal=2,
                )
                for img in imgs:
                    img.close()
                print(f"   ✅ GIF 合成成功（减半帧数）: {gif_path}")
            except Exception as e2:
                print(f"   ❌ 减半重试也失败: {e2}")
                gif_path = None
        else:
            gif_path = None

    # ---------- 输出统计 ----------
    print(f"\n{'=' * 50}")
    print(f"✅ Animation Tuner 完成!")
    print(f"   对齐帧: {out}/ ({len(tuned_frames)} 帧)")
    if gif_path and Path(gif_path).exists():
        gif_size_kb = Path(gif_path).stat().st_size / 1024
        print(f"   GIF:    {gif_path} ({gif_size_kb:.0f} KB)")
    print(f"   画布:   {cw}×{ch}")
    if failed_count:
        print(f"   失败:   {failed_count} 帧")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抠图帧对齐 + GIF 合成")
    parser.add_argument("input_dir", help="抠图帧目录")
    parser.add_argument("--output", type=str, default="./tuned", help="输出目录")
    parser.add_argument("--gif", type=str, default="./output.gif", help="输出 GIF 路径")
    parser.add_argument("--padding", type=int, default=30, help="画布边距 (默认: 30)")
    parser.add_argument("--canvas-w", type=int, default=0, help="画布宽度 (0=自动)")
    parser.add_argument("--canvas-h", type=int, default=0, help="画布高度 (0=自动)")
    parser.add_argument("--duration", type=int, default=100, help="每帧毫秒 (默认: 100)")
    parser.add_argument("--pattern", type=str, default="matted_*.png", help="输入帧模式")
    args = parser.parse_args()

    check_deps()
    tune(
        args.input_dir,
        args.output,
        args.gif,
        args.padding,
        args.canvas_w,
        args.canvas_h,
        args.duration,
        args.pattern,
    )
