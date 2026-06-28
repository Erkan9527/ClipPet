#!/usr/bin/env python3
"""
01_extract_frames.py — 从视频按帧率抽帧（ClipPet）

用法:
  python 01_extract_frames.py <视频路径> [--fps 10] [--output ./frames]

示例:
  python scripts/01_extract_frames.py demo.mp4 --fps 10 --output frames_raw
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg():
    """检查 ffmpeg 是否可用，返回版本信息或抛出友好错误。"""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("❌ 未找到 ffmpeg，请先安装：")
        print()
        print("   # macOS:")
        print("   brew install ffmpeg")
        print()
        print("   # Ubuntu/Debian:")
        print("   sudo apt install ffmpeg")
        print()
        print("   # Arch Linux:")
        print("   sudo pacman -S ffmpeg")
        print()
        print("   # Windows (choco):")
        print("   choco install ffmpeg")
        sys.exit(1)
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True, text=True, timeout=10,
        )
        first_line = result.stdout.split("\n")[0] if result.stdout else ffmpeg_path
        print(f"✅ ffmpeg: {first_line}")
    except Exception as e:
        print(f"⚠️  ffmpeg 路径存在但执行失败: {e}")
        sys.exit(1)
    return ffmpeg_path


def extract_frames(video_path: str, fps: int = 10, output_dir: str = "./frames"):
    src = Path(video_path)
    if not src.exists():
        print(f"❌ 视频文件不存在: {src}")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 检查输出目录是否非空，提醒用户
    existing = list(out.glob("frame_*.jpg"))
    if existing:
        print(f"⚠️  输出目录已有 {len(existing)} 帧，将覆盖: {out}/")

    pattern = str(out / "frame_%03d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",
        pattern,
    ]

    print(f"🎬 抽帧: fps={fps}, 输出目录={out}/")
    print(f"   命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else "(无错误输出)"
        print(f"❌ ffmpeg 抽帧失败（退出码 {e.returncode}）")
        print(f"   错误: {stderr[:500]}")
        print()
        print("可能的原因：")
        print("  - 视频文件损坏或格式不受支持")
        print("  - 视频编码需要额外解码器（试试 ffprobe 查看编码格式）")
        print("  - fps 值异常")
        sys.exit(1)
    except FileNotFoundError:
        # shutil.which 已检查过，但兜底处理
        print("❌ ffmpeg 未找到（尽管预检通过，可能被中途移除）")
        sys.exit(1)

    files = sorted(out.glob("frame_*.jpg"))
    if not files:
        print("⚠️  ffmpeg 执行成功但未生成任何帧，请检查视频文件")
        sys.exit(1)

    print(f"✅ 完成: {len(files)} 帧 → {out}/")
    # 打印前三帧和后三帧的尺寸作为参考
    try:
        from PIL import Image
        sample = Image.open(files[0])
        print(f"   帧尺寸: {sample.width}×{sample.height}")
        sample.close()
    except ImportError:
        pass  # PIL 不是抽帧的硬依赖
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频按帧率抽帧")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("--fps", type=int, default=10, help="抽帧帧率 (默认: 10)")
    parser.add_argument("--output", type=str, default="./frames", help="输出目录")
    args = parser.parse_args()

    check_ffmpeg()
    extract_frames(args.video, args.fps, args.output)
