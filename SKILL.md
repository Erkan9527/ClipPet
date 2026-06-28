---
name: ClipPet
description: 视频转透明背景精灵帧动画（ClipPet）。三步走：抽帧 → BiRefNet AI抠图（无需绿幕）→ alpha bbox 居中对齐消除抖动 → 导出精灵序列帧 + GIF。适用于任意角色/物体/UI动效的视频素材转干净可复用的透明帧动画。用户提到"视频转帧""视频抽帧""抠图""去背景""matting""透明背景动画""逐帧动画""spritesheet""精灵帧""帧序列""GIF透明""动画锚定""帧对齐""消除抖动""animation tuner""frame pipeline""帧动画工具"时触发。
---

# ClipPet — 视频转精灵帧动画 Pipeline

```
src.mp4 → [抽帧] → frames_raw/ → [AI抠图] → frames_matted/ → [锚定对齐] → frames_tuned/ + output.gif
```

三阶段顺序执行，每阶段独立可单独运行。

---

## 0️⃣ 依赖预检（先跑这个）

> **渐进式加载说明**：AI Agent 加载此 skill 后，**优先检查依赖是否就绪**，再决定是否进入后续阶段。如依赖不全，直接给出安装指引，不继续执行。

```bash
# 检查所有关键命令是否可用
python scripts/01_extract_frames.py --help       # ffmpeg 自动检测
python scripts/02_birefnet_matting.py --no-check  # 跳过依赖预检
```

## 安装到 AI 编程工具

ClipPet 安装后，用户提到"抠图""去背景""透明动画""视频转帧""精灵帧"等关键词时会**自动触发**。

### OpenCode / Cursor / Windsurf

```bash
# 全局安装（所有项目可用）
git clone https://github.com/Erkan9527/ClipPet.git ~/.config/opencode/skills/ClipPet

# 或在项目中注册（推荐 .opencode.json）
# {
#   "skills": {
#     "ClipPet": {
#       "description": "视频转透明背景精灵帧动画"
#     }
#   }
# }
```

### Claude Code

```bash
# 克隆到技能目录，自动加载
git clone https://github.com/Erkan9527/ClipPet.git ~/.claude/skills/ClipPet
```

### 其他 AI 工具

大多数 AI 编程工具支持将项目克隆到其技能/agents 目录下，或通过项目本地 `.agents/skills/` 目录加载：

```bash
git clone https://github.com/Erkan9527/ClipPet.git .agents/skills/ClipPet
```

## 依赖

**各阶段依赖速查表：**

| 阶段 | 脚本 | 硬依赖 | 可选 |
|------|------|--------|------|
| ① 抽帧 | `01_extract_frames.py` | `ffmpeg` | `pillow`（查帧尺寸） |
| ② 抠图 | `02_birefnet_matting.py` | `torch`, `transformers`, `pillow`, `torchvision` | CUDA（自动检测） |
| ③ 对齐 | `03_animation_tuner.py` | `pillow`, `numpy` | — |

**依赖安装（全量）：**

```bash
# 基础
pip install pillow numpy

# PyTorch（按显卡选）
#   NVIDIA 显卡 → CUDA 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   无 NVIDIA 显卡 → CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# HuggingFace 模型加载
pip install transformers huggingface_hub
```

---

## 1️⃣ 快速开始（最快路径）

```bash
# 一步到位，从头到尾跑完整 pipeline
python scripts/01_extract_frames.py input.mp4 --fps 10 --output frames_raw
python scripts/02_birefnet_matting.py frames_raw --output frames_matted
python scripts/03_animation_tuner.py frames_matted --output frames_tuned --gif output.gif
```

---

## 2️⃣ 各阶段详细说明

### 阶段①: 视频抽帧 — `01_extract_frames.py`

> ⚠️ **兜底**：自动检测 ffmpeg 是否安装，未安装直接给出各平台安装命令。ffmpeg 失败时输出完整 stderr 和排查建议。

```bash
python scripts/01_extract_frames.py <video> [--fps 10] [--output ./frames]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `video` | (必填) | 输入视频路径 |
| `--fps` | `10` | 抽帧帧率 |
| `--output` | `./frames` | 输出 JPG 目录 |

---

### 阶段②: AI 抠图 — `02_birefnet_matting.py`

> 🔄 **兜底链（优先级从高到低）**：
> 1. 启动时预检 `torch` / `transformers` / `pillow` / `torchvision` 是否安装，缺失则提示精确安装命令后退出
> 2. 检测模型是否本地缓存 → 未缓存则探测 HuggingFace 可达性
> 3. HuggingFace 不可达 → 依次尝试 `ZhengPeng7/BiRefNet` → `briaai/RMBG-2.0` → `briaai/RMBG-1.4` 作为备用
> 4. 全部不可达 → 输出解决指引（HF 镜像、手动缓存），**不会静默失败**
> 5. CUDA OOM → 自动将模型转移到 CPU（fp32）重试当前帧
> 6. 单帧处理异常 → 跳过该帧继续（不会中断整个 pipeline）

```bash
python scripts/02_birefnet_matting.py <input_dir> [--output ./matted] [--size 1024] [--model ZhengPeng7/BiRefNet]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_dir` | (必填) | 原始帧目录 |
| `--output` | `./matted` | 输出 RGBA PNG 目录 |
| `--model` | `ZhengPeng7/BiRefNet` | HuggingFace 模型 ID |
| `--size` | `1024` | 模型输入分辨率（降低可省显存） |
| `--pattern` | `frame_*.jpg` | 匹配的帧文件模式 |
| `--no-check` | — | 跳过依赖预检 |

> **模型不可达的解决方法**：
> ```bash
> # 使用 HuggingFace 镜像
> export HF_ENDPOINT=https://hf-mirror.com
> python scripts/02_birefnet_matting.py ...
> ```

---

### 阶段③: 锚定对齐 + GIF 合成 — `03_animation_tuner.py`

核心算法：每帧分析 alpha 通道，计算非透明像素区域的 bounding box，把内容几何中心对齐到画布中心，消除原始视频运动导致的抖动。

> ⚠️ **兜底**：
> - PNG 损坏检测（空文件、无法解码）→ 跳过该帧
> - 完全透明帧 → 跳过
> - GIF 合成失败 → 自动尝试减半帧数重试；PNG 序列始终保留
> - 零有效帧 → 清晰报错

```bash
python scripts/03_animation_tuner.py <input_dir> [--output ./tuned] [--gif ./output.gif] [--padding 30] [--duration 100]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_dir` | (必填) | 抠图帧目录 |
| `--output` | `./tuned` | 对齐帧输出目录 |
| `--gif` | `./output.gif` | 输出 GIF 路径 |
| `--padding` | `30` | 内容到画布边缘的最小边距 |
| `--canvas-w` | `0` | 固定画布宽度（0=自动计算） |
| `--canvas-h` | `0` | 固定画布高度（0=自动计算） |
| `--duration` | `100` | GIF 每帧毫秒数 |
| `--pattern` | `matted_*.png` | 输入帧文件名模式 |

**自动尺寸模式**（`--canvas-w/h=0`）：扫描所有帧的 bbox 最大宽高 + padding，确保所有帧内容完整可见。

---

## 3️⃣ 典型参数速查

| 用途 | fps | duration | 效果 |
|------|-----|----------|------|
| 循环待机动效 | `10` | `100ms` | 流畅，文件小 |
| 快速动作 | `15` | `66ms` | 更丝滑 |
| 慢速卡点 | `6` | `166ms` | 节奏感强 |
| 表情包/GIF | `8` | `125ms` | 体积可控 |

---

## 4️⃣ 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `ffmpeg not found` | 未安装 ffmpeg | 按提示安装 |
| `CUDA OOM` | 显存不足 | 加 `--size 512` 降低分辨率，或换 CPU |
| `模型加载失败` | HuggingFace 连不上 | `export HF_ENDPOINT=https://hf-mirror.com` 重试 |
| `GIF 合成失败` | 帧数太多/内存不足 | 自动减半重试；或手动减少帧数 |
| `未找到匹配文件` | 路径/文件名模式不对 | 检查 `--pattern` 和目录路径 |
| `代理相关报错` | 环境变量异常 | `unset http_proxy https_proxy` 后再试 |

---

## 5️⃣ 产物说明

| 阶段产物 | 格式 | 说明 |
|---------|------|------|
| `frame_001.jpg` | JPG | 原始抽帧（阶段①输出） |
| `matted_001.png` | RGBA PNG | 抠图帧（阶段②输出） |
| `tuned_001.png` | RGBA PNG | 居中对齐帧（阶段③输出） |
| `output.gif` | GIF | 最终动图（阶段③输出） |

对齐帧输出 `tuned_NNN.png` 是画布一致、居中稳定的透明精灵帧序列，可直接用于：

- **Flutter**: `Image.asset('assets/tuned_${i}.png', gaplessPlayback: true)`
- **Web/Canvas**: `new Image()` + `ctx.drawImage()` 逐帧绘制
- **游戏引擎**: 直接作为 sprite 序列帧
- **视频编辑**: 导入 PNG 序列作为透明素材
