# Parrots Examples

本目录包含了 Parrots 库的各种示例程序，帮助你快速上手语音识别（ASR）和语音合成（TTS）功能。

## 📋 目录

- [语音合成 (TTS)](#语音合成-tts)
  - [基础TTS示例](#基础tts示例)
  - [流式TTS示例](#流式tts示例)
  - [Gradio Web界面](#gradio-web界面)
- [语音识别 (ASR)](#语音识别-asr)
- [音频文件说明](#音频文件说明)

---

## 🎤 语音合成 (TTS)

### 基础TTS示例

**文件**: `demo_tts.py`

最简单的TTS使用示例，将文本转换为语音并保存为音频文件。

#### 快速开始

```bash
# 使用默认参数
python demo_tts.py

# 自定义参数
python demo_tts.py \
    --speaker_model shibing624/parrots-gpt-sovits-speaker-maimai \
    --speaker_name MaiMai \
    --device cuda \
    --text "你好，欢迎来北京。Welcome to the city." \
    --lang auto \
    --output_path output_audio.wav
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--speaker_model` | `shibing624/parrots-gpt-sovits-speaker-maimai` | 说话人模型路径 |
| `--speaker_name` | `MaiMai` | 说话人名称 |
| `--device` | `cpu` | 运行设备 (cpu/cuda) |
| `--half` | False | 是否使用半精度 (FP16) |
| `--text` | `你好，欢迎来北京。welcome to the city.` | 要合成的文本 |
| `--lang` | `auto` | 语言 (zh/en/ja/auto) |
| `--output_path` | `output_audio.wav` | 输出音频文件路径 |

#### 支持的说话人

- `MaiMai` - 麦麦
- `XingTong` - 星瞳
- `XuanShen` - 玄神
- `KusanagiNene` - 草薙宁宁
- `LongShouRen` - 龙守人
- `KuileBlanc` - 奎勒布朗

---

### 流式TTS示例

**文件**: `demo_tts_stream.py`

**⭐ 推荐使用！** 支持流式输出的TTS示例，实现超低延迟的实时语音合成和播放。

> 💡 **最新改进**: 查看 [STREAMING_TTS_IMPROVEMENTS.md](STREAMING_TTS_IMPROVEMENTS.md) 了解智能边界切分技术，彻底解决单词截断问题！

#### ✨ 核心特性

- 🚀 **超低延迟**: 首帧延迟从 ~3秒 降低到 ~100-300ms
- 🔄 **滑动窗口解码**: 使用重叠解码保证音频连贯性，避免截断单词
- 📦 **增量输出**: 只输出新增音频部分，无重复播放
- 🎵 **实时播放**: 边生成边播放，无需等待完整音频
- 💾 **完整保存**: 同时支持保存完整音频文件
- 🔧 **灵活配置**: 可调节chunk大小和重叠大小

#### 快速开始

```bash
# 基础使用
python demo_tts_stream.py

# 最低延迟模式（推荐）
python demo_tts_stream.py \
    --prepare_ref \
    --chunk_size 15 \
    --text "你好，欢迎使用流式TTS！"

# 高质量模式
python demo_tts_stream.py \
    --chunk_size 40 \
    --device cuda \
    --text "这是一个高质量的语音合成示例。"
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--speaker_model` | `shibing624/parrots-gpt-sovits-speaker-maimai` | 说话人模型路径 |
| `--speaker_name` | `MaiMai` | 说话人名称 |
| `--device` | `cpu` | 运行设备 (cpu/cuda) |
| `--half` | False | 是否使用半精度 (FP16) |
| `--text` | 默认示例文本 | 要合成的文本 |
| `--lang` | `auto` | 语言 (zh/en/ja/auto) |
| `--chunk_size` | `20` | 流式chunk大小 |
| `--output_path` | `output_stream.wav` | 输出音频文件路径 |
| `--no_play` | False | 禁用实时播放 |
| `--prepare_ref` | False | 预处理参考音频（降低延迟） |

#### Chunk Size 调优指南

| Chunk Size | 延迟 | 音质 | 适用场景 |
|------------|------|------|----------|
| 10-15 | 极低 | 一般 | 实时对话、快速响应 |
| **20-30** | 低 | 好 | **推荐默认值** |
| 40-50 | 中 | 很好 | 高质量内容生成 |

#### 性能对比

| 指标 | 传统TTS | 流式TTS |
|------|---------|---------|
| **首帧延迟** | ~3秒 | ~100-300ms |
| **用户体验** | 等待完整生成 | 边生成边播放 |
| **参考音频** | 每次重复计算 | 缓存复用 |

#### Python API 使用

```python
from parrots.tts import TextToSpeech

# 初始化模型
tts = TextToSpeech(
    speaker_name="MaiMai",
    device="cuda"
)

# 预处理参考音频（可选，但推荐）
tts.prepare_reference()

# 流式生成
for audio_chunk in tts.predict_stream(
    text="你好，这是流式TTS演示",
    text_language="auto",
):
    # 实时处理音频块
    print(f"Generated chunk: {len(audio_chunk)} samples")
    # 可以在这里实时播放或处理音频
```

#### 依赖安装

```bash
# 安装sounddevice用于实时播放
pip install sounddevice
```

---

### Gradio Web界面

**文件**: `tts_gradio_demo.py`

提供一个友好的Web界面来使用TTS功能。

```bash
python tts_gradio_demo.py
```

运行后会在浏览器中打开一个Web界面，可以直接输入文本并试听合成效果。

---

## 🎧 语音识别 (ASR)

**文件**: `demo_asr.py`

语音识别示例，将音频文件转换为文本。

### 快速开始

```bash
# 使用默认模型
python demo_asr.py

# 指定模型
python demo_asr.py \
    --model_name_or_path BELLE-2/Belle-distilwhisper-large-v2-zh
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name_or_path` | `BELLE-2/Belle-distilwhisper-large-v2-zh` | ASR模型路径 |

---

## 🎵 音频文件说明

示例目录中包含了一些测试音频文件：

| 文件名 | 说明 |
|--------|------|
| `ref.wav` | 参考音频文件（用于TTS克隆） |
| `tushuguan.wav` | 测试音频（"图书馆"） |
| `kejiguan.wav` | 测试音频（"科技馆"） |
| `en.wav` | 英文测试音频 |
| `output_audio.wav` | TTS输出示例 |

---

## 💡 使用建议

### 对于TTS任务

1. **快速测试**: 使用 `demo_tts.py`
2. **实时应用**: 使用 `demo_tts_stream.py` + `--prepare_ref`
3. **交互体验**: 使用 `tts_gradio_demo.py`

### 性能优化

1. **使用GPU**: 添加 `--device cuda` 参数
2. **使用半精度**: 添加 `--half` 参数（需要GPU支持）
3. **预处理参考音频**: 添加 `--prepare_ref` 参数（流式TTS）
4. **调整chunk大小**: 根据需求调整 `--chunk_size` 参数

### 常见问题

#### Q: 如何降低TTS延迟？
A: 使用流式TTS (`demo_tts_stream.py`) 并添加 `--prepare_ref` 参数，设置较小的 `--chunk_size` (如15-20)。

#### Q: 如何提高音质？
A: 增大 `--chunk_size` 参数（如40-50），使用GPU (`--device cuda`)。

#### Q: 支持哪些语言？
A: 支持中文(zh)、英文(en)、日文(ja)，以及自动检测(auto)。

#### Q: 如何使用自己的参考音频？
A: 准备3-10秒的参考音频，在代码中指定 `ref_wav_path` 和 `ref_prompt` 参数。

---

## 📚 更多资源

- [Parrots 主仓库](https://github.com/shibing624/parrots)
- [在线文档](https://github.com/shibing624/parrots/blob/main/README.md)
- [问题反馈](https://github.com/shibing624/parrots/issues)

---

## 📄 License

本项目遵循 Apache License 2.0 开源协议。
