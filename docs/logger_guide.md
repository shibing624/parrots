# Parrots Logger 使用指南

## 简介

Parrots 提供了统一的日志管理功能，基于 `loguru` 库实现，支持灵活的日志级别控制。

## 快速开始

### 1. 使用默认日志级别（INFO）

```python
from parrots import logger

logger.debug("这条调试信息不会显示")
logger.info("这条信息会显示")
logger.warning("这条警告会显示")
logger.error("这条错误会显示")
```

### 2. 动态修改日志级别

```python
from parrots import logger, set_log_level

# 设置为 DEBUG 级别，查看更详细的日志
set_log_level("DEBUG")
logger.debug("现在调试信息可以看到了！")

# 设置为 WARNING 级别，只显示警告和错误
set_log_level("WARNING")
logger.info("这条信息不会显示")
logger.warning("这条警告会显示")
```

### 3. 通过环境变量设置日志级别

在运行程序前设置环境变量：

**Linux/Mac:**
```bash
export PARROTS_LOG_LEVEL=DEBUG
python your_script.py
```

**Windows:**
```cmd
set PARROTS_LOG_LEVEL=DEBUG
python your_script.py
```

## 支持的日志级别

按严重程度从低到高排列：

- `TRACE`: 最详细的跟踪信息
- `DEBUG`: 调试信息
- `INFO`: 一般信息（默认级别）
- `SUCCESS`: 成功信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误信息

## 完整示例

```python
from parrots import logger, set_log_level
from parrots.tts import TextToSpeech

# 开发调试时，使用 DEBUG 级别
set_log_level("DEBUG")

# 初始化模型（会显示详细的加载信息）
m = TextToSpeech(
    speaker_model_path="shibing624/parrots-gpt-sovits-speaker-maimai",
    speaker_name="MaiMai",
    device="cpu",
)

# 生产环境中，使用 WARNING 级别减少日志输出
set_log_level("WARNING")

# 进行语音合成
m.predict(
    text="你好，欢迎使用 Parrots！",
    text_language="zh",
    output_path="output.wav"
)
```

## API 参考

### `logger`

全局日志记录器实例，支持以下方法：

- `logger.trace(message)`: 记录跟踪信息
- `logger.debug(message)`: 记录调试信息
- `logger.info(message)`: 记录一般信息
- `logger.success(message)`: 记录成功信息
- `logger.warning(message)`: 记录警告信息
- `logger.error(message)`: 记录错误信息
- `logger.critical(message)`: 记录严重错误信息

### `set_log_level(level: str)`

设置全局日志级别。

**参数:**
- `level`: 日志级别字符串，可选值：`"TRACE"`, `"DEBUG"`, `"INFO"`, `"SUCCESS"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`

**示例:**
```python
from parrots import set_log_level

set_log_level("DEBUG")  # 启用调试日志
set_log_level("WARNING")  # 只显示警告和错误
```

### `get_logger()`

获取配置好的日志记录器实例。

**返回:**
- `logger`: loguru.Logger 实例

**示例:**
```python
from parrots import get_logger

logger = get_logger()
logger.info("使用获取的 logger 实例")
```

## 注意事项

1. 日志级别设置是全局的，会影响所有使用 parrots logger 的模块
2. 环境变量 `PARROTS_LOG_LEVEL` 只在首次导入时生效
3. 如果需要在运行时动态调整日志级别，请使用 `set_log_level()` 函数
4. 日志输出到标准错误流（stderr），不会干扰标准输出（stdout）

## 更多示例

查看 `examples/demo_logger.py` 获取更多使用示例。
