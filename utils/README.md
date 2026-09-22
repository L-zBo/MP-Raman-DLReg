# Utils 工具模块

本目录包含项目的核心工具模块，提供配置管理、日志记录、数据验证以及统一模型口径。

## 📁 文件列表

| 文件名 | 功能 | 状态 |
|--------|------|------|
| `config.py` | 配置管理（单例模式） | ✅ 核心模块 |
| `logger.py` | 日志记录系统 | ✅ 核心模块 |
| `validation.py` | 数据验证工具 | ✅ 辅助模块 |
| `model_identity.py` | 最终模型口径与最终参数登记 | ✅ 核心模块 |
| `model_identity.md` | `model_identity.py` 说明文档 | ✅ 说明文档 |
| `__init__.py` | 模块导出 | ✅ 必需 |

---

## 🎯 模块功能概览

### 1. config.py - 配置管理 ⭐⭐⭐⭐⭐

**用途**：统一加载和管理项目配置

**核心特性**：
- 单例模式，全局唯一配置实例
- 支持点分隔的嵌套访问（如 `preprocessing.als.lambda`）
- 自动将相对路径转换为绝对路径
- 从 `config.yaml` 加载配置

**使用示例**：
```python
from utils import get_config, get

config = get_config()

# 获取配置
als_lambda = config.get('preprocessing.als.lambda', 1e5)
output_dir = config.get_output_dir('models')

# 快捷方式
als_lambda = get('preprocessing.als.lambda', 1e5)
```

---

### 2. logger.py - 日志记录 ⭐⭐⭐⭐⭐

**用途**：提供统一的日志记录功能

**核心特性**：
- 支持控制台和文件双输出
- 可配置日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
- 统一的日志格式
- 从配置文件读取日志设置

**使用示例**：
```python
from utils import get_logger, log_info

# 获取logger实例
logger = get_logger('my_module')
logger.info("处理开始")
logger.warning("发现异常")

# 快捷方式
log_info("处理开始")
```

---

### 3. validation.py - 数据验证 ⭐⭐

**用途**：提供数据完整性检查和验证功能

**核心函数**：
- `clean_spectrum()` - 清理光谱异常值（NaN/Inf）
- `validate_spectrum()` - 验证光谱数据
- `validate_hypercube()` - 验证高光谱数据
- `safe_load_npy()` - 安全加载npy文件

**使用示例**：
```python
from utils import clean_spectrum, validate_spectrum

# 清理异常值
cleaned = clean_spectrum(spectrum, fill_nan=0.0, clip_negative=True)

# 验证数据
is_valid, errors = validate_spectrum(spectrum, expected_length=1024)
```

---

### 4. model_identity.py - 统一模型口径 ⭐⭐⭐⭐⭐

**用途**：统一记录当前最终模型池、最终采用版本和最终参数口径。

**说明**：
- 主对比实验和最终模型引用应以该文件为准
- 箱型图专用参数单独保存在 `visualization/boxplot/boxplot_analysis.py`
- 详细说明见 `model_identity.md`

---

## 🚀 快速使用

```python
from utils import get_config, get_logger, clean_spectrum

# 配置
config = get_config()
output_dir = config.get_output_dir('results')

# 日志
logger = get_logger('preprocess')
logger.info("开始处理")

# 数据清理
cleaned = clean_spectrum(raw_spectrum)
```

---

## 🔗 相关文件

- **配置文件**：`config.yaml` - 项目配置
- **日志目录**：`logs/` - 日志文件存储
- **输出目录**：`output/` - 结果输出

---

**最后更新**：2026-01-24
