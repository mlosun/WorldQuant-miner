# Naive-Ollama Alpha 因子生成系统

一个基于本地Ollama大语言模型的智能Alpha因子生成系统，能够自动生成、测试并提交Alpha因子到WorldQuant Brain平台。该系统完全基于本地部署，无需依赖云服务，提供更好的性能和隐私保护。

## 🚀 核心特性

- **本地LLM集成**: 使用Ollama运行llama3.2:3b、deepseek-r1:8b等模型
- **智能模型舰队管理**: 自动VRAM监控和模型降级，确保系统稳定运行
- **并发执行架构**: Alpha生成器和挖掘器同时运行，提高效率
- **Web监控界面**: 实时监控和控制界面，支持状态查看和手动操作
- **自动化工作流**: 连续Alpha生成、挖掘和提交，支持定时任务
- **WorldQuant Brain集成**: 直接API集成，支持测试和提交
- **Conda环境支持**: 完全基于Python环境，无需Docker

## 📋 系统要求

- **Python 3.8+**: 支持现代Python特性
- **Conda环境**: 推荐使用conda管理依赖
- **Ollama服务**: 本地运行Ollama服务器
- **WorldQuant Brain账户**: 用于Alpha测试和提交
- **GPU支持** (可选): NVIDIA GPU加速推理

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web监控界面   │    │  Alpha生成器    │    │  WorldQuant API │
│   (Flask)       │◄──►│   (Ollama)      │◄──►│   (外部服务)    │
│   端口 5000     │    │   端口 11434    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────►│ Alpha编排器     │◄─────────────┘
                        │   (Python)      │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   结果和日志    │
                        │     存储       │
                        └─────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate pytorch

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置凭据

创建`credential.txt`文件，包含你的WorldQuant Brain凭据：
```json
["your.email@worldquant.com", "your_password"]
```

### 3. 启动Ollama服务

```bash
# 启动Ollama服务
ollama serve

# 拉取模型（如果还没有）
ollama pull deepseek-r1:8b
ollama pull llama3.2:3b
```

### 4. 启动系统

```bash
# 启动连续挖掘模式（推荐）
python alpha_orchestrator.py --mode continuous --credentials credential.txt --batch-size 1 --max-concurrent 1 --restart-interval 480

# 或者单独运行生成器
python alpha_generator_ollama.py --credentials credential.txt --ollama-url "http://localhost:11434" --ollama-model deepseek-r1:8b
```

### 5. 访问Web界面

打开浏览器访问：http://localhost:5000

## 📊 Web监控界面功能

### 状态监控
- **Ollama状态**: 模型加载、API连接状态
- **编排器状态**: 生成活动、挖掘计划
- **WorldQuant状态**: API连接、认证状态
- **统计信息**: 生成的Alpha、成功率、24小时指标

### 手动控制
- **生成Alpha**: 触发单次Alpha生成
- **触发挖掘**: 运行Alpha表达式挖掘
- **触发提交**: 提交成功的Alpha
- **刷新状态**: 更新所有指标

### 实时日志
- **Alpha生成器日志**: 过滤显示Alpha生成活动
- **系统日志**: 完整系统活动
- **最近活动**: 事件时间线

## 🔧 配置说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `continuous` | 运行模式: `daily`, `continuous`, `miner`, `submitter`, `generator` |
| `--max-concurrent` | `1` | 最大并发模拟数量 |
| `--mining-interval` | `6` | 连续模式下的挖掘间隔（小时） |
| `--batch-size` | `1` | 操作批次大小 |
| `--credentials` | `./credential.txt` | 凭据文件路径 |
| `--ollama-url` | `http://localhost:11434` | Ollama API URL |
| `--ollama-model` | `deepseek-r1:8b` | 使用的Ollama模型 |
| `--restart-interval` | `30` | 进程重启间隔（分钟） |

### 运行模式

1. **continuous** (默认): 同时运行生成器和挖掘器
2. **daily**: 运行完整的每日工作流（生成器 → 挖掘器 → 提交器）
3. **miner**: 仅运行Alpha表达式挖掘器
4. **submitter**: 仅运行Alpha提交器
5. **generator**: 仅运行Alpha生成器

## 🧠 智能模型舰队管理

### 模型层次结构

系统维护从大到小的模型舰队：
1. **deepseek-r1:8b** (5.2 GB) - 大型模型 - 8B参数，推理能力强
2. **llama3.2:3b** (2.0 GB) - 中型模型 - 3B参数，平衡性能
3. **deepseek-r1:1.5b** (1.1 GB) - 小型模型 - 1.5B参数，低内存
4. **phi3:mini** (2.2 GB) - 紧急备用模型

### 自动VRAM管理

- **实时监控**: 每30秒监控VRAM使用情况
- **错误检测**: 自动检测VRAM相关错误
- **智能降级**: 连续3次错误后自动降级到更小模型
- **状态持久化**: 保存模型选择和错误计数到`model_fleet_state.json`

### VRAM错误模式识别

系统监控以下VRAM错误模式：
- `"gpu VRAM usage didn't recover within timeout"`
- `"VRAM usage didn't recover"`
- `"gpu memory exhausted"`
- `"CUDA out of memory"`
- `"GPU memory allocation failed"`

## 📁 文件结构

```
naive-ollama/
├── alpha_generator_ollama.py      # 主要Alpha生成脚本
├── alpha_orchestrator.py          # 编排和调度
├── alpha_expression_miner.py      # Alpha表达式挖掘
├── improved_alpha_submitter.py    # Alpha提交到WorldQuant
├── template_grid_miner.py         # 模板网格挖掘器
├── web_dashboard.py               # Flask Web监控界面
├── model_fleet_manager.py         # 模型舰队管理器
├── vram_monitor.py                # VRAM监控
├── templates/                     # Web界面模板
├── results/                       # 生成的Alpha结果
├── credential.txt                 # WorldQuant凭据
├── requirements.txt               # Python依赖
├── model_fleet_state.json         # 模型舰队状态
├── hopeful_alphas.json            # 有希望的Alpha列表
└── submission_log.json            # 提交日志
```

## 🔄 工作流程

### 1. Alpha生成
- **连续模式**: 每6小时生成Alpha
- **批次处理**: 每批次生成1个Alpha
- **Ollama集成**: 使用本地LLM生成Alpha想法
- **WorldQuant测试**: 立即测试每个Alpha

### 2. Alpha挖掘
- **表达式挖掘**: 分析有希望的Alpha寻找变体
- **模式识别**: 识别成功的Alpha模式
- **优化建议**: 对现有Alpha提出改进建议

### 3. Alpha提交
- **每日限制**: 每天仅提交一次
- **成功过滤**: 只提交性能良好的Alpha
- **速率限制**: 遵守WorldQuant API限制

## 📈 监控和日志

### 实时指标
- **Alpha生成率**: 每小时生成的Alpha数量
- **成功率**: 成功Alpha的百分比
- **GPU利用率**: 内存和计算使用情况
- **API响应时间**: WorldQuant API性能

### 日志文件
- `alpha_orchestrator.log`: 主要编排器日志
- `alpha_generator_ollama.log`: 生成器日志
- `alpha_miner.log`: 挖掘器日志
- `improved_alpha_submitter.log`: 提交器日志

### 关键日志消息
- `"Both alpha generator and expression miner are running concurrently"`
- `"Max concurrent simulations: 1"`
- `"Found X alphas to mine"`
- `"Alpha generator started with PID: X"`

## 🛠️ 故障排除

### 常见问题

#### 1. hopeful_alphas.json未找到
- 启动期间正常 - 生成器需要时间创建有希望的Alpha
- 检查生成器日志中的错误

#### 2. 认证失败
- 验证`credential.txt`存在且格式正确
- 检查WorldQuant Brain API状态

#### 3. 进程崩溃
- 编排器将自动重启失败的进程
- 检查日志中的具体错误消息

#### 4. 并发模拟过多
- 减少`--max-concurrent`值
- 检查WorldQuant Brain API限制

#### 5. API速率限制 (429错误)
- 暂停手动提交，等待冷却期
- 减少提交批次大小
- 避免"双重提交"（手动+计划）

### 性能调优

- **提高吞吐量**: 增加`--max-concurrent`（但遵守API限制）
- **减少资源使用**: 减少`--batch-size`
- **更快挖掘**: 减少`--mining-interval`
- **更频繁检查**: 修改`start_alpha_expression_miner_continuous`中的检查间隔

### VRAM问题解决

如果VRAM问题持续存在：
1. 减少`--batch-size`
2. 降低GPU层设置
3. 减少内存限制
4. 增加VRAM清理间隔

## 🔒 安全特性

- **本地处理**: 所有LLM推理都在本地进行
- **凭据保护**: 凭据存储在本地文件中
- **网络隔离**: 仅与WorldQuant Brain API通信
- **API速率限制**: 遵守外部API限制

## 📝 日志记录

### 日志级别
- **INFO**: 正常操作消息
- **WARNING**: 非关键问题
- **ERROR**: 关键故障
- **DEBUG**: 详细调试信息

### 日志位置
- **应用日志**: `./`目录中的各种.log文件
- **Web监控界面**: 实时日志显示

## 🚀 高级功能

### 模板网格挖掘

使用`template_grid_miner.py`进行结构化Alpha生成：
```bash
python template_grid_miner.py --credentials credential.txt --max 100 --batch-size 5
```

### 模型舰队状态管理

```bash
# 检查当前舰队状态
python alpha_orchestrator.py --mode fleet-status

# 重置到最大模型
python alpha_orchestrator.py --mode fleet-reset

# 强制降级到下一个更小模型
python alpha_orchestrator.py --mode fleet-downgrade
```

## 🔮 未来增强

1. **性能指标**: 跟踪模型性能与VRAM使用的关系
2. **智能调度**: 在低使用期间使用更大的模型
3. **多GPU支持**: 在多个GPU上分布模型
4. **自定义模型支持**: 添加对自定义微调模型的支持
5. **动态阈值**: 根据模型大小调整错误阈值

## 📞 支持

如有问题和疑问：
1. 检查故障排除部分
2. 查看相关日志文件
3. 检查Web监控界面的实时状态
4. 在GitHub上提交问题

---

**祝您Alpha生成愉快！🚀**
