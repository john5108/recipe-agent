# 外卖商家菜品研发与上架助手

一款基于 LangGraph 的外卖菜单研发与上架助手，面向外卖中小商家与餐饮创业者。它不是通用菜谱生成器，而是把食材和经营需求整理成可继续编辑、配图和上架的菜品方案。

> 新对话或新协作者请先读 [PROJECT_BRIEF.md](PROJECT_BRIEF.md)。
>
> 详细文档：[项目状态](docs/PROJECT_STATUS.md) · [路线图](docs/ROADMAP.md) · [同类项目调研](docs/COMPETITOR_RESEARCH.md)

## 当前产品定位

当前聚焦三件事：

1. **图片识材**：用户拍摄冰箱、备餐台或食材堆照片，系统识别其中食材，再组合成可做的外卖菜品。
2. **文字研发**：用户直接输入手里的食材、口味或经营约束，系统给出多道可落地的菜品方案。
3. **上架素材**：为研发出的菜品补齐菜名、卖点、成本与售价、上架文案和菜品图，服务后续小程序或外卖平台上架。

规划中但尚未实现：季节与祛湿等功效标签、结构化成本/毛利字段、批量菜单、CSV 导出，以及一键打包上传外卖平台。现阶段先把“识材 → 研发 → 结构化结果 → 配图”链路做稳。

## 项目来源与二次开发

本项目最初跟随网课示例搭建，后续在原项目基础上做了功能延伸和工程改造：

- 增加 SQLite 菜品研发历史库与多会话记录。
- 增加多模态食材识别、Tavily 参考图搜索和通义万相生图流程。
- 增加 Chainlit 会话界面、参考图展示和手动生图按钮。
- 增加 `bind_tools` 工具循环、LangGraph checkpoint、结构化输出和上下文预算。
- 修复闲聊空状态、图片只识材、JSON 截断和生图模型配置问题。

## 功能特点

- 🧠 **全流程智能体编排**：基于 LangGraph 构建有状态工作流，串联需求分析、菜谱研发、文案生成、图片处理四大环节，支持多轮迭代打磨菜品方案
- 🧰 **模型工具调用**：模型通过 `bind_tools` 选择 Tavily 搜索、OSS 上传、历史菜品查询等工具，工具结果返回模型后继续生成
- 🖼️ **多模态食材识别**：上传食材实拍图，自动识别食材种类与分量，匹配生成可落地的外卖菜品配方
- 🔍 **参考图搜索**：接入 Tavily 搜索菜品参考图；Tavily 属于可选能力，需要有效 API Key，失败时会跳过并提示原因
- 📝 **上架文案一键生成**：自动输出符合外卖平台规范的菜品标题、卖点描述、口味标签、食材说明，可直接复制上架
- 🎨 **菜品图生成**：通过 DashScope `qwen-image-2.1-turbo` 生成外卖风格菜品图；同时保留阿里云 OSS 上传工具，便于后续托管图片外链
- 💾 **多轮状态与菜品库**：LangGraph `MemorySaver` 按 `thread_id` 保存会话内多轮消息；菜品研发记录使用 SQLite 持久化
- ⚡ **上下文控制**：限制送入模型的历史消息数量，参考图最多取 3 张，视觉分析只分析前 2 张，降低多模态输入成本
- ✅ **结构化输出校验**：菜品 JSON 通过 Pydantic `RecipeOutput` 校验，避免坏数据直接进入界面


## 技术栈

- **前端交互**: Chainlit
- **智能体框架**: LangGraph + LangChain
- **搜索工具**: Tavily API
- **大模型**: OpenAI-compatible API（支持 Qwen / DeepSeek / OpenAI）
- **存储方案**: 阿里云OSS + SQLite
- **开发语言**: Python

## 安装步骤

1. 克隆项目：
   ```bash
   git clone https://github.com/john5108/recipe-agent.git
   cd recipe-agent
   ```

2. 创建虚拟环境并安装依赖：

   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   # macOS / Linux
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. 配置环境变量：

   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入你的API密钥
   ```

4. 运行应用：

   ```powershell
   python run.py
   ```

默认访问地址为 `http://127.0.0.1:8001`。`run.py` 会自动切换到项目目录，并检查依赖和 `.env`。

## 环境变量说明

- `TAVILY_API_KEY`: Tavily搜索API密钥
- `WANX_MODEL`: 文生图模型，默认 `qwen-image-2.1-turbo`
- `OPENAI_API_KEY`: OpenAI API密钥
- `ALIBABA_CLOUD_ACCESS_KEY_ID`: 阿里云Access Key ID
- `ALIBABA_CLOUD_ACCESS_KEY_SECRET`: 阿里云Access Key Secret
- `OSS_ENDPOINT`: OSS端点
- `OSS_BUCKET_NAME`: OSS存储空间名称
- `MULTIMODAL_MODEL`: 多模态模型名称
- `TEXT_MODEL`: 文本模型名称
- `MAX_TOKENS`: 单次模型输出上限，默认 4000
- `MODEL_ENABLE_THINKING`: Qwen 思考模式开关，默认 `false`
- `MAX_CONTEXT_MESSAGES`: 送入模型的历史消息条数上限，默认 12
- `MAX_CONTEXT_CHARS`: 历史上下文总字符预算，默认 12000
- `MAX_TOOL_RESULT_CHARS`: 单次工具结果字符上限，默认 8000
- `QWEN_API_KEY` / `DEEPSEEK_API_KEY`: 可选模型供应商密钥

## 使用方法

1. 启动应用后，输入食材，或上传一张冰箱/食材照片
2. 智能体会识别食材并生成多道外卖菜品方案
3. 可以直接问「查询包含土豆的历史菜品」，模型会调用 SQLite 查询工具
4. 菜谱结构化成功后，可查看参考图并点击按钮生成菜品优化图

## 验证与测试

不需要模型额度即可运行离线验收：

```bash
python test_app.py
python scripts/verify_week2.py
```

- `test_app.py`：11 项基础检查，包含工具加载、SQLite 查询、`bind_tools`、checkpointer、意图分流、JSON 修复和 Pydantic schema 校验。
- `scripts/verify_week2.py`：用假模型发起真实 `tool_call`，查询真实 SQLite，并验证两轮 checkpoint 对话。

真实模型验证需要有效额度：

```bash
python scripts/verify_week2_real.py
```

如果供应商返回 `Free quota exhausted`，需要在模型服务商控制台充值或关闭“仅使用免费额度”模式。

## 当前实现边界

- `MemorySaver` 只在当前 Python 进程内保存对话状态，重启服务后会话消息会丢失；SQLite 中的菜品研发记录仍会保留。
- 生产环境可以把 checkpointer 换成 SQLite/Postgres 版本。
- MCP、multi-agent 和会话摘要中间件尚未实现，当前不要把这些写成已完成能力。
- 菜品图生成依赖 `dashscope`、`QWEN_API_KEY` 和对应额度；模型应使用 `qwen-image-*` 或 `wanx*`，不能填写 `qwen-mt-image-*`。
- Tavily 参考图是可选能力。API Key 缺失、过期或额度不足时会跳过参考图，并在页面提示原因。
- OSS 上传和外卖平台一键打包上传尚未做真实端到端验收。

## 工具调用与 MCP

- 当前项目使用 LangChain `@tool` + `bind_tools`。SQLite、Tavily 和本地图片处理都在同一个 Python 进程内，链路短、调试成本低。
- MCP 是外部工具的标准化接入协议，不会绕过 API Key、额度或权限校验。Tavily MCP 仍然需要有效的 Tavily Key。
- 对本地 SQLite 和当前几个简单 API，直接函数工具更合适；当需要接入多个外部服务、远程数据库或百炼 MCP 平台时，再把 MCP 作为独立连接层接入。
- 当前阶段不替换现有工具协议。后续可以用一个百炼 MCP Server 做对照验证，而不是一次性重写全部工具。

## 未来扩展功能

- 菜名、价格和毛利的结构化字段，支持成本误差统计与批量编辑。
- 季节、口味和祛湿等功效标签；功效表述需避免医疗夸大。
- 多商家画像、批量菜单研发和 CSV 导出。
- 人工确认、改价、审批和菜品库状态流转。
- 生成外卖平台所需素材包，并预留一键上传能力。
- 图片识别后自动组合多组菜品，而不是只取第一张图片。

## 项目结构

```
recipe-agent/
├── app/                 # 应用核心代码
│   ├── agent.py         # LangGraph智能体
│   ├── ui.py            # Chainlit界面
│   └── schemas.py       # Pydantic 结构化输出
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── ROADMAP.md
│   └── COMPETITOR_RESEARCH.md
├── tools/              # 自定义工具
│   ├── recipe_tools.py # SQLite、Tavily、OSS 工具
│   └── image_utils.py  # 参考图搜索与视觉特征分析
├── config/             # 配置文件
│   └── settings.py     # 环境配置
├── utils/              # 工具函数
│   └── dish_tracker.py # SQLite 菜品研发历史
├── scripts/
│   ├── verify_week2.py # 离线工具调用与多轮 checkpoint 验收
│   └── verify_week2_real.py # 真实模型工具调用验收（消耗额度）
├── test_app.py         # 基础测试
├── run.py              # 本地启动与环境检查脚本
├── PROJECT_BRIEF.md    # 新对话恢复上下文
├── chainlit.md         # Chainlit 欢迎页
├── requirements.txt    # 依赖包
└── .env.example        # 环境变量示例
```

`.env`、`dish_development.db`、虚拟环境、上传文件和缓存不会提交到 GitHub。

## 注意事项

- 需要有效的API密钥才能正常运行
- 使用多模态模型会产生较高的API费用，请注意成本控制
- 首次运行可能需要较长时间加载模型

## 常见问题

- **Tavily 返回 401**：说明 Key 无效或已过期。替换 `.env` 中的 `TAVILY_API_KEY` 后必须重启 Chainlit，环境变量只在进程启动时加载。
- **生图返回 `InvalidParameter`**：确认 `WANX_MODEL=qwen-image-2.1-turbo`，不能使用 `qwen-mt-image-*`。
- **模型返回空内容**：保持 `MODEL_ENABLE_THINKING=false`，并确认 `MAX_TOKENS` 不低于 4000。
- **闲聊提示 `NoneType`**：旧版本问题，当前已通过 `normalize_final_recipe()` 修复。
