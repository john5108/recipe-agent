# GitHub 同类项目调研

> 调研时间：2026-10-25

## 调研范围

检索了 GitHub Topics、公开仓库页面和 README，重点查看以下方向：

- 冰箱照片识材和菜谱生成。
- 餐厅菜单工程、成本和毛利分析。
- 食谱、价格和 Agent 工具链。
- 库存和菜谱相关 MCP Server。

这是一份方向性样本，不是穷尽式市场调研。

## 消费级识材与菜谱生成

### FridgeFusion

[SauravSrivastav/fridgefusion-st](https://github.com/SauravSrivastav/fridgefusion-st)

- 冰箱照片识别食材。
- Gemini Vision 识别，OpenAI 生成菜谱。
- 面向家庭用户。

可借鉴：识图到菜谱的用户流程、营养信息。

### PantryAI

[shri30a/PantryAI](https://github.com/shri30a/PantryAI)

- 图片识别食材。
- 维护库存。
- 使用检索增强生成推荐菜谱。

可借鉴：食材库存、RAG 菜谱推荐。

### Pantry to Plate

[MrAfoo/Pantry-to-Plate-Recipe-Generator](https://github.com/MrAfoo/Pantry-to-Plate-Recipe-Generator)

- 多图上传。
- 识别后可编辑食材。
- 分层菜谱、营养信息、替代食材和 PDF 导出。

可借鉴：确认食材后再生成、历史记录和导出体验。

## 餐厅菜单工程与经营分析

### Restaurant Menu Engineering

[restaurant-menu-engineering/restaurant-menu-engineering](https://github.com/restaurant-menu-engineering/restaurant-menu-engineering)

- 菜谱解析、食品成本和毛利管线。
- 强调可审计和确定性计算。

可借鉴：成本计算必须由确定性代码完成。

### PlateProfitAI

[PointZer0AI/plateprofitai](https://github.com/PointZer0AI/plateprofitai)

- 菜品成本、价格带、菜单矩阵和利润模拟。
- AI 顾问建立在结构化经营数据上。

可借鉴：菜单工程矩阵、价格模拟和决策解释。

### Menu Engineering Bot

[stephaniepatriciaans/menu-engineering-bot](https://github.com/stephaniepatriciaans/menu-engineering-bot)

- 导入 POS CSV。
- 做菜单表现、定价情景和 AI 决策备忘录。
- 财务计算由 Python 完成，模型负责解释。

可借鉴：POS 数据标准化、指标计算和可解释建议。

## Agent 与工具链

### Multi-Agent LLM Recipes and Prices

[mradovic38/multi-agent-llm-recipe-prices](https://github.com/mradovic38/multi-agent-llm-recipe-prices)

- 菜谱检索、SQL、向量库和食材价格。
- 支持预算、过敏和已有食材约束。

可借鉴：结构化数据和检索工具的分工。

### AI Restaurant Recommendation Engine

[malikmuqeet95-dev/ai-restaurant-recommendation-engine](https://github.com/malikmuqeet95-dev/ai-restaurant-recommendation-engine)

- 动态菜单上下文。
- Pydantic 严格结构。
- 预算和过敏原约束。

可借鉴：菜品推荐必须限制在当前菜单目录内。

### KitchenOwl Insights MCP

[lukegskw/kitchenowl-insights-mcp](https://github.com/lukegskw/kitchenowl-insights-mcp)

- 将库存估算和菜谱推荐封装成 MCP Server。
- 支持 stdio 和 HTTP 传输。

可借鉴：MCP 适合隔离外部系统，不替代确定性业务计算。

## 结论

1. 普通“上传食材生成菜谱”项目很多，不能成为核心差异化。
2. 餐厅经营项目有成本计算，但通常不处理多模态识材和生成式上架素材。
3. 当前项目的机会在“商家菜单研发 + 成本毛利 + 上架素材 + 人工确认”的完整闭环。
4. 近期最值得补的是结构化成本、商家画像、历史菜品中心和 CSV/审批。
5. RAG、POS 和 MCP 应该服务于业务数据，而不是先堆技术框架。
