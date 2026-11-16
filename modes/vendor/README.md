# LangMem 摘要组件

`langmem_summarization.py` 是 LangMem 的 `src/langmem/short_term/summarization.py` 原文件，保留 MIT 许可于 `LANGMEM_LICENSE`。

- 上游：https://github.com/langchain-ai/langmem
- 文件修订：`a463b1f86520230dcaff7822f8a73a3bacf350dd`
- 获取日期：2026-10-01
- 原文：https://github.com/langchain-ai/langmem/blob/a463b1f86520230dcaff7822f8a73a3bacf350dd/src/langmem/short_term/summarization.py

直接复用 `summarize_messages` 和 `RunningSummary`，接到项目现有本地模型 HTTP 客户端；没有引入 LangMem 的长期记忆服务、其他模型供应商 SDK 或 API 密钥。项目适配、原始聊天保留和阈值设置在 `modes/memory.py`。

## MCP Time 时钟适配

`modes/chat_tools.py::get_current_time` 改编自 [MCP servers/time 的 TimeServer.get_current_time](https://github.com/modelcontextprotocol/servers/blob/main/src/time/src/mcp_server_time/server.py)，获取日期为 2026-10-02。保留 `datetime.now(ZoneInfo(...))`、IANA 时区、ISO 时间、星期和夏令时结构；改为 LangChain `@tool` 函数及字典返回，增加会话默认时区和可读错误，不启用独立 MCP 服务。

上游当日处于 MIT 向 Apache 2.0 过渡期；已获同意的贡献适用 Apache 2.0，未获同意的原贡献仍为 MIT。完整上游声明、许可和版权信息保存为 `MCP_TIME_LICENSE`，不能笼统表述为全仓库 MIT。取用的原文件副本及当日许可证另保存在 `reports/interactive_modes/20261002T032754Z_open_chat_tools/`，项目修改为上述适配。

Open WebUI、LibreChat 仅用于产品交互和工具选择设计参考，本次未复制其代码。

## OATutor 渐进提示

`oatutor_hints.py::is_locked` 从 [CAHLR/OATutor SubHintSystem.isLocked](https://github.com/CAHLR/OATutor/blob/939eb0e30d8ae2389c8b096c8cd522713a79090e/src/components/problem-layout/SubHintSystem.js) 改编为 Python，保留“第一条提示可用；后续提示在依赖提示已看过后解锁”的规则。项目适配函数 `reveal_next` 将本题的两条提示组成线性序列，不重复生成同一提示。完整 MIT 许可和版权在 `OATUTOR_LICENSE`；获取日期2026-10-02，原文件副本保存在本批 `upstream/`。

同时参考其作答反馈、请求提示与界面翻译分离的设计；没有复制数学题库、套用 BKT 掌握度或宣称护理教学效果已经验证。评价与停止条件仍是本项目适配代码，由现有本地模型和会话状态共同实现。界面继续直接复用 Chainlit 的 Message/Action/UserSession API。

## 日常工具（2026-10-02）

`daily_tools.py`直接调用已安装的 [DDGS](https://github.com/deedy5/ddgs) 多引擎搜索、Trafilatura正文提取、PyMuPDF与既有Crawl4AI读取器；没有重写搜索引擎或浏览器。补齐当前CPU环境缺少的 `lxml_html_clean==0.4.5`，不修改模型/RAG环境。天气按[Open-Meteo官方API](https://open-meteo.com/en/docs)和[城市接口](https://open-meteo.com/en/docs/geocoding-api)调用免费非商业服务，保留来源；未复制其AGPL服务端代码。计算使用Python标准库AST/Decimal，通过现有LangChain `@tool`封装，不执行模型生成的程序。

产品设计参考[Open WebUI工具](https://docs.openwebui.com/features/extensibility/plugin/tools/)将网页搜索、URL读取和资料库拆成独立工具，以及[DeepSeek Harness的MCP接口](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/mcp/mcp-client/README.md)。两项目仅作接口/交互研究，本次未复制其源代码或部署另一套Agent。上游调研原文保存在本批报告的`upstream/`，不宣称这些项目的默认搜索无需账号或免费。
