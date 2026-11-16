# 部署

## 三个服务

| 服务 | 默认地址 | 环境 |
|---|---|---|
| 网页与教学状态管理 | `127.0.0.1:8021` | Python 3.10+，CPU |
| 本地模型 | `127.0.0.1:8001/v1` | 独立 vLLM / 模型环境 |
| 审核知识库 | `127.0.0.1:8020` | 独立检索环境，可选 GPU |

网页启动脚本会读取根目录 `.env`，路径均相对仓库解析。默认回环监听，远程个人访问可通过 SSH 转发；多人公网部署应在前面增加 HTTPS、登录认证和访问限制。内置学习档案隔离不是完整的多人租户认证系统。

```bash
ssh -N -L 19021:127.0.0.1:8021 your-user@your-server
```

然后访问本机 `http://127.0.0.1:19021/ui/`。本机端口不可用时选择其他可监听端口。

## 模型

项目默认模型服务名是 `ministral-3-14b`。权重不随仓库分发，应从发布方下载并遵守模型许可。原研究部署使用过 NF4 单卡量化；具体显存取决于模型、量化方式、上下文长度和并发。

在独立模型环境中准备与权重兼容的 vLLM，启动方式示例：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/your/model \
  --host 127.0.0.1 --port 8001 \
  --served-model-name ministral-3-14b --max-model-len 32768
```

这条命令没有自动开启量化。量化加载参数应按所选权重和 vLLM 版本设置。项目需要 `/v1/chat/completions`、`/v1/models`、`/tokenize`，并使用流式输出和 JSON Schema 约束；仅“兼容聊天接口”不一定满足全部功能。

```bash
curl http://127.0.0.1:8001/v1/models
python scripts/19_check_model_contract.py --help
```

`LocalLLM` 当前只连接回环地址。如模型位于另一台机器，可通过 SSH 转发到本机。

仓库也保留了 NF4 专用启动脚本 `scripts/17_serve_local_model.py`。使用它之前，在 `config.yaml` 的 `local_model` 中填写独立模型环境的 Python 路径和权重目录，并在该环境安装兼容的 vLLM、bitsandbytes 等依赖。该脚本采用单请求配置，需要显式指定空闲 GPU 和新的日志路径：

```bash
python scripts/17_serve_local_model.py --gpu 0 --log logs/model-nf4.log
```

## 知识库

在独立环境安装 `requirements-rag.txt`，按照 [DATA.md](DATA.md) 准备审核语料与索引，然后启动：

```bash
python scripts/15_serve_rag.py --config config.yaml --device cpu
```

`--device cuda:0` 可将检索模型放在该进程可见的第一张 GPU；不会自动结束其他进程。不要在生产环境中同时用多个进程写入同一个本地 Qdrant 目录。

## 网页与外部服务

```bash
python scripts/start_web.py
curl http://127.0.0.1:8021/api/health
curl http://127.0.0.1:8021/api/readiness
```

`health` 表示网页进程存活；`readiness` 分别报告模型和知识库状态。服务配置可以通过 `MDRPI_MODES_CONFIG`、`MDRPI_MODES_DATA`、`MDRPI_MODES_PORT`、`MDRPI_RAG_URL` 覆盖。不要让测试与正式网页共用会话数据库。

搜索与独立命题审核使用 `.env` 中的外部凭据。动态页面读取可选安装 `requirements-web.txt` 并配置独立浏览器解释器；普通 HTTP 正文读取不依赖该环境。

## 可选工具工作进程

`tools/` 提供隔离文件与计算工具代码及 Dockerfile。主进程将工具任务作为受限操作处理；启用前需自行构建镜像并配置运行环境。基本问答和教学不要求 Docker。

## 第三方许可

项目代码为 MIT。PDF 处理依赖 PyMuPDF（AGPL 或商业许可），教学内容以及其他依赖也有独立许可，详见第三方声明。分发或部署包含这些组件的完整服务时，需要遵守相应组件的条件。
