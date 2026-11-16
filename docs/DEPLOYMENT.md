# Deployment

## Three services

| Service | Default address | Environment |
|---|---|---|
| Web and teaching state | `127.0.0.1:8021` | Python 3.10+, CPU |
| Local model | `127.0.0.1:8001/v1` | Separate vLLM/model environment |
| Reviewed knowledge base | `127.0.0.1:8020` | Separate retrieval environment; optional GPU |

The web launcher reads `.env` from the repository root and resolves project paths relative to the repository. It listens on loopback by default. Use SSH forwarding for personal remote access; shared public deployments need HTTPS, authentication and access limits. Built-in learner-profile isolation is not a complete multi-tenant authentication system.

```bash
ssh -N -L 19021:127.0.0.1:8021 your-user@your-server
```

Then open `http://127.0.0.1:19021/ui/` locally. Choose another available local port if necessary.

## Model

The default served model name is `ministral-3-14b`. Download weights separately from their publisher and observe the model license. Research deployments have used single-GPU NF4 quantization; actual VRAM needs depend on weights, quantization, context length and concurrency.

Install a compatible vLLM release in a separate model environment. Example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/your/model \
  --host 127.0.0.1 --port 8001 \
  --served-model-name ministral-3-14b --max-model-len 32768
```

This command does not enable quantization. Set quantized-loading arguments for your weights and vLLM version. The project uses `/v1/chat/completions`, `/v1/models`, `/tokenize`, streaming and JSON Schema constraints; a chat-compatible endpoint alone may not support every feature.

```bash
curl http://127.0.0.1:8001/v1/models
python scripts/19_check_model_contract.py --help
```

`LocalLLM` currently accepts loopback endpoints only. Forward a remote model endpoint to the local machine when needed.

The repository also retains `scripts/17_serve_local_model.py` for NF4 deployments. Before using it, set the model-environment Python path and weights directory under `local_model` in `config.yaml`, then install compatible vLLM and bitsandbytes dependencies in that environment. This launcher uses a single-request configuration; explicitly choose an available GPU and a new log path:

```bash
python scripts/17_serve_local_model.py --gpu 0 --log logs/model-nf4.log
```

## Knowledge service

Install `requirements-rag.txt` in a separate environment, prepare reviewed documents and indexes as described in [DATA.md](DATA.md), then run:

```bash
python scripts/15_serve_rag.py --config config.yaml --device cpu
```

`--device cuda:0` places retrieval models on the first GPU visible to that process. It does not terminate other processes. Avoid concurrent writers to the same local Qdrant directory in production.

## Web and external services

```bash
python scripts/start_web.py
curl http://127.0.0.1:8021/api/health
curl http://127.0.0.1:8021/api/readiness
```

`health` reports web-process availability. `readiness` reports model and knowledge-service status separately. Override configuration with `MDRPI_MODES_CONFIG`, `MDRPI_MODES_DATA`, `MDRPI_MODES_PORT` and `MDRPI_RAG_URL`. Keep test and production session databases separate.

Search and independent authoring review use external credentials from `.env`. For optional dynamic-page reading, install `requirements-web.txt` and configure a separate browser interpreter. Ordinary HTTP page reading does not require that environment.

The interface supports English and Chinese through the top-right language selector. Language preferences are stored in the browser; response language is set separately in Preferences & settings.

## Optional tool worker

`tools/` contains isolated file/computation tools and a Dockerfile. The main process treats tool tasks as restricted operations. Build the image and configure the runtime before enabling them. Basic Chat and Guided practice do not require Docker.

## Third-party licenses

Project code is MIT. PDF processing uses PyMuPDF (AGPL or commercial license); teaching content and other dependencies retain separate terms. See [third-party notices](../THIRD_PARTY_NOTICES.md) before distributing or deploying a service that includes these components.
