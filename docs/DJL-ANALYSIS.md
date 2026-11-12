# What is in `djl-serving-master.zip`, and what MiniDL took from it

DJL Serving is the model server of the Deep Java Library (DJL). Only the **Java** parts were used as a model for MiniDL; the
Python files were read to understand the Python side (see the end).

## What the zip contains (counted from the files)
| Module | Java files | Python files | What it does |
|---|---:|---:|---|
| `serving` | 95 | 12 | The server: Netty HTTP handlers (`InferenceRequestHandler`, `ManagementRequestHandler`), model store, config, Prometheus export |
| `wlm` | 34 | 0 | The **WorkLoad Manager**: `WorkLoadManager`, `WorkerPool`, `WorkerThread`, `BatchAggregator`, `Job`, `ModelInfo` |
| `engines` | 21 | 92 | The Python engine: Java starts a Python process and talks to it (`PyEnv`), Python side in `djl_python/` |
| `plugins` | 35 | 0 | Optional extras (management console, cache, KServe, static files) |
| `prometheus` | 5 | 0 | Metrics in Prometheus text format |
| `benchmark`, `awscurl`, `tools` | 29 | 0 | Load testing and a curl-like client |
| `tests`, `buildSrc` | 2 | 35 | Integration tests; Gradle build logic (Kotlin, not used) |

Total: 222 Java files (about 40,000 lines), 139 Python files, 1 Kotlin file. Built with Gradle; depends on DJL core, Netty, Gson, SLF4J.

## The Java ideas, and where they are in MiniDL
| DJL Serving (Java) | Idea | MiniDL |
|---|---|---|
| `WorkLoadManager`, `WorkerPool` | One pool of worker threads per model, all taking jobs from one queue | `ModelPool` |
| `WorkerThread` | A worker owns its own predictor (models are not thread-safe) | each `ModelPool` worker loads its OWN copy of the model |
| `BatchAggregator.drainTo`, `batch_size`, `max_batch_delay` | **Dynamic batching**: collect requests until the batch is full or a short delay passed, run ONE forward pass | `ModelPool.Worker.nextBatch` (`batchSize`, `maxBatchDelayMs`) |
| `queue_size`, `WlmCapacityException` | Bounded queue: refuse at once when full (HTTP 503) instead of piling up | `ModelPool.CapacityException` -> 503 |
| `min_workers` / `max_workers`, `scaleUp` | Add workers when the queue grows; scale by API | `ModelPool.scale`, auto scale-up when the queue is over 2 batches |
| `LoadModelRequest`, `ModelStore` | Register models by name from a model store folder | `POST /models?url=file.bin&name=...`; `url` is only a file name inside the store |
| `InferenceRequestHandler`: `/ping`, `/predictions/{model}`, `/models` | The REST API | `ModelServer` (same paths and parameter names) |
| `PrometheusExporter`, `/metrics` | Counters, gauges, latency histogram | `ModelServer.metrics()` |
| `BadRequestException`, `ServiceUnavailableException` | Map errors to 400 / 503 / 404 | `HttpError` (400, 404, 405, 409, 413, 503, 504) |

Deliberately NOT copied: Netty (MiniDL has no dependencies; the JDK's `HttpServer` is enough), workflows, adapters, cluster mode,
SageMaker compatibility, TLS, plugins, GPU worker placement.

## The Python language in the zip
- **Python 3** (the `djl_python` package, 139 `.py` files; the Python engine in `engines/python`).
- It runs deep-learning models with **PyTorch** (`import torch`) and Hugging Face **transformers** (`huggingface.py`, `streaming_utils.py`);
  other handlers target vLLM and TensorRT-LLM (large language model serving).
- The Java server and the Python handler are two processes: Java (`PyEnv` uses `ProcessBuilder`) starts Python and they exchange
  requests over a socket. This is how DJL Serving runs PyTorch models without a native Java PyTorch binding of its own.

## What MiniDL did about PyTorch
Nothing in this project is Python. MiniDL cannot run PyTorch inside the JVM, and DJL's way (a second Python process) was not copied. Instead
`SafeTensors.java` reads and writes the safetensors file format, so weights can move between PyTorch and MiniDL as plain numbers
(Dense, Conv2D, BatchNorm2D mapped to PyTorch names and layouts). Running arbitrary PyTorch models from Java is what DJL does and is not attempted.
