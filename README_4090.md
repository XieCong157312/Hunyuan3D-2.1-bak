# Hunyuan3D-2.1 Shape / Paint 串行显存版补丁

本补丁基于用户提供的 `api_server.py`、`api_models.py`、`model_worker.py` 三个文件修改。

## 改动

- Shape 和 Paint 改为按需加载，不在 Worker 初始化时同时加载。
- 一个任务内执行 Shape → 导出中间 GLB → 删除 Shape pipeline 引用 → `gc.collect()` / `torch.cuda.empty_cache()` / `torch.cuda.ipc_collect()` → 加载 Paint → 纹理生成 → 导出最终 GLB。
- Worker 使用一把 GPU 锁覆盖完整任务，避免多个请求在 Shape/Paint 阶段交错占用显存。
- 原有 `/generate`、`/send`、`/status/{uid}`、`/health` 保留。
- 新增 `/generate-textured`、`/send-textured`、`/result/{uid}`。
- `/generate-textured` 和 `/send-textured` 支持两种输入：
  - 原有 JSON：`{"image": "base64...", ...}`
  - 原始图片二进制请求体：`Content-Type: image/png` 或 `image/jpeg` 等。
- 原有 `/generate` 仍由 `texture` 字段决定是否运行 Paint；新组合接口会强制 `texture=true`。
- Paint 默认采用 `max_num_view=4`、`resolution=512`，可通过启动参数调整。

## 启动

在 Hunyuan3D-2.1 仓库根目录，用原有 Python 环境启动：

```bash
python api_server.py --host 0.0.0.0 --port 8081 --model_path tencent/Hunyuan3D-2.1 --low_vram_mode --paint-max-num-view 4 --paint-resolution 512
```

请将本补丁中的三个 `.py` 文件覆盖仓库根目录对应文件。其余模型目录、依赖、配置文件和权重保持原样。

## 同步生成：直接上传图片并下载 GLB

```bash
curl -X POST "http://127.0.0.1:8081/generate-textured" \
  -H "Content-Type: image/png" \
  --data-binary "@input.png" \
  -o result.glb
```

也兼容原有 JSON/base64 格式：

```bash
curl -X POST "http://127.0.0.1:8081/generate-textured" \
  -H "Content-Type: application/json" \
  -d '{"image":"BASE64_IMAGE_HERE","seed":1234}' \
  -o result.glb
```

## 异步生成

```bash
curl -X POST "http://127.0.0.1:8081/send-textured" \
  -H "Content-Type: image/png" \
  --data-binary "@input.png"
```

响应会返回 `uid`。随后轮询：

```bash
curl "http://127.0.0.1:8081/status/<uid>"
```

任务完成后通过以下地址下载：

```bash
curl "http://127.0.0.1:8081/result/<uid>" -o result.glb
```

## 说明与限制

- `torch.cuda.empty_cache()` 只能释放 PyTorch 缓存中已经没有引用的显存；因此代码还会先清除 Worker 属性和局部 pipeline 引用，再做垃圾回收和 CUDA 清理。
- Shape/Paint 串行化可以避免两个模型权重同时常驻，但不代表 Paint 单阶段峰值一定低于 24GB。若 Paint 单独仍 OOM，可继续降低 `--paint-max-num-view`（例如 3）或 `--paint-resolution`（例如 384），画质和覆盖范围可能下降。
- 代码已完成 Python 语法编译检查；当前未在 RTX 4090 上执行真实模型推理，因此不能声称已完成硬件实测。
