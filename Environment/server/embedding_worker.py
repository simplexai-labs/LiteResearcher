#!/usr/bin/env python3
"""
🤖 Embedding Worker - Redis 队列消费者

架构：
    检索后端 (local_rag_server.py)  ──rpush──▶  Redis 队列 "embed_queue"
                                                      │
                                                      ▼  blpop
                                           本 worker 取出文本，跑 BGE-M3
                                                      │
                                                      ▼  rpush "res:{id}"
    检索后端  ◀──blpop──  结果（dense + sparse，msgpack 编码）

可启动多个 worker 进程（每张 GPU 一个）并行消费同一个队列。

环境变量：
    REDIS_HOST   (默认 127.0.0.1)
    REDIS_PORT   (默认 6379)
    INPUT_QUEUE  (默认 embed_queue)
    WORKER_GPU   (默认取 diskann_config.API_DEVICE 轮询；可显式指定如 cuda:0)
"""

import os
import time
import msgpack
import numpy as np
import redis
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from diskann_config import API_BGE_MODEL_PATH, API_BGE_MAX_LENGTH, MAX_BATCH_SIZE, API_DEVICE

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
INPUT_QUEUE = os.environ.get("INPUT_QUEUE", "embed_queue")


def pick_gpu():
    """选择本 worker 使用的 GPU。优先 WORKER_GPU 环境变量，否则取 API_DEVICE 第一个。"""
    env = os.environ.get("WORKER_GPU")
    if env:
        return env
    if isinstance(API_DEVICE, list) and API_DEVICE:
        return API_DEVICE[0]
    return API_DEVICE if isinstance(API_DEVICE, str) else "cuda:0"


def serialize_sparse_row(sparse_matrix, row: int = 0):
    """
    将单行稀疏向量序列化为 backend 期望的字典格式。
    backend 用 coo_array((values,(indices,cols)), shape) 还原，故：
        indices = 行号(全 0)，cols = token id，values = 权重，shape = (1, vocab)
    """
    if sparse_matrix is None:
        return None
    try:
        coo = sparse_matrix.tocoo()
        return {
            "indices": coo.row.tolist(),   # 单行恒为 0
            "values": coo.data.tolist(),
            "cols": coo.col.tolist(),
            "shape": list(coo.shape),
        }
    except Exception as e:
        print(f"❌ Sparse 序列化失败: {e}", flush=True)
        return None


def main():
    gpu = pick_gpu()
    print("=" * 70, flush=True)
    print("🤖 LiteResearcher Embedding Worker (Redis 消费者)", flush=True)
    print("=" * 70, flush=True)
    print(f"📡 Redis: {REDIS_HOST}:{REDIS_PORT}  队列: {INPUT_QUEUE}", flush=True)
    print(f"🎮 GPU: {gpu}", flush=True)
    print(f"🤖 模型: {API_BGE_MODEL_PATH}", flush=True)
    print(f"🔢 精度: FP32 (DISKANN 要求)", flush=True)

    embedding_function = BGEM3EmbeddingFunction(
        model_name=API_BGE_MODEL_PATH,
        batch_size=MAX_BATCH_SIZE,
        device=gpu,
        use_fp16=False,            # DISKANN 使用 FP32
        max_length=API_BGE_MAX_LENGTH,
        normalize_embeddings=True,
    )
    # 预热
    embedding_function(["warmup"])
    print("✅ 模型加载完成，开始消费队列...", flush=True)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    r.ping()

    while True:
        try:
            # 阻塞等待一条请求
            item = r.blpop(INPUT_QUEUE, timeout=5)
            if item is None:
                continue
            _, raw = item
            req = msgpack.unpackb(raw, raw=False)
            req_id = req["id"]
            text = req["text"]

            t0 = time.time()
            result = embedding_function([text])
            infer_time = time.time() - t0

            dense_vec = np.asarray(result["dense"][0], dtype=np.float32)

            payload = {
                "dense": dense_vec.tobytes(),
                "dense_shape": list(dense_vec.shape),
                "infer_time": infer_time,
            }

            sparse = result.get("sparse")
            if sparse is not None:
                try:
                    sparse_row = sparse[[0]]
                except Exception:
                    sparse_row = sparse
                sparse_dict = serialize_sparse_row(sparse_row)
                if sparse_dict is not None:
                    payload["sparse"] = msgpack.packb(sparse_dict)

            r.rpush(f"res:{req_id}", msgpack.packb(payload))

        except KeyboardInterrupt:
            print("\n🛑 收到中断，退出 worker", flush=True)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 出错时也尽量回写错误，避免 backend 一直阻塞到超时
            try:
                if 'req_id' in locals():
                    r.rpush(f"res:{req_id}", msgpack.packb({"error": str(e)}))
            except Exception:
                pass


if __name__ == "__main__":
    main()
