import argparse
import time

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ONNX model latency")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--num-iters", type=int, default=50)
    p.add_argument("--num-warmup", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session = ort.InferenceSession(args.model)
    dummy = np.random.randn(1, 2, args.input_size).astype(np.float32)
    for _ in range(args.num_warmup):
        session.run(None, {"input": dummy})
    start = time.perf_counter()
    for _ in range(args.num_iters):
        session.run(None, {"input": dummy})
    end = time.perf_counter()
    latency_ms = (end - start) / args.num_iters * 1000.0
    print(f"Average latency: {latency_ms:.3f} ms")


if __name__ == "__main__":
    main()
