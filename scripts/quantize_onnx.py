import argparse
import importlib.util


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--op-types",
        type=str,
        default="MatMul,Gemm",
        help="Comma-separated operator types to quantize",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("onnx") is None:
        print("onnx package not installed, skipping quantization.")
        return
    from onnxruntime.quantization import QuantType, quantize_dynamic

    op_types = [op.strip() for op in args.op_types.split(",") if op.strip()]
    quantize_dynamic(
        args.input,
        args.output,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
    )


if __name__ == "__main__":
    main()
