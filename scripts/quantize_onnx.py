import argparse

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    quantize_dynamic(args.input, args.output, weight_type=QuantType.QInt8)


if __name__ == "__main__":
    main()
