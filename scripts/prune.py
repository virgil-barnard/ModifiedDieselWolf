import argparse
from typing import List, Tuple

import torch
from torch.nn.utils import prune

from dieselwolf.models import AMRClassifier, ConfigurableCNN


def collect_prunable_layers(
    model: torch.nn.Module,
) -> List[Tuple[torch.nn.Module, str]]:
    modules: List[Tuple[torch.nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            modules.append((m, "weight"))
    return modules


def apply_global_pruning(model: torch.nn.Module, amount: float) -> None:
    parameters = collect_prunable_layers(model)
    prune.global_unstructured(parameters, prune.L1Unstructured, amount=amount)
    for module, name in parameters:
        prune.remove(module, name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune a checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=11)
    p.add_argument("--amount", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = AMRClassifier(
        ConfigurableCNN(args.num_samples, args.num_classes), args.num_classes
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    apply_global_pruning(model, args.amount)
    torch.save({"state_dict": model.state_dict()}, args.output)


if __name__ == "__main__":
    main()
