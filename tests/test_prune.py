import torch

from dieselwolf.models import AMRClassifier, ConfigurableCNN
from scripts.prune import apply_global_pruning


def test_global_pruning_reduces_params():
    model = AMRClassifier(ConfigurableCNN(16, 4), num_classes=4)
    total = 0
    for p in model.parameters():
        total += p.numel()
    apply_global_pruning(model, amount=0.5)
    pruned = 0
    for p in model.parameters():
        pruned += torch.count_nonzero(p).item()
    assert pruned <= int(total * 0.6)
