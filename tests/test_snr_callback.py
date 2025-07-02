import torch
from dieselwolf.callbacks import SNRCurriculumCallback
from dieselwolf.data.DigitalModulations import DigitalModulationDataset
from dieselwolf.data.TransformsRF import AWGN


class DummyTrainer:
    def __init__(self):
        self.callback_metrics = {}


def test_snr_curriculum():
    ds = DigitalModulationDataset(num_examples=1, num_samples=8, transform=AWGN(20))
    cb = SNRCurriculumCallback(ds, start_snr=20, step=5, patience=1)
    trainer = DummyTrainer()
    cb.on_train_start(trainer, None)
    assert ds.transform.SNRdB == 20
    trainer.callback_metrics["val_loss"] = torch.tensor(1.0)
    cb.on_validation_end(trainer, None)
    trainer.callback_metrics["val_loss"] = torch.tensor(1.0)
    cb.on_validation_end(trainer, None)
    assert ds.transform.SNRdB == 15
