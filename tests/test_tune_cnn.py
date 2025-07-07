import subprocess


def test_tune_cnn_script(tmp_path):
    subprocess.check_call(
        [
            "python",
            "scripts/tune_cnn.py",
            "--epochs",
            "1",
            "--num-examples",
            "4",
            "--num-samples",
            "16",
            "--max-trials",
            "1",
            "--adv-eps",
            "0.0",
            "0.1",
            "--adv-weight",
            "0.5",
            "0.8",
            "--adv-norm",
            "inf",
            "2",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert any(tmp_path.iterdir())
