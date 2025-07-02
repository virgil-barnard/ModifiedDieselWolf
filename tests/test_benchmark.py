import subprocess


def test_benchmark_script(tmp_path):
    out_dir = tmp_path / "bench"
    subprocess.check_call(
        [
            "python",
            "scripts/benchmark.py",
            "--num-examples",
            "2",
            "--num-samples",
            "16",
            "--batch-size",
            "1",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert (out_dir / "accuracy_vs_snr.png").exists()
    assert (out_dir / "confusion_matrix.png").exists()
    assert (out_dir / "latency_ms.txt").exists()
