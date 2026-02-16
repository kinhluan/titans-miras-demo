import torch
from titans_memory.data import generate_repeating_with_anomalies, generate_frequency_shift


def test_repeating_with_anomalies_shape():
    seq, anomaly_mask = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0], repeats=10, anomaly_indices=[7, 15], anomaly_value=99.0
    )
    assert seq.shape == (30,)
    assert anomaly_mask.shape == (30,)


def test_repeating_with_anomalies_values():
    seq, anomaly_mask = generate_repeating_with_anomalies(
        pattern=[1.0, 2.0, 3.0], repeats=4, anomaly_indices=[5], anomaly_value=99.0
    )
    assert seq[5].item() == 99.0
    assert anomaly_mask[5].item() == 1.0
    assert seq[0].item() == 1.0
    assert seq[1].item() == 2.0
    assert anomaly_mask[0].item() == 0.0


def test_frequency_shift_shape():
    seq, shift_points = generate_frequency_shift(
        length=200, base_freq=0.1, shifted_freq=0.5, shift_at=[100]
    )
    assert seq.shape == (200,)
    assert shift_points == [100]
