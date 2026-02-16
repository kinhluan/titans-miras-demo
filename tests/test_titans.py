import torch
from titans_memory.titans import TitansMemoryLayer


def test_titans_layer_processes_sequence():
    layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16, momentum=0.9)
    seq = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 99.0, 1.0, 2.0])
    result = layer.process_sequence(seq.unsqueeze(-1))
    assert "outputs" in result
    assert "surprise_scores" in result
    assert "memory_snapshots" in result
    assert len(result["surprise_scores"]) == 9


def test_anomaly_has_high_surprise():
    torch.manual_seed(42)
    layer = TitansMemoryLayer(input_dim=1, memory_dim=16, hidden_dim=16)
    values = [1.0, 2.0, 3.0] * 10 + [99.0]
    seq = torch.tensor(values).unsqueeze(-1)
    result = layer.process_sequence(seq)
    scores = result["surprise_scores"]
    avg_normal = sum(s.item() for s in scores[:30]) / 30
    anomaly_surprise = scores[30].item()
    assert anomaly_surprise > avg_normal
