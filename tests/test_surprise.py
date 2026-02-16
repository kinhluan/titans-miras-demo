import torch
from titans_memory.surprise import SurpriseMetric


def test_surprise_metric_output_shape():
    metric = SurpriseMetric(input_dim=1, hidden_dim=16)
    x = torch.randn(1)
    surprise_score = metric(x)
    assert surprise_score.shape == ()


def test_high_surprise_for_novel_input():
    torch.manual_seed(42)
    metric = SurpriseMetric(input_dim=1, hidden_dim=16)
    familiar = torch.tensor([1.0])
    for _ in range(50):
        metric(familiar)
        metric.update_predictor(familiar, lr=0.01)
    surprise_familiar = metric(familiar)
    novel = torch.tensor([99.0])
    surprise_novel = metric(novel)
    assert surprise_novel.item() > surprise_familiar.item()


def test_momentum_smooths_surprise():
    metric = SurpriseMetric(input_dim=1, hidden_dim=16, momentum=0.9)
    scores = []
    for val in [1.0, 1.0, 1.0, 99.0, 1.0, 1.0]:
        x = torch.tensor([val])
        score = metric.compute_with_momentum(x)
        scores.append(score.item())
    assert scores[4] > scores[0]
