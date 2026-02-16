import torch
from titans_memory.memory import MemoryModule


def test_memory_module_output_shape():
    mem = MemoryModule(input_dim=4, memory_dim=16)
    x = torch.randn(4)
    output = mem.read(x)
    assert output.shape == (4,)


def test_memory_write_changes_state():
    mem = MemoryModule(input_dim=4, memory_dim=16)
    x = torch.randn(4)
    output_before = mem.read(x).clone()
    mem.write(x, surprise_score=torch.tensor(1.0))
    output_after = mem.read(x)
    assert not torch.allclose(output_before, output_after)


def test_low_surprise_minimal_update():
    mem = MemoryModule(input_dim=4, memory_dim=16)
    x = torch.randn(4)
    output_before = mem.read(x).clone()
    mem.write(x, surprise_score=torch.tensor(0.0))
    output_after = mem.read(x)
    assert torch.allclose(output_before, output_after)


def test_forgetting_decays_weights():
    mem = MemoryModule(input_dim=4, memory_dim=16, forget_rate=0.1)
    x = torch.randn(4)
    mem.write(x, surprise_score=torch.tensor(1.0))
    output_right_after = mem.read(x).clone()
    for _ in range(50):
        mem.apply_forgetting()
    output_after_forget = mem.read(x)
    assert output_after_forget.norm().item() < output_right_after.norm().item()
