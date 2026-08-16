import torch
from torch import nn


class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy = temperatures == 0
        safe_temperatures = torch.where(greedy, 1.0, temperatures)
        scaled_logits = logits.float().div_(safe_temperatures.unsqueeze(1))
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled = probs.div_(
            torch.empty_like(probs).exponential_().clamp_min_(1e-10)
        ).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy, greedy_tokens, sampled)
