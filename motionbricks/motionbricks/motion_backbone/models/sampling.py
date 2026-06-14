import torch as t


def log(x, eps=1e-20):
    return t.log(x.clamp(min=eps))


def gumbel_noise(x):
    noise = t.zeros_like(x).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(x, temperature=1., dim=-1):
    return ((x / max(temperature, 1e-10)) + gumbel_noise(x)).argmax(dim=dim)
