import torch
from torch import nn
from EpisodeModule import EpisodeModule

class MemoryModule(nn.Module):

  def __init__(self, embedding_size, middle_number, num_iterations, device):

    super().__init__()

    self.embedding_size = embedding_size
    self.num_iterations = num_iterations
    self.middle_number = middle_number

    episode = EpisodeModule(embedding_size=self.embedding_size,
                                 num_iterations=self.num_iterations,
                                 middle_number = self.middle_number
                                 )
    self.episode = episode.to(device)

  def forward(self, m0, facts, question, h0=0):

    return self.episode(m0, facts, question, h0)
