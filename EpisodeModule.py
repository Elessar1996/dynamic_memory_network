import torch
from torch import nn
from FeatureVector import FeatureVector


class EpisodeModule(nn.Module):

  def __init__(self, embedding_size, middle_number, num_iterations):

    super().__init__()

    self.embedding_size = embedding_size
    self.middle_number = middle_number
    self.episode_gru = nn.GRU(
        input_size=embedding_size,
        hidden_size=embedding_size,
        num_layers=1,
        batch_first=True
    )
    self.memory_gru = nn.GRU(
        input_size=embedding_size,
        hidden_size=embedding_size,
        num_layers=1,
        batch_first=True
    )

    self.num_iterations= num_iterations

    self.g_score = FeatureVector(embedding_size, middle_number)


  def forward(self, m0, facts, question, h0=0):

    m = None
    h = None

    e = None

    for i in range(self.num_iterations):

      if i == 0:
        m = m0
        h = h0

      g = self.g_score(facts, m, question)

      if isinstance(h, int):
        
        episode_gru_output = self.episode_gru(facts)[1] 
        h_new = g*episode_gru_output + (1 - g)*torch.zeros_like(g)
      else:
        h_new = g*self.episode_gru(facts, h)[1] + (1 - g)*h
      h = h_new
      e = h_new
      m = self.memory_gru(m, e)[0]
      print('------------------------------------------------')

    return m
