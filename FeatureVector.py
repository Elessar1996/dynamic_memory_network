import torch 

from torch import nn

class FeatureVector(nn.Module):
  
  def __init__(self, embedding_size, middle_number):

    super().__init__()

    self.weight = nn.Linear(in_features=embedding_size, out_features=embedding_size)
    self.size_of_scoring_network_layer = 7*embedding_size*middle_number + 2*embedding_size*embedding_size
    self.layer_1 = nn.Linear(in_features=self.size_of_scoring_network_layer, out_features=self.size_of_scoring_network_layer)
    self.layer_2 = nn.Linear(in_features=self.size_of_scoring_network_layer, out_features=1)
    self.fnn = nn.Sequential(
        self.layer_1, 
        nn.Tanh(),
        self.layer_2, 
        nn.Sigmoid()
    )

  def forward(self, c, m, q):

    weighted_c = self.weight(c)

    cwm = torch.matmul(torch.transpose(weighted_c, 1, 2), m)
    cwq = torch.matmul(torch.transpose(weighted_c, 1, 2), q)
    flatten = nn.Flatten(start_dim=1, end_dim=-1)
    z = torch.cat((flatten(c), flatten(m), flatten(q), flatten(torch.mul(c, q)), flatten(torch.mul(c, m)), flatten(torch.abs(torch.sub(c, q))), flatten(torch.abs(torch.sub(c, m))), flatten(cwq), flatten(cwm)), 1)
    return self.fnn(z)