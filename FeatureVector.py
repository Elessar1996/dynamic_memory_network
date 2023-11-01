import torch 

from torch import nn

class FeatureVector(nn.Module):
  
  def __init__(self, embedding_size, middle_number):

    super().__init__()

    self.weight = nn.Linear(in_features=embedding_size, out_features=embedding_size)
    print('embedding size:', embedding_size, 'middle_number:', middle_number)
    self.size_of_scoring_network_layer = 7*embedding_size*middle_number + 2*embedding_size*embedding_size
    print('size_of_scoring_network_layer', self.size_of_scoring_network_layer)
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

    print(f'weighted_c: {weighted_c.shape}')

    print(f'm: {m.shape}')

    cwm = torch.matmul(torch.transpose(weighted_c, 1, 2), m)
    cwq = torch.matmul(torch.transpose(weighted_c, 1, 2), q)
    flatten = nn.Flatten(start_dim=1, end_dim=-1)
    print(f'flatten c: {flatten(c).shape}')
    print(f'flatten m: {flatten(m).shape}')
    print(f'flatten q: {flatten(q).shape}')
    print(f'cwm shape: {cwm.shape}')
    print(f'cwq: {cwq.shape}')
    print(f'|c - q|: {flatten(torch.abs(torch.sub(c, q))).shape}')
    print(f'|c - m|: {flatten(torch.abs(torch.sub(c, m))).shape}')
    print(f'cm: {flatten(torch.mul(c, m)).shape}')
    print(f'cq: {flatten(torch.mul(c, q)).shape}')
    
    

    z = torch.cat((flatten(c), flatten(m), flatten(q), flatten(torch.mul(c, q)), flatten(torch.mul(c, m)), flatten(torch.abs(torch.sub(c, q))), flatten(torch.abs(torch.sub(c, m))), flatten(cwq), flatten(cwm)), 1)
    # z = torch.cat((flatten(c), flatten(m), flatten(q), flatten(torch.mul(c, q)), flatten(torch.mul(c, m)), flatten(torch.abs(torch.sub(c, q))), flatten(torch.abs(torch.sub(c, m)))), 1)
    # z = torch.cat((flatten(c), flatten(m), flatten(q), flatten(torch.mul(c, q)), flatten(torch.mul(c, m)), flatten(cwq), flatten(cwm)), 1)
    print(f'z shape: {z.shape}')
    return self.fnn(z)