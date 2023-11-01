import torch

from torch import nn
class AnswerModule(nn.Module):

  '''
  This Answer Module is specifically designed for bAbI dataset in which answers
  has only one word.
  '''

  def __init__(self, embedding_size, middle_size, vocab_size):

    super().__init__()
    print(f'vocab size: {vocab_size}')
    self.embedding_size = embedding_size 
    self.middle_size = middle_size 
    self.gru = nn.GRU(input_size=embedding_size, hidden_size=embedding_size, 
                      num_layers=1, batch_first=True)
    self.softmax = nn.Softmax()
    self.linear_layer_1 = nn.Linear(in_features=embedding_size, out_features=1)
    self.linear_layer_2 = nn.Linear(in_features=middle_size+1, out_features=1)

  def forward(self, m0, q):
    
    a0 = m0[:, -1, :]

    # y0 = self.softmax(self.linear_layer(a0))
    
    print('**********************')
    # print(f'y0 shape: {y0.shape}')
    print(f'q shape: {q.shape}')
    print(f'a0 shape: {a0.unsqueeze(dim=1).shape}')
    print('**********************')
    
    concatenated_tensor = torch.cat((q, a0.unsqueeze(dim=1)), dim=1)
    print(f'concatenated_tensor shape: {concatenated_tensor.shape}')
    a1 = self.gru(concatenated_tensor, a0.unsqueeze(dim=0))[0]
    print(f'a1 shape: {a1.shape}')

    intermediate_a1 = self.linear_layer_1(a1)
    print(f'intermediate_a1: {intermediate_a1.shape}')
    upper_intermediate_a1 = self.linear_layer_2(torch.transpose(intermediate_a1, 1, 2))
    print(f'upper_intermediate_a1: {upper_intermediate_a1.shape}')
    advanced_a1 = upper_intermediate_a1.squeeze(dim=-1)
    y1 = self.softmax(advanced_a1)
    
    return y1 