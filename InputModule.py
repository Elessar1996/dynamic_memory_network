from torch import nn


class InputModule(nn.Module):

  def __init__(self, vocab_size, embedding_dim=50, padding_idx=0, num_gru_layers=1):

    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

    self.rnn = nn.GRU(
        input_size=embedding_dim,
        hidden_size=embedding_dim,
        num_layers=num_gru_layers,
        batch_first=True
    )


  def forward(self, x):

    return self.rnn(self.embedding(x))[0]