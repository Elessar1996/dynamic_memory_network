import spacy
import torch

def eng_tokenizer(text, eng):

    return [token.text for token in eng.tokenizer(text)]


def get_tokens(data_iter):
    for item in data_iter:
        yield eng_tokenizer(item)

def get_eos_indexes(input_string):

  '''
  this function takes in a batch of input stories, each story
  consisting of multiple sentences, and give for each sentence
  the index in which the eos token happens.

  '''
  batch_eos_idxes = []

  for story in input_string:

    eos_idxes = []

    for idx, token in enumerate(story):

      if token == 2:

        eos_idxes.append(idx)

    batch_eos_idxes.append(eos_idxes)

  return batch_eos_idxes

def get_representations(representation_batch, list_of_indexes):

  '''
  This function takes a representation tensor and also, a list of list of indexes
  as inputs, and returns the representations in places where an eos token happens.

  Args:

    representation_batch: This is the output of the GRU unit in the Input module.
    It's a torch tensor with the shape: (#batches, token, embedding)

    list_of_indexes: This actually a list containing list of indexes for each
    batch. The number of senteces of each story differs. So it cannot be converted
    to a tensor.

  Returns:

    a list containing a torch tensors of representation where the eos token happens.

  '''

  fact_batch = []

  for idx, idx_list in enumerate(list_of_indexes):

    facts = []

    for i in idx_list:

      facts.append(representation_batch[idx, i, :])

    fact_batch.append(facts)

  return fact_batch


def get_max_length(iterable):
  return max([len(i) for i in iterable])

def fill_the_fact_representation(facts, length, embedding_dim=50):
  '''
  Takes the a single batch fact and make all of its items' sizes the
  same.

  Args:
    fact: A list of lists of facts
    length: The size that the fact representation will end up
  Return:
    a torch tensor consisting of the batch facts
  '''

  new_facts = []

  for f in facts:

    mediation_f = []

    if len(f) < length:

      # new_f = f + (length - len(f))*torch.zeros(embedding_dim).tolist()

      for ff in f:
        mediation_f.append(ff.tolist())

      for _ in range(length - len(f)):

        mediation_f.append(torch.zeros(embedding_dim).tolist())

    elif len(f) == length:

      for ff in f:
        mediation_f.append(ff.tolist())

    new_facts.append(mediation_f)

  return torch.tensor(new_facts)


def get_and_fill_facts_1(embedded_story, story, length):

  story_eos_indexes = get_eos_indexes(story)
  facts = get_representations(embedded_story, story_eos_indexes)
  # max_length_of_facts = get_max_length(facts)
  filled_facts = fill_the_fact_representation(facts, length)

  return filled_facts

def get_and_fill_facts(embedded_story, story):

  story_eos_indexes = get_eos_indexes(story)
  facts = get_representations(embedded_story, story_eos_indexes)
  max_length_of_facts = get_max_length(facts)
  filled_facts = fill_the_fact_representation(facts, max_length_of_facts)

  return filled_facts

def equalize_shape(a, b, embedding_size=50):

  '''
  Takes two tensor, namely a and b, each one with the shape (n1, n2, n3),
  which n1 and n3 is the same for both, then get the one that has smaller n2, 
  and torch.zeros(embedding_size) tensors to it until its n2 becomes even with the 
  bigger one. 

  '''

  smaller_tensor = None
  bigger_tensor = None
  if a.shape[1] < b.shape[1]:
    smaller_tensor = a
    bigger_tensor = b
  else:
    smaller_tensor = b
    bigger_tensor = a

  equalization_size = bigger_tensor.shape[1]

  number_of_tensors_to_add = equalization_size - smaller_tensor.shape[1]
  smaller_tensor_tolist = smaller_tensor.tolist()
  
  for i in range(smaller_tensor.shape[0]):
    for j in range(number_of_tensors_to_add):

      filler = torch.zeros(embedding_size)
      filler = filler.tolist()
      smaller_tensor_tolist[i].append(filler)

  smaller_tensor = torch.tensor(smaller_tensor_tolist)

  return smaller_tensor
