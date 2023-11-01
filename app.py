from VectorizedbAbI import VectorizedbAbI
import torch 
from torch import nn
from InputModule import InputModule
from InputQuestionModule import InputQuestionModule
from utils import * 
from MemoryModule import MemoryModule
from AnswerModule import AnswerModule

data_path = 'data'

task_ids = [3]


data_pipe = VectorizedbAbI(
    task_ids=task_ids
)

train_datapipe = data_pipe.train_data_pipe
test_datapipe = data_pipe.test_data_pipe


inp_module = InputQuestionModule(vocab_size=len(data_pipe.train_vocab.get_itos()),
                                            embedding_dim=50,
                                            padding_idx=0)
midd_numbers = []
for s, q, a in train_datapipe:
    embedded_story, embedded_question = inp_module(s, q)

    facts = get_and_fill_facts(embedded_story, s)

    midd_numbers.append(facts.shape[1])

for s, q, a in test_datapipe:
    embedded_story, embedded_question = inp_module(s, q)

    facts = get_and_fill_facts(embedded_story, s)

    midd_numbers.append(facts.shape[1])

max_midd_num = max(midd_numbers)



MIDDLE_NUMBER = max_midd_num
EMBEDDING_SIZE = 50
NUM_ITERATIONS = 3

memory_module = MemoryModule(embedding_size=EMBEDDING_SIZE,
                             middle_number=max_midd_num,
                             num_iterations=NUM_ITERATIONS)

answer_module = AnswerModule(
    embedding_size=EMBEDDING_SIZE,
    middle_size=max_midd_num,
    vocab_size=len(data_pipe.train_vocab.get_itos())
)

##TODO: set a limit for the length of each story

for story, question, answer in train_datapipe:

    embedded_story, embedded_question = inp_module(story, question)
    
    facts = get_and_fill_facts_1(embedded_story, story, length=MIDDLE_NUMBER)
    print(f'fact shape{facts.shape}')
    # print(f'fact first batch: {facts[5, :, :]}')
    print(f'question shape: {embedded_question.shape}')
    # equalized_question = equalize_shape(embedded_question, facts)
    equalized_question = fill_the_fact_representation(facts, MIDDLE_NUMBER)
    print(f'equalized_question: {equalized_question.shape}')

    m = memory_module(m0=equalized_question, facts=facts, question=equalized_question, h0=0)
    print(f'm: {m.shape}')
    print(f'the end')

    y = answer_module(m, equalized_question)

    print(f'y shape: {y.shape}')

    break

    


    
