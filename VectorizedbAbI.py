import torch
from ParsedbAbIDateset import BagOfTheWordsAggregate
from torchdata.datapipes.iter import IterableWrapper
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from utils import eng_tokenizer, get_tokens
import spacy

class VectorizedbAbI:

    def __init__(self, data_directory='./data/en', task_ids=[1], batch_size=32):

        self.data_directory = data_directory
        self.task_ids = task_ids
        self.bAbI = BagOfTheWordsAggregate(
            data_directory=data_directory,
            task_id=task_ids
        )
        self.batch_size = batch_size
        self.eng = spacy.load('en_core_web_sm')
        self.parsed_train_data = self.bAbI.parsed_train_data
        self.parsed_test_data = self.bAbI.parsed_test_data

        self.train_data_pipe = IterableWrapper(self.parsed_train_data)
        self.test_data_pipe = IterableWrapper(self.parsed_test_data)

        self.all_train_words = self.aggregate_all_words(self.train_data_pipe)
        self.all_test_words = self.aggregate_all_words(self.test_data_pipe)

        self.train_vocab = build_vocab_from_iterator(
            self.get_tokens(self.all_train_words),
            min_freq=2,
            specials=['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )

        self.train_vocab.set_default_index(self.train_vocab['<unk>'])

        self.test_vocab = build_vocab_from_iterator(
            self.get_tokens(self.all_test_words),
            min_freq=2,
            specials=['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        self.test_vocab.set_default_index(self.test_vocab['<unk>'])

        self.train_data_pipe = self.train_data_pipe.map(lambda data_instance: self.apply_transform(data_instance, self.train_vocab))
        self.test_data_pipe = self.test_data_pipe.map(lambda data_instance: self.apply_transform(data_instance, self.test_vocab))

        self.train_data_pipe = self.train_data_pipe.bucketbatch(
            batch_size=self.batch_size,
            batch_num=len(self.train_data_pipe),
            bucket_num=1,
            use_in_batch_shuffle=True
        )

        self.test_data_pipe = self.test_data_pipe.bucketbatch(
            batch_size=self.batch_size,
            batch_num=len(self.test_data_pipe),
            bucket_num=1,
            use_in_batch_shuffle=False
        )

        self.train_data_pipe = self.train_data_pipe.map(self.separate_stories_queries_answers)
        self.test_data_pipe = self.test_data_pipe.map(self.separate_stories_queries_answers)


        self.train_data_pipe = self.train_data_pipe.map(self.apply_padding)
        self.test_data_pipe = self.test_data_pipe.map(self.apply_padding)





    def aggregate_all_words(self, data_pipe):

        all_story_sentence = [' '.join(s) for story, _, _ in data_pipe for s in story]
        all_query_sentence = [' '.join(query) for _, query, _ in data_pipe]
        aggregated_answers = [w for _, _, answer in data_pipe for w in answer]
        all_answer_words = ' '.join(aggregated_answers)

        return all_story_sentence + all_query_sentence + [all_answer_words]

    def eng_tokenizer(self, text):

        return [token.text for token in self.eng.tokenizer(text)]

    def get_tokens(self, data_iter):

      for item in data_iter:
        yield self.eng_tokenizer(item)

    def get_transform(self, vocab):
        text_transform = T.Sequential(
            T.VocabTransform(vocab=vocab),
            T.AddToken(1, begin=True),
            T.AddToken(2, begin=False)
        )

        return text_transform

    def apply_transform(self, data_instance, vocab):

      story, query, answer = data_instance
      story_indexes = []

      for s in story:

        story_indexes += self.get_transform(vocab)(self.eng_tokenizer(' '.join(s)))

      query_indexes = self.get_transform(vocab)(self.eng_tokenizer(' '.join(query)))

      answer_index = self.get_transform(vocab)(answer)

      answer_index_reformed = [i for i in answer_index if (i != 1 and i != 2)]


      return story_indexes, query_indexes, answer_index_reformed



    def separate_stories_queries_answers(self, data_item):

        stories, queries, answers = zip(*data_item)

        return stories, queries, answers

    def apply_padding(self, sequence_batch):

        stories, queries, answers = sequence_batch

        padded_stories = T.ToTensor(0)(list(stories))

        padded_queries = T.ToTensor(0)(list(queries))

        padded_answers = T.ToTensor(0)(list(answers))


        return padded_stories, padded_queries, padded_answers


