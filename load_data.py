from torch.utils.data import Dataset, DataLoader
from vocab import Vocab
import pandas as pd
import datasets
from utils import collate_fn

class MyDataset(Dataset):
    def __init__(self, data, vocab_en: Vocab, vocab_de: Vocab):
        super(MyDataset, self).__init__()
        '''
            data: taken from bentrevett/multi30k, divided into 3 sets train, valid, test, data will be one of those 3 sets
            data format: [
                {
                    'en': ... ,
                    'de': ...
                }, ...
            ]
        '''
        # Tạo list riêng cho 'en' và 'de'
        self.en_list = [d['en'] for d in data]
        self.de_list = [d['de'] for d in data]

        self.vocab_en = vocab_en
        self.vocab_de = vocab_de

    def __len__(self):
        return len(self.en_list)
    
    def __getitem__(self, index):
        en = self.en_list[index]
        de = self.de_list[index]

        en_ids = self.vocab_en.convert_tokens_to_ids(en)
        de_ids = self.vocab_de.convert_tokens_to_ids(de)

        return {
            'en_sentence': en,
            'de_sentence': de,
            'en_ids': en_ids,
            'de_ids': de_ids
        }
    

class Load_Data:
    def __init__(self, data_path, max_length, batch_size):
        self.dataset = datasets.load_dataset(data_path)
        self.batch_size = batch_size

        list_train = list(self.dataset["train"])
        list_valid = list(self.dataset["validation"])
        list_test = list(self.dataset["test"])

        # Build Vocab En and De
        en_list = [d['en'] for d in list_train]
        de_list = [d['de'] for d in list_train]

        self.vocab_en = Vocab(en_list, max_length= max_length)
        self.vocab_de = Vocab(de_list, max_length= max_length)

        self.train_dataset = MyDataset(data= list_train, vocab_en= self.vocab_en, vocab_de= self.vocab_de)
        self.valid_dataset = MyDataset(data= list_valid, vocab_en= self.vocab_en, vocab_de= self.vocab_de)
        self.test_dataset = MyDataset(data= list_test, vocab_en= self.vocab_en, vocab_de= self.vocab_de)

    def load_train_valid(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size= self.batch_size,
                                      shuffle= True,
                                      collate_fn= collate_fn)
        
        valid_dataloader = DataLoader(self.valid_dataset,
                                      batch_size= self.batch_size,
                                      shuffle= False,
                                      collate_fn= collate_fn)
        
        return train_dataloader, valid_dataloader
    
    def load_test(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size= self.batch_size,
                                     shuffle= False,
                                     collate_fn= collate_fn)
        
        return test_dataloader