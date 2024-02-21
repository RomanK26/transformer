import torch
from torch.utils.data import Dataset,DataLoader
from indicnlp.tokenize import indic_tokenize
import spacy
from conf import *


def load_tokenizers():
    return indic_tokenize, spacy.load('en_core_web_sm')

def tokenize_ne(text: str, tokenizer):
        return [tok for tok in tokenizer.trivial_tokenize(text)]



def tokenize_en(text:str,tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]





class CustomDataset(Dataset):
    def __init__(self, source: str, target: str):
        self.nepali_root = source
        self.english_root = target
        self.current=0
        self.max_src_len=0
        self.max_trg_len=0
        self.tokenizers =load_tokenizers()
        self.src_vocab=set()
        self.trg_vocab=set()
        self.src_vocab.update(['<sos>', '<eos>', '<pad>','<unk>']) 
        self.trg_vocab.update(['<sos>', '<eos>', '<pad>','<unk>'])
        self.total_sentences = 0
        self.data = []

        
        with open(self.nepali_root, 'r') as nepali, open(self.english_root, 'r') as english:
            for nep, eng in zip(nepali, english):
                tokenized_nepali = tokenize_ne(nep, self.tokenizers[0])
                tokenized_eng = tokenize_en(eng, self.tokenizers[1])
                self.total_sentences += 1
                self.max_src_len = max(self.max_src_len, len(tokenized_nepali))
                self.max_trg_len = max(self.max_trg_len, len(tokenized_eng))
                self.src_vocab.update(tokenized_nepali)
                self.trg_vocab.update(tokenized_eng)
                self.data.append((tokenized_nepali, tokenized_eng))
#                 random.shuffle(self.data)

        
        self.src_vocab_dict={word: i for i, word in enumerate(self.src_vocab)}
        self.trg_vocab_dict={word: i for i, word in enumerate(self.trg_vocab)}
        

        self.trg_pad_idx = self.trg_vocab_dict['<pad>']
        self.trg_sos_idx = self.trg_vocab_dict['<sos>']
        self.trg_eos_idx = self.trg_vocab_dict['<eos>']
        self.src_pad_idx = self.src_vocab_dict['<pad>']
        self.src_sos_idx = self.src_vocab_dict['<sos>']   
        self.src_eos_idx = self.src_vocab_dict['<eos>']
        self.src_unk_idx = self.src_vocab_dict['<unk>'] 
        self.trg_unk_idx = self.trg_vocab_dict['<unk>']
        self.enc_voc_size = len(self.src_vocab)
        self.dec_voc_size= len(self.trg_vocab)
    
        train_ratio = 0.8
        val_ratio = 0.1
        data_len = len(self.data)
        train_size = int(data_len * train_ratio)
        val_size = int(data_len * val_ratio)
        test_size = data_len - train_size - val_size


        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:train_size+val_size]
        self.test_data = self.data[train_size+val_size:]
  


    def __len__(self):
        if self.train_data:
            return len(self.train_data)
        elif self.val_data:
            return len(self.val_data)
        elif self.test_data:
            return len(self.test_data)
        else:
            raise ValueError("No data available!")


    def __getitem__(self, idx):
        src, trg = self.train_data[idx]
        src = [self.src_vocab_dict[token] if token in self.src_vocab_dict else self.src_vocab_dict['<unk>'] for token in src]
        trg = [self.trg_vocab_dict[token] if token in self.trg_vocab_dict else self.trg_vocab_dict['<unk>'] for token in trg]
        return [src, trg]   
       


    def __iter__(self):
        return self    



    def __next__(self):
        if self.current < len(self.train_data):
            self.current += 1
            return self.__getitem__(self.current)

        raise StopIteration 


    def printv(self):
        print(self.src_vocab_dict)

           

def custom_collate(batch, src_sos_idx, src_eos_idx, trg_sos_idx, trg_eos_idx, src_pad_idx, trg_pad_idx,src_vocab_dict:dict,trg_vocab_dict:dict):
    src_batch, trg_batch = zip(*batch)
#     print(src_batch,trg_batch)
    # print("Batch Sizes (Before Padding):", [len(src) for src in src_batch], [len(trg) for trg in trg_batch])
    src_batch = [[src_vocab_dict[token] if token in src_vocab_dict else src_vocab_dict['<unk>'] for token in src] for src in src_batch]
    trg_batch = [[trg_vocab_dict[token] if token in trg_vocab_dict else trg_vocab_dict['<unk>'] for token in trg] for trg in trg_batch]
    # Pad sequences to the fixed length max_len
    padded_src = [torch.cat([torch.tensor([src_sos_idx]), torch.tensor(src), torch.tensor([src_eos_idx]), torch.full((max_len - len(src) - 2,), src_pad_idx, dtype=torch.long)]) for src in src_batch]
    padded_trg = [torch.cat([torch.tensor([trg_sos_idx]), torch.tensor(trg), torch.tensor([trg_eos_idx]), torch.full((max_len - len(trg) - 2,), trg_pad_idx, dtype=torch.long)]) for trg in trg_batch]
    
    # Stack the padded sequences
    padded_src = torch.stack(padded_src)
    padded_trg = torch.stack(padded_trg)
    # print("Batch Sizes (Before Padding):", [len(src) for src in padded_src], [len(trg) for trg in padded_trg])
    # print("Batch Sizes (After Padding):", padded_src.shape, padded_trg.shape)
    return [padded_src, padded_trg]

         


dataset = CustomDataset('/Users/romankasichhwa/Desktop/complete/500_only/nep.txt', '/Users/romankasichhwa/Desktop/complete/500_only/eng.txt')