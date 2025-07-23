from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch

class MaskedBillSumDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = []
        self.dataset = dataset

        for sample in tqdm(self.dataset):
            article = f"ARTICLE: {sample['text']}{self.tokenizer.eos_token}SUMMARY: "
            summary = f"{sample['summary']}{self.tokenizer.eos_token}"

            article_encoding = self.tokenizer(article, max_length=self.max_length, truncation=True)
            summary_encoding = self.tokenizer(summary, max_length=self.max_length, truncation=True)

            input_ids = article_encoding['input_ids'] + summary_encoding['input_ids']
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(article_encoding['input_ids']) + summary_encoding['input_ids']

            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                labels += [-100] * padding_length
            else:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            self.processed_data.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
