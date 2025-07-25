from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
from loguru import logger

class MaskedBillSumDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = []
        self.dataset = dataset
        self.pad_token_id = tokenizer.eot_token

        logger.info("Processng and masking dataset for summarization...")
        for sample in tqdm(self.dataset):
            article = f"ARTICLE: {sample['text']}{self.tokenizer.decode([self.tokenizer.eot_token])}SUMMARY: "
            summary = f"{sample['summary']}{self.tokenizer.decode([self.tokenizer.eot_token])}"

            article_tokens = self.tokenizer.encode(article, allowed_special="all")
            summary_tokens = self.tokenizer.encode(summary, allowed_special="all")

            input_ids = article_tokens + summary_tokens
            # attention_mask = [1] * len(input_ids)
            labels = [-100] * len(article_tokens) + summary_tokens

            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.pad_token_id] * padding_length
                # attention_mask += [0] * padding_length
                labels += [-100] * padding_length
            else:
                input_ids = input_ids[:self.max_length]
                # attention_mask = attention_mask[:max_length]
                labels = labels[:self.max_length]

            self.processed_data.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                # "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
