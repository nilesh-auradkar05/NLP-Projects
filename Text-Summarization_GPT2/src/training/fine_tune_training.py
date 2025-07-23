import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm
from loguru import logger
import tiktoken

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from transformers import get_cosine_schedule_with_warmup

from src.model.prepare_for_fine_tune import PrepareModelWithPreTrainedWeights
from utils.load_dataset import MaskedBillSumDataset
from utils.get_config import LoadModelConfig
from utils.util import calculate_metrics


class SummarizationNNHead(nn.Module):
    """A Multi-Layer Perceptron Head for Summarization"""
    def __init__(self, embedding_dim, vocab_size, hidden_dim_factor=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * hidden_dim_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_factor * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)


# 1. Lightning Data Module
class SummarizationDataModule(pl.LightningDataModule):
    """Encapsulates Data loading Logic"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        train_set = load_dataset("FiscalNote/billsum", split="train")
        val_set = load_dataset("FiscalNote/billsum", split="test")
        test_set = load_dataset("FiscalNote/billsum", split="ca_test")

        self.train_dataset = MaskedBillSumDataset(train_set, self.tokenizer, self.cfg['context_length'])
        self.val_dataset = MaskedBillSumDataset(val_set, self.tokenizer, self.cfg['context_length'])
        self.test_dataset = MaskedBillSumDataset(test_set, self.tokenizer, self.cfg['context_length'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg["batch_size"], num_workers=self.cfg["num_workers"], drop_last=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg["batch_size"], num_workers=self.cfg["num_workers"], drop_last=False, shuffle=False)
    
class SummarizationFineTuneModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        model_loader = PrepareModelWithPreTrainedWeights(model_name=self.hparams.model_name)
        self.gpt2_base = model_loader.model

        for param in self.gpt2_base.parameters():
            param.requires_grad = False

        self.summarization_head = SummarizationNNHead(
            embedding_dim=self.hparams['embedding_dim'],
            vocab_size=self.hparams['vocab_size']
        )

    def forward(self, batch):
        with torch.no_grad():
            x = self.gpt2_base.tok_embeddings(batch['input_ids']) + self.gpt2_base.pos_embeddings(torch.arange(batch['input_ids'].shape[1], device=self.device))
            x = self.gpt2_base.drop_embeddings(x)
            x = self.gpt2_base.transformer_blocks(x)
            hidden_states = self.gpt2_base.final_norm(x)

        logits = self.summarization_head(hidden_states)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss, acc, perplexity = calculate_metrics(logits, batch['labels'])

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_perplexity', perplexity, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss, acc, perplexity = calculate_metrics(logits, batch['labels'])

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.summarization_head.parameters(), lr=self.hparams['learning_rate'])
        num_warmup_steps = 0.1 * self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
if __name__ == "__main__":
    # Load Model Configs
    
    ...