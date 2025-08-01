import os
import psutil
from datetime import datetime

from tqdm.auto import tqdm
from loguru import logger
import tiktoken
import yaml
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset

from src.model.prepare_for_fine_tune import PrepareModelWithPreTrainedWeights
from utils.load_dataset import MaskedBillSumDataset
from utils.util import calculate_metrics


class SummarizationNNHead(nn.Module):
    """A Multi-Layer Perceptron Head for Summarization"""
    def __init__(self, embedding_dim, vocab_size, hidden_dim_factor=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * hidden_dim_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_factor * embedding_dim, vocab_size),
        )

    def forward(self, x):
        return self.layers(x)
    
class MemoryUsageLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        
        ram_stats = psutil.virtual_memory()
        ram_used_gb = ram_stats.used / (1024 ** 3)

        vram_used_gb = 0
        if torch.cuda.is_available():
            device_idx = trainer.local_rank
            vram_used_gb = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)

        metrics = {"memory/ram_used_gb": ram_used_gb, "memory/vram_used_gb": vram_used_gb}
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        logger.info(f"Memory @ epoch {trainer.current_epoch}: RAM: {ram_used_gb:.2f}GB | VRAM: {vram_used_gb:.2f}GB")


# 1. Lightning Data Module
class SummarizationDataModule(pl.LightningDataModule):
    """Encapsulates Data loading Logic"""
    def __init__(self, model_config, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def setup(self, stage=None):
        train_set = load_dataset("FiscalNote/billsum", split="train")
        val_set = load_dataset("FiscalNote/billsum", split="test")
        test_set = load_dataset("FiscalNote/billsum", split="ca_test")

        print(f"Hparams: {self.hparams}")

        self.train_dataset = MaskedBillSumDataset(train_set, self.tokenizer, self.hparams.model_config['context_length'])
        self.val_dataset = MaskedBillSumDataset(val_set, self.tokenizer, self.hparams.model_config['context_length'])
        self.test_dataset = MaskedBillSumDataset(test_set, self.tokenizer, self.hparams.model_config['context_length'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.train_config["batch_size"], num_workers=self.hparams.train_config["num_workers"], drop_last=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.train_config["batch_size"], num_workers=self.hparams.train_config["num_workers"], drop_last=False, shuffle=False)
    
class SummarizationFineTuneModel(pl.LightningModule):
    def __init__(self, model_name, model_config, train_config, num_training_steps):
        super().__init__()
        self.save_hyperparameters()

        model_loader = PrepareModelWithPreTrainedWeights(model_name=self.hparams.model_name)
        self.gpt2_base = model_loader.model

        logger.info(f"Freezing all parameters of the {model_name} model.....")
        for param in self.gpt2_base.parameters():
            param.requires_grad = False

        self.summarization_head = SummarizationNNHead(
            embedding_dim=self.hparams.model_config['embedding_dim'],
            vocab_size=self.hparams.model_config['vocab_size']
        )

    def forward(self, batch):
        with torch.no_grad():
            in_idx = batch['input_ids']
            tok_embeds = self.gpt2_base.tok_embeddings(in_idx)
            pos_embeds = self.gpt2_base.pos_embeddings(torch.arange(in_idx.shape[1], device=self.device))
            input_embeds = tok_embeds + pos_embeds
            x = self.gpt2_base.drop_embeddings(input_embeds)
            x = self.gpt2_base.transformer_blocks(x)
            hidden_states = self.gpt2_base.final_norm(x)

        logits = self.summarization_head(hidden_states)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss, acc, perplexity = calculate_metrics(logits, batch['labels'])

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_perplexity", perplexity, on_epoch=True, logger=True, sync_dist=True)
        # print(f"Train Loss: {loss} | Train Acc: {acc} | Train Perplexity Score: {perplexity}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss, acc, perplexity = calculate_metrics(logits, batch['labels'])

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_epoch=True, logger=True, sync_dist=True)
        # print(f"Val loss: {loss} | Val Acc: {acc} | Val Perplexity Score: {perplexity}")
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.summarization_head.parameters(), lr=self.hparams.train_config['learning_rate'])
        num_training_steps = self.hparams.num_training_steps
        num_warmup_steps = 0.1 * num_training_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
if __name__ == "__main__":
    # Load Model Configs
    
    with open("./config/training_config.yaml", "r") as f:
        train_config_full = yaml.safe_load(f)

    with open("./config/model_config.yaml", "r") as f:
        model_config_full = yaml.safe_load(f)

    # Extract model and training config
    model_name = model_config_full["model_name"]
    model_config = model_config_full["model_configs"][model_name]
    train_config = train_config_full["train_config"]
    wandb_config = train_config_full["wandb_config"]
    experiment_path = train_config_full["experiment_path"]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb_project_name = wandb_config["project"]
    wandb_run_name = f"fine-tune-{model_name}-{timestamp}"


    os.makedirs(os.path.join(experiment_path, "checkpoints"), exist_ok=True)

    # Initialize Data Module and Lightning Module
    data_module = SummarizationDataModule(model_config, train_config)
    data_module.setup(stage="fit")

    num_training_steps = (len(data_module.train_dataset) // train_config["batch_size"]) * train_config["num_epochs"]
    model_module = SummarizationFineTuneModel(model_name, model_config, train_config, num_training_steps)

    wandb_logger = WandbLogger(
        name=wandb_run_name,
        project=wandb_project_name,
        log_model="all",
        config={"model_config": model_config, "train_config": train_config},
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_path, "checkpoints"),
        filename="summarization-gpt2-finetune-Epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")
    memory_logger_callback = MemoryUsageLogger()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        max_epochs=train_config["num_epochs"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, memory_logger_callback, early_stop_callback],
    )

    trainer.fit(model_module, datamodule=data_module)

    if trainer.is_global_zero:
        wandb.finish()

    print("Training complete!")