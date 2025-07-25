import torch
import numpy as np
from utils.load_gpt2_weights import download_and_load_gpt2
from utils.get_config import LoadModelConfig
from src.model.model import GPT2ModelClone


class PrepareModelWithPreTrainedWeights:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._load_all()

    def _load_all(self):
        self.settings, self.params = self._get_settings_and_params()
        self.model_config = self._get_model_config()
        # print(f"Self.model_config: {self.model_config}")
        self.model = GPT2ModelClone(self.model_config)
        self.model.eval()
        self._load_gpt2_weights_into_model()
        self.model.to(self.device)

    def _assign_params(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    def _get_settings_and_params(self):
        settings, params = download_and_load_gpt2(model_size="124M", models_dir="./model_weights/")
        print(f"Settings: {settings}")
        print(f"Params: {params.keys()}")
        return settings, params

    def _get_model_config(self):
        config = LoadModelConfig()
        print(config.list_all_models())
        model_config = config.get_model_config(model_name=self.model_name)
        print(f"Returned Model config for {self.model_name}: {model_config}")
        return model_config

    def _load_gpt2_weights_into_model(self):
        self.model.pos_embeddings.weight = self._assign_params(self.model.pos_embeddings.weight, self.params["wpe"])
        self.model.tok_embeddings.weight = self._assign_params(self.model.tok_embeddings.weight, self.params["wte"])

        for block in range(len(self.params["blocks"])):
            # Load the weights for Query, key and value
            q_w, k_w, v_w = np.split(
                (self.params["blocks"][block]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.model.transformer_blocks[block].mask_attn.W_query.weight = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.W_query.weight, q_w.T)
            self.model.transformer_blocks[block].mask_attn.W_key.weight = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.W_key.weight, k_w.T)
            self.model.transformer_blocks[block].mask_attn.W_value.weight = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.W_value.weight, v_w.T)

            # Load the weights for bias
            q_b, k_b, v_b = np.split(
                (self.params["blocks"][block]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.model.transformer_blocks[block].mask_attn.W_query.bias = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.W_query.bias, q_b)
            self.model.transformer_blocks[block].mask_attn.W_key.bias = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.W_key.bias, k_b)
            self.model.transformer_blocks[block].mask_attn.W_value.bias = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.W_value.bias, v_b)
            
            # Load output layer weights
            self.model.transformer_blocks[block].mask_attn.out_proj.weight = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.out_proj.weight,
                self.params["blocks"][block]["attn"]["c_proj"]["w"].T)
            self.model.transformer_blocks[block].mask_attn.out_proj.bias = self._assign_params(
                self.model.transformer_blocks[block].mask_attn.out_proj.bias,
                self.params["blocks"][block]["attn"]["c_proj"]["b"])
            
            # Load the weights for feed forward block
            self.model.transformer_blocks[block].ffn_block.layers[0].weight = self._assign_params(
                self.model.transformer_blocks[block].ffn_block.layers[0].weight,
                self.params["blocks"][block]["mlp"]["c_fc"]["w"].T)
            self.model.transformer_blocks[block].ffn_block.layers[0].bias = self._assign_params(
                self.model.transformer_blocks[block].ffn_block.layers[0].bias,
                self.params["blocks"][block]["mlp"]["c_fc"]["b"])
            self.model.transformer_blocks[block].ffn_block.layers[2].weight = self._assign_params(
                self.model.transformer_blocks[block].ffn_block.layers[2].weight,
                self.params["blocks"][block]["mlp"]["c_proj"]["w"].T)
            self.model.transformer_blocks[block].ffn_block.layers[2].bias = self._assign_params(
                self.model.transformer_blocks[block].ffn_block.layers[2].bias,
                self.params["blocks"][block]["mlp"]["c_proj"]["b"])

            # Load Normalization weights
            self.model.transformer_blocks[block].norm_1.scale = self._assign_params(
                self.model.transformer_blocks[block].norm_1.scale,
                self.params["blocks"][block]["ln_1"]["g"])
            self.model.transformer_blocks[block].norm_1.shift = self._assign_params(
                self.model.transformer_blocks[block].norm_1.shift,
                self.params["blocks"][block]["ln_1"]["b"])
            self.model.transformer_blocks[block].norm_2.scale = self._assign_params(
                self.model.transformer_blocks[block].norm_2.scale,
                self.params["blocks"][block]["ln_2"]["g"])
            self.model.transformer_blocks[block].norm_2.shift = self._assign_params(
                self.model.transformer_blocks[block].norm_2.shift,
                self.params["blocks"][block]["ln_2"]["b"])
            
        self.model.final_norm.scale = self._assign_params(self.model.final_norm.scale, self.params["g"])
        self.model.final_norm.shift = self._assign_params(self.model.final_norm.shift, self.params["b"])
        self.model.out_head.weight = self._assign_params(self.model.out_head.weight, self.params["wte"])
        
    
        