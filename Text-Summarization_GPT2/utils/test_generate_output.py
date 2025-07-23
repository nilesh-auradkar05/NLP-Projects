import torch
import tiktoken

from src.model.prepare_for_fine_tune import PrepareModelWithPreTrainedWeights

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuned_model = PrepareModelWithPreTrainedWeights(model_name="gpt2-medium (355M)", device=device)
    model = tuned_model.model
    config = tuned_model.model_config
    print(f"Settings: {config}")
    token_ids = generate(model=model,
                         idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
                         max_new_tokens=25,
                         context_size=config["context_length"],
                         top_k=50,
                         temperature=0.7)
    
    print("Output text: \n", token_ids_to_text(token_ids, tokenizer))