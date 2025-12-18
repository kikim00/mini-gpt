from dataclasses import dataclass
from data import LMStream, load_text, build_ids
from model import GPTConfig, MiniGPT
from tokenizer import CharTokenizer
import torch
import time

@dataclass
class TrainingConfig:
    batch_size: int = 32
    block_size: int = 128
    lr: float = 3e-4
    max_steps: int = 10_000
    eval_interval: int = 250
    eval_steps: int = 10
    device: str = "mps"
    sample_prompt: str = """In the late summer of that year we lived in a house in a village that looked across the river and the plain to the mountains. In the bed of the river there were pebbles and boulders, dry and white in the sun, and the water was clear and swiftly moving and blue in the channels."""
    num_sample_output: int = 200

def train(config: TrainingConfig):
    print("Device:", config.device)
    text = load_text('data.txt')
    tokenizer = CharTokenizer(text)
    tokens = build_ids(tokenizer, text)
    cutoff_index = int(len(tokens) * 0.85)
    train_dataloader = LMStream(tokens[:cutoff_index], config.block_size, config.batch_size, config.device)
    eval_dataloader = LMStream(tokens[cutoff_index:], config.block_size, config.batch_size, config.device)
    
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_layers=5,
        n_heads=8,
        d_model=256,
        d_ffn=1024,
        dropout=0.1
    )
    model = MiniGPT(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    start_ts = time.time()
    for i in range(1, config.max_steps+1):
        train_loss = train_step(train_dataloader, model, optimizer)
        if i % 100 == 0:
            print(f"train loss: {train_loss}, duration: {(time.time() - start_ts):.2f} [{i}/{config.max_steps}]")
        
        if i % config.eval_interval == 0:
            eval_loop(eval_dataloader, model, config.eval_steps)
            print(sample(config.sample_prompt, model, tokenizer, config.block_size, config.device, config.num_sample_output))
        

def train_step(dataloader, model, optimizer):
    model.train()
    optimizer.zero_grad()
    X, y = dataloader.get_batch()
    logits, loss = model(X, y)
    loss.backward()
    optimizer.step()

    return loss

def eval_loop(dataloader, model, num_steps):
    model.eval()
    loss_sum = 0
    for i in range(num_steps):
        X, y = dataloader.get_batch()
        with torch.no_grad():
            _, loss = model(X, y)
            loss_sum += loss.item()
    
    print(f"eval loss: {(loss_sum / num_steps):>5f}")

def sample(original_prompt, model, tokenizer, block_size, device, num_output_tokens):
    model.eval()
    with torch.no_grad():
        output = tokenizer.encode(original_prompt)
        for _ in range(num_output_tokens):
            X = torch.tensor(output[-block_size:]) # take last N tokens
            X = X.reshape(1, -1) # shape into batch
            logits, loss = model(X.to(device))
            next_token = torch.argmax(logits[:, -1, :])
            output.append(next_token.item())
    
        return tokenizer.decode(output)

if __name__ == "__main__":
    train(TrainingConfig())
