from dataclasses import dataclass
from data import LMStream, load_text, build_ids
from model import GPTConfig, MiniGPT
from tokenizer import CharTokenizer
import torch
import time
import argparse

@dataclass
class TrainingConfig:
    batch_size: int = 32
    block_size: int = 256
    lr: float = 3e-4
    max_steps: int = 10_000
    eval_interval: int = 250
    eval_steps: int = 50
    device: str = "mps"
    sample_prompt: str = """In the late summer of that year we lived in a house in a village that looked across the river and the plain to the mountains. In the bed of the river there were pebbles and boulders, dry and white in the sun, and the water was clear and swiftly moving and blue in the channels."""
    num_sample_output: int = 200
    sample_temperature: float = 0.9
    sample_topk: int = 40

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
        n_layers=3,
        n_heads=8,
        d_model=128,
        d_ffn=512,
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
            print(sample(
                config.sample_prompt, 
                model, 
                tokenizer, 
                config.block_size, 
                config.device, 
                config.num_sample_output, 
                config.sample_temperature,
                config.sample_topk))
        

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

def sample(original_prompt, model, tokenizer, block_size, device, num_output_tokens, temperature, topk):
    model.eval()
    with torch.no_grad():
        output = tokenizer.encode(original_prompt)
        for _ in range(num_output_tokens):
            X = torch.tensor(output[-block_size:], device=device) # take last N tokens
            X = X.reshape(1, -1) # shape into batch
            logits, loss = model(X)
            logits_last = logits[:, -1, :] / temperature

            # apply top-k filtering
            topk_values, topk_indices = torch.topk(logits_last, k=topk)
            masked_logits = torch.full_like(logits_last, float('-inf'))
            masked_logits.scatter_(-1, topk_indices, topk_values)

            # apply softmax and then sample next token
            probs_last = torch.nn.functional.softmax(masked_logits, dim=-1)
            next_token = torch.multinomial(probs_last, num_samples=1)
            output.append(next_token.item())
    
        return tokenizer.decode(output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # training
    p.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    p.add_argument("--block-size", type=int, default=TrainingConfig.block_size)
    p.add_argument("--lr", type=float, default=TrainingConfig.lr)
    p.add_argument("--max-steps", type=int, default=TrainingConfig.max_steps)
    p.add_argument("--eval-interval", type=int, default=TrainingConfig.eval_interval)
    p.add_argument("--eval-steps", type=int, default=TrainingConfig.eval_steps)
    p.add_argument("--device", type=str, default=TrainingConfig.device)

    # sampling
    p.add_argument("--num-sample-output", type=int, default=TrainingConfig.num_sample_output)
    p.add_argument("--sample-temperature", type=float, default=TrainingConfig.sample_temperature)
    p.add_argument("--sample-prompt", type=str, default=TrainingConfig.sample_prompt)
    p.add_argument("--sample-topk", type=int, default=TrainingConfig.sample_topk)

    args = p.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        device=args.device,
        sample_prompt=args.sample_prompt,
        num_sample_output=args.num_sample_output,
        sample_temperature=args.sample_temperature,
        sample_topk=args.sample_topk
    )

    train(config)
