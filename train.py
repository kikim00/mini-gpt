import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass

import torch
import yaml

from data import LMStream, build_ids, load_text
from model import GPTConfig, MiniGPT
from tokenizer import CharTokenizer


@dataclass
class TrainingConfig:
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def override(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

    batch_size: int = 64
    block_size: int = 512
    lr: float = 3e-4
    max_steps: int = 2_000
    eval_interval: int = 250
    eval_steps: int = 50
    device: str = "cuda"
    sample_prompt: str = (
        "In the late summer of that year we lived in a house in a village that "
        "looked across the river "
    )
    num_sample_output: int = 200
    sample_temperature: float = 0.9
    sample_topk: int = 40
    training_data_char_limit: int = -1  # -1 means use all data
    use_positional_embedding: bool = True
    tie_embedding_weights: bool = True
    use_residual_connections: bool = True
    num_transformer_layers: int = 4
    training_eval_steps: int = 100
    enable_post_ln: bool = False


def train(config: TrainingConfig):
    logger = logging.getLogger(__name__)

    logger.info("Starting training with config: %s", config)
    text = load_text("data.txt")[:config.training_data_char_limit]
    tokenizer = CharTokenizer(text)
    tokens = build_ids(tokenizer, text)
    cutoff_index = int(len(tokens) * 0.85)
    train_dataloader = LMStream(
        tokens[:cutoff_index], config.block_size, config.batch_size, config.device
    )
    eval_dataloader = LMStream(
        tokens[cutoff_index:], config.block_size, config.batch_size, config.device
    )

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_layers=config.num_transformer_layers,
        n_heads=4,
        d_model=128,
        d_ffn=512,
        dropout=0.1,
        use_positional_embedding=config.use_positional_embedding,
        tie_embedding_weights=config.tie_embedding_weights,
        use_residual_connections=config.use_residual_connections,
        enable_post_ln=config.enable_post_ln,
    )
    model = MiniGPT(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    start_ts = time.time()
    train_loss_sum = 0.0

    for i in range(1, config.max_steps + 1):
        train_loss, logits = train_step(train_dataloader, model, optimizer)
        train_loss_sum += train_loss.item()

        if i % config.training_eval_steps == 0:
            duration = time.time() - start_ts
            logger.info(
                "train loss: %.5f, duration: %.2f seconds [Step %s/%s]",
                train_loss_sum / config.training_eval_steps,
                duration,
                i,
                config.max_steps,
            )
            # logger.info("logits mean: %.5f, max: %.5f", logits.mean().item(), logits.max().item())
            train_loss_sum = 0.0

        if i % config.eval_interval == 0:
            eval_loop(eval_dataloader, model, config.eval_steps)
            logger.info("Sampling after step %s: %s", i, 
                sample(
                    config.sample_prompt,
                    model,
                    tokenizer,
                    config.block_size,
                    config.device,
                    config.num_sample_output,
                    config.sample_temperature,
                    config.sample_topk,
                )
            )


def train_step(dataloader, model, optimizer):
    model.train()
    optimizer.zero_grad()
    X, y = dataloader.get_batch()
    logits, loss = model(X, y)
    loss.backward()
    optimizer.step()

    return loss, logits


def eval_loop(dataloader, model, num_steps):
    logger = logging.getLogger(__name__)
    model.eval()
    loss_sum = 0
    for i in range(num_steps):
        X, y = dataloader.get_batch()
        with torch.no_grad():
            _, loss = model(X, y)
            loss_sum += loss.item()

    logger.info("eval loss: %.5f", (loss_sum / num_steps))


def sample(
    original_prompt,
    model,
    tokenizer,
    block_size,
    device,
    num_output_tokens,
    temperature,
    topk,
):
    model.eval()
    with torch.no_grad():
        output = tokenizer.encode(original_prompt)
        for _ in range(num_output_tokens):
            X = torch.tensor(output[-block_size:], device=device)  # take last N tokens
            X = X.reshape(1, -1)  # shape into batch
            logits, loss = model(X)
            logits_last = logits[:, -1, :] / temperature

            # apply top-k filtering
            topk_values, topk_indices = torch.topk(logits_last, k=topk)
            masked_logits = torch.full_like(logits_last, float("-inf"))
            masked_logits.scatter_(-1, topk_indices, topk_values)

            # apply softmax and then sample next token
            probs_last = torch.nn.functional.softmax(masked_logits, dim=-1)
            next_token = torch.multinomial(probs_last, num_samples=1)
            output.append(next_token.item())

        return tokenizer.decode(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # training
    p.add_argument("--batch-size", type=int)
    p.add_argument("--block-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--eval-interval", type=int)
    p.add_argument("--eval-steps", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--training-data-char-limit", type=int)  # -1 means use all data

    # sampling
    p.add_argument("--num-sample-output", type=int)
    p.add_argument("--sample-temperature", type=float)
    p.add_argument("--sample-prompt", type=str)
    p.add_argument("--sample-topk", type=int)

    # whole config
    p.add_argument("--experiment-path", type=str, default=None)
    p.add_argument("--config", type=str, default=None)

    # model
    p.add_argument(
        "--use-positional-embedding",
        action="store_true",
        dest="use_positional_embedding",
        help="Enable positional embeddings",
    )
    p.add_argument(
        "--no-positional-embedding",
        action="store_false",
        dest="use_positional_embedding",
        help="Disable positional embeddings",
    )
    p.set_defaults(use_positional_embedding=None)
    p.add_argument(
        "--tie-embedding-weights",
        action="store_true",
        dest="tie_embedding_weights",
        help="Tie weights between token embedding and final linear layer",
    )
    p.add_argument(
        "--no-tie-embedding-weights",
        action="store_false",
        dest="tie_embedding_weights",
        help="Do not tie weights between token embedding and final linear layer",
    )
    p.set_defaults(tie_embedding_weights=None)
    p.add_argument(
        "--use-residual-connections",
        action="store_true",
        dest="use_residual_connections",
        help="Enable residual connections in transformer blocks",
    )
    p.add_argument(
        "--no-residual-connections",
        action="store_false",
        dest="use_residual_connections",
        help="Disable residual connections in transformer blocks",
    )
    p.set_defaults(use_residual_connections=None)
    p.add_argument(
        "--num-transformer-layers",
        type=int,
        help="Number of transformer layers in the model",
    )
    p.add_argument(
        "--enable-post-ln",
        action="store_true",
        dest="enable_post_ln",
        help="Enable post-layer normalization",
    )
    p.add_argument(
        "--disable-post-ln",
        action="store_false",
        dest="enable_post_ln",
        help="Disable post-layer normalization",
    )
    p.set_defaults(enable_post_ln=None)

    args = p.parse_args()

    # Load config. Priority: config file in experiment_path > config file > defaults
    if args.experiment_path is not None:
        config = TrainingConfig.from_yaml(
            os.path.join(args.experiment_path, "config.yaml")
        )
    elif args.config is not None:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    config.override(
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
        sample_topk=args.sample_topk,
        training_data_char_limit=args.training_data_char_limit,
        use_positional_embedding=args.use_positional_embedding,
        tie_embedding_weights=args.tie_embedding_weights,
        use_residual_connections=args.use_residual_connections,
        num_transformer_layers=args.num_transformer_layers,
        enable_post_ln=args.enable_post_ln,
    )

    # set up logger
    log_path = "train.log" if args.experiment_path is None else os.path.join(
        args.experiment_path, "train.log"
    )

    # delete existing log file
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # start training
    train(config)
