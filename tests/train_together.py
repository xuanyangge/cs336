import torch
import argparse
from pathlib import Path
from .adapters import MyTransformerLM, AdamW, run_get_lr_cosine_schedule, run_cross_entropy, run_gradient_clipping
import numpy.typing as npt
import numpy as np
import uuid


def parse_args():
    p = argparse.ArgumentParser(description="LLM training loop")

    # Model architecture
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=48)
    p.add_argument("--d_model", type=int, default=1600)
    p.add_argument("--num_heads", type=int, default=25)
    p.add_argument("--d_ff", type=int, default=4288)

    # Optimizer (AdamW)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)

    # Training loop
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=200)
    p.add_argument("--gradient_clipping_max", type=float, default=1)
    p.add_argument("--num_steps", type=int, default=10_000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--checkpoint_interval", type=int, default=1_000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--num_warmup_steps", type=int, default=500)
    p.add_argument("--cosine_cycle_iters", type=int, default=5000)

    # Paths
    p.add_argument("--train_data", type=Path, required=True)
    p.add_argument("--eval_data", type=Path, required=True)
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--resume_from", type=Path, default=None)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--rope_theta", type=float, default=10000)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(path, args):
    dataset = np.memmap(path, dtype=np.uint16, mode="r")

    return dataset


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    sample_start_ind = np.random.choice(len(dataset) - context_length, batch_size)
    device = torch.device(device)

    input_tensor = torch.tensor(
        np.array([dataset[i : i + context_length] for i in sample_start_ind]), device=device, dtype=torch.int
    )
    output_tensor = torch.tensor(
        np.array([dataset[i + 1 : i + context_length + 1] for i in sample_start_ind]), device=device, dtype=torch.int
    )
    return input_tensor, output_tensor


def build_model(args):
    model = MyTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        num_heads=args.num_heads,
        device=args.device,
    )
    # model = model.to(args.device)
    return model


def build_optimizer(model, args):
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    return optimizer


def train_step(model, batch, optimizer, args, step):
    (input, output) = batch
    logits = model(input)
    loss = run_cross_entropy(logits, output)

    loss.backward()
    run_gradient_clipping(model.parameters(), args.gradient_clipping_max)
    for param_group in optimizer.param_groups:
        param_group["lr"] = run_get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.num_warmup_steps, args.cosine_cycle_iters
        )

    optimizer.step()

    optimizer.zero_grad()
    return loss.item()


def evaluate(model, batch, args):
    input, output = batch
    logits = model(input)
    loss = run_cross_entropy(logits, output)
    return loss


def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "iteration": step}, out)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]


def main():
    args = parse_args()
    run_name = str(uuid.uuid4())
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_dir / "training_param.txt", "w") as f:
        f.write(str(args))

    set_seed(args.seed)

    train_data = load_dataset(args.train_data, args)
    eval_data = load_dataset(args.eval_data, args)
    eval_batch = get_batch(eval_data, args.eval_batch_size, args.context_length, args.device)

    model = build_model(args)
    optimizer = build_optimizer(model, args)

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, optimizer)

    print("Run name ", run_name)
    with open(checkpoint_dir / "training_log.txt", "w") as f:
        for step in range(start_step, args.num_steps):
            batch = get_batch(train_data, args.batch_size, args.context_length, args.device)
            loss = train_step(model, batch, optimizer, args, step)

            if step % args.log_interval == 0:
                print(f"step={step} train_loss={loss:.10f}")
                f.write(f"step={step} train_loss={loss:.10f}\n")

            if step > 0 and step % args.eval_interval == 0:
                eval_loss = evaluate(model, eval_batch, args)
                print(f"step={step} eval_loss={eval_loss:.4f}")
                f.write(f"step={step} eval_loss={eval_loss:.4f}\n")

            if step > 0 and step % args.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, step, checkpoint_dir)

    save_checkpoint(model, optimizer, args.num_steps, checkpoint_dir)


if __name__ == "__main__":
    main()
