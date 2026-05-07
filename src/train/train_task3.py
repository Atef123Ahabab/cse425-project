"""Training / decoding script for Task 3: Transformer decoder (minimal).
"""
import argparse
import torch
import torch.nn as nn
from src.data import make_dataloader
from src.models.transformer_decoder import TransformerDecoderModel
import os
import yaml
import json
import numpy as np
from src.utils.midi_io import tokens_to_midi


def train_epoch(model, loader, optim, device):
    model.train()
    total = 0.0
    for x, lengths in loader:
        x = x.to(device)
        optim.zero_grad()
        # teacher forcing: feed inputs as targets
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        loss.backward()
        optim.step()
        total += loss.item()
    return total / len(loader)


def evaluate_perplexity(model, loader, device):
    """Calculate perplexity on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, lengths in loader:
            x = x.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += x.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def evaluate_comprehensive(model, val_loader, generated_dir, device, config):
    """Comprehensive evaluation including all metrics."""
    from src.utils.metrics import evaluate_generated_compositions, create_evaluation_report
    
    print("=== Comprehensive Evaluation ===")
    
    # Perplexity evaluation
    perplexity = evaluate_perplexity(model, val_loader, device)
    print(f"✓ Perplexity: {perplexity:.2f}")
    
    # Advanced metrics for generated compositions
    eval_results = evaluate_generated_compositions(generated_dir)
    print(f"✓ Evaluated {eval_results.get('num_compositions', 0)} compositions")
    
    if 'summary' in eval_results:
        summary = eval_results['summary']
        print(f"✓ Avg Rhythm Diversity: {summary.get('avg_rhythm_diversity', 0):.3f}")
        print(f"✓ Avg Repetition Ratio: {summary.get('avg_repetition_ratio', 0):.3f}")
    
    # Create comprehensive report
    report_path = os.path.join(generated_dir, "comprehensive_evaluation_report.md")
    create_evaluation_report(eval_results, perplexity, report_path)
    print(f"✓ Comprehensive report saved to {report_path}")
    
    # Save detailed results
    results = {
        "perplexity": perplexity,
        "composition_metrics": eval_results,
        "model_config": config,
        "training_epochs": 5
    }
    with open(os.path.join(generated_dir, "comprehensive_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return perplexity, eval_results


def generate_sequence(model, start_tokens, max_length=512, temperature=1.0, device='cpu'):
    """Autoregressive generation of token sequence."""
    model.eval()
    generated = start_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_length - len(start_tokens)):
            # Get next token prediction
            logits = model(generated.unsqueeze(0).to(device))
            next_logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Append to sequence
            generated = torch.cat([generated, torch.tensor([next_token])])
            
            # Stop if we generate padding token (vocab_size - 1)
            if next_token == model.embedding.num_embeddings - 1:
                break
    
    return generated


def generate_compositions(model, num_compositions=10, seq_length=512, device='cpu'):
    """Generate multiple music compositions."""
    compositions = []
    
    # Load vocabulary for start tokens
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'vocab.json')
    if not os.path.exists(vocab_path):
        # Try absolute path
        vocab_path = os.path.join(os.getcwd(), 'data', 'raw', 'vocab.json')
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Find some common starting tokens (middle C note-on)
    start_pitch = 60  # Middle C
    start_token = vocab.get(f"note_on_{start_pitch}", 60)  # fallback to 60
    start_tokens = torch.tensor([start_token])
    
    for i in range(num_compositions):
        print(f"Generating composition {i+1}/{num_compositions}...")
        generated_tokens = generate_sequence(model, start_tokens, max_length=seq_length, device=device)
        compositions.append(generated_tokens.numpy())
        
        # Save as MIDI file
        midi_path = f"outputs/task3/composition_{i+1:02d}.mid"
        try:
            tokens_to_midi(generated_tokens.numpy(), midi_path)
        except Exception as e:
            print(f"Warning: Could not save MIDI {i+1}: {e}")
    
    return compositions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3.yaml")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="outputs/task3")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument("--generate", action="store_true", help="Generate compositions after training")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get('device', 'cpu'))
    batch_size = config.get('batch_size', 8)
    seq_len = config.get('seq_len', 512)
    
    # Training
    if args.epochs > 0:
        loader = make_dataloader(args.data_dir, batch_size=batch_size, seq_len=seq_len)
        model = TransformerDecoderModel()
        model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))

        os.makedirs(args.output_dir, exist_ok=True)
        for epoch in range(args.epochs):
            loss = train_epoch(model, loader, optim, device)
            print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f}")
            torch.save({"model_state_dict": model.state_dict()}, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1:03d}.pt"))
    
    # Evaluation
    if args.evaluate:
        val_loader = make_dataloader(args.data_dir, split="val", batch_size=batch_size, seq_len=seq_len)
        perplexity, eval_results = evaluate_comprehensive(model, val_loader, args.output_dir, device, config)
    
    # Generation
    if args.generate:
        print("\n=== Generating Compositions ===")
        checkpoint_path = os.path.join(args.output_dir, "checkpoint_epoch_005.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model = TransformerDecoderModel()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            
            compositions = generate_compositions(model, num_compositions=10, seq_length=512, device=device)
            print(f"Generated {len(compositions)} compositions in {args.output_dir}/")
        else:
            print("No checkpoint found for generation")


if __name__ == "__main__":
    main()
