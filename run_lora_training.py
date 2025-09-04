"""
Simple script to run LoRA fine-tuning for paraphrase detection.

This script provides an easy way to run the LoRA-enabled paraphrase detection
training with sensible defaults.
"""

import subprocess
import sys
import argparse


def run_lora_training(args):
    """Run LoRA training with the specified arguments."""
    
    cmd = [
        sys.executable, "lora_paraphrase_detection.py",
        "--para_train", args.para_train,
        "--para_dev", args.para_dev,
        "--para_test", args.para_test,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--model_size", args.model_size,
        "--seed", str(args.seed)
    ]
    
    if args.use_gpu:
        cmd.append("--use_gpu")
    
    print("Running LoRA fine-tuning with command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("LoRA training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning for paraphrase detection")
    
    # Data arguments
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv",
                       help="Path to training data")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv",
                       help="Path to dev data")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv",
                       help="Path to test data")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (recommended: 3-4)")
    parser.add_argument("--batch_size", type=int, default=12,
                       help="Batch size (Tesla T4 safe: 8-16, can try 12-20)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (LoRA recommended: 1e-4 to 2e-4)")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU for training")
    parser.add_argument("--seed", type=int, default=11711,
                       help="Random seed")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="gpt2",
                       choices=["gpt2", "gpt2-medium", "gpt2-large"],
                       help="GPT-2 model size")
    
    # LoRA arguments (Tesla T4 optimized defaults)
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank (recommended: 4-16 for good performance/memory balance)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA alpha parameter (typically rank * 2-4)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout rate (0.0-0.1 for regularization)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 OPTIMIZED LoRA Fine-tuning for Paraphrase Detection")
    print("=" * 70)
    print(f"Model: {args.model_size}")
    print(f"LoRA Configuration:")
    print(f"  - Rank: {args.lora_rank}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Dropout: {args.lora_dropout}")
    print(f"Training Parameters:")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - GPU: {args.use_gpu}")
    print("=" * 70)
    
    run_lora_training(args)


if __name__ == "__main__":
    main()