from src.utils.config import load_config, GlobalConfig

def main():
    # Example usage: python src/training/train_flat.py --config configs/flat.yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/flat.yaml")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    print(f"Initializing model {config.model.base_id} and tokenizer...")
    print(f"Starting {config.training.algorithm} training for Flat Baseline...")
    print(f"Logging to {config.logging['output_dir']}")
    # trainer = GRPOTrainer(...)
    # trainer.train()

if __name__ == "__main__":
    main()
