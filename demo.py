from generate_names import load_names, build_vocab, NameGeneratorTransformer, generate_names
import torch

def main():
    print("Sanskrit Name Generator Demo")
    print("===========================\n")
    
    # Load the model and vocabulary
    try:
        # First load the names to get vocabulary size
        names = load_names('sanskrit_names.txt')
        char_to_idx, idx_to_char = build_vocab(names)
        vocab_size = len(char_to_idx) + 1
        
        # Create model instance with same parameters as training
        model = NameGeneratorTransformer(
            vocab_size=vocab_size,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            context_size=10,
            dropout=0.1
        )
        
        # Load the state dict
        state_dict = torch.load('model.pth')
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        print("Model loaded successfully!\n")
        
        # Generate names with different temperatures
        print("Generating names with different creativity levels:\n")
        
        print("Conservative (temperature=0.5):")
        names = generate_names(model, char_to_idx, idx_to_char, num_names=5, temperature=0.5)
        for name in names:
            print(f"- {name}")
        
        print("\nBalanced (temperature=0.8):")
        names = generate_names(model, char_to_idx, idx_to_char, num_names=5, temperature=0.8)
        for name in names:
            print(f"- {name}")
        
        print("\nCreative (temperature=1.0):")
        names = generate_names(model, char_to_idx, idx_to_char, num_names=5, temperature=1.0)
        for name in names:
            print(f"- {name}")
        
    except FileNotFoundError:
        print("Error: Model file (model.pth) or names file (sanskrit_names.txt) not found.")
        print("Please train the model first using generate_names.py")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Try running generate_names.py first to train and save the model.")

if __name__ == "__main__":
    main() 