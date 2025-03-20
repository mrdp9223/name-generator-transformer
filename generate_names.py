import os
import unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader, random_split

# Data Preparation Functions
def load_names(file_path):
    """Load names from a text file and normalize them."""
    with open(file_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    
    # Normalize Unicode (important for Tamil, Sanskrit) and append end token '.'
    names = [unicodedata.normalize('NFKC', name) + '.' for name in names]
    
    return list(set(names))  # Remove duplicates

def build_vocab(names):
    """Create a character vocabulary and mappings."""
    unique_chars = sorted(set("".join(names)))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars, start=1)}  # 1-based index
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

def encode_names(names, char_to_idx):
    """Convert names into sequences of character indices."""
    return [[char_to_idx[char] for char in name] for name in names]

def clean_generated_name(name):
    """Remove invalid starting diacritics and fix character sequencing issues."""
    if name and name[0] in "ंः्":  # Remove standalone diacritics at the start
        name = name[1:]
    return name

# Dataset Class
class NameDataset(Dataset):
    """Custom dataset for name generation with fixed context size."""
    def __init__(self, encoded_names, context_size=10):
        self.data = []
        self.context_size = context_size
        for name in encoded_names:
            for i in range(1, len(name)):
                context = name[max(0, i - context_size):i]  # Fixed-length context window
                target = name[i]  # Next character
                self.data.append((context, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = [0] * (self.context_size - len(x)) + x  # Left-pad with 0s
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Transformer-based Name Generator Model
class NameGeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=10, num_layers=3, ff_dim=64, context_size=10, dropout=0.1):
        super(NameGeneratorTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, context_size, embed_dim))
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding  # Add embeddings + positional encodings
        x = self.transformer(x)  # Pass through Transformer encoder
        out = self.fc(x[:, -1, :])  # Predict next character from last token
        return out

def generate_names(model, char_to_idx, idx_to_char, num_names=5, max_length=20, temperature=1.0):
    """Generate multiple names with random starting letters."""
    model.eval()
    context_size = model.positional_encoding.shape[1]
    generated_names = []
    
    starting_chars = list(char_to_idx.keys())
    starting_chars.remove('.')  # Remove end token from selection
    
    for _ in range(num_names):
        start_char = random.choice(starting_chars)
        name_indices = [char_to_idx[start_char]]
        
        for _ in range(max_length):
            input_seq = name_indices[-context_size:]  # Ensure input is at most context_size
            input_seq = [0] * (context_size - len(input_seq)) + input_seq  # Left-pad
            input_tensor = torch.tensor([input_seq], dtype=torch.long)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output / temperature, dim=-1)
                next_idx = torch.multinomial(probabilities, num_samples=1).item()
            
            if next_idx == char_to_idx['.']:
                break  # Stop when '.' is generated
            
            name_indices.append(next_idx)
        
        generated_name = "".join(idx_to_char[idx] for idx in name_indices)
        generated_names.append(clean_generated_name(generated_name))
    
    return generated_names

if __name__ == "__main__":
    # Training Setup
    file_path = "sanskrit_names.txt"  # Replace with actual file path
    context_size = 10  # Define context size
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    ff_dim = 64
    dropout = 0.1
    batch_size = 16
    epochs = 50  # Increased epochs for better learning
    early_stopping_patience = 5  # Stop training if validation loss doesn't improve
    learning_rate = 0.005  # Reduced learning rate for stability

    if os.path.exists(file_path):
        names = load_names(file_path)
        print(f"Loaded {len(names)} unique names.")
        
        char_to_idx, idx_to_char = build_vocab(names)
        vocab_size = len(char_to_idx) + 1  # Include padding index
        
        encoded_names = encode_names(names, char_to_idx)
        dataset = NameDataset(encoded_names, context_size)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = NameGeneratorTransformer(vocab_size, embed_dim, num_heads, num_layers, ff_dim, context_size, dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training Loop with Validation and Early Stopping
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the model with state dict instead of full model
                torch.save(model.state_dict(), 'model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
        
        # Generate example names
        print("\nGenerating example names:")
        generated_names = generate_names(model, char_to_idx, idx_to_char, num_names=5, temperature=0.8)
        for name in generated_names:
            print(f"Generated name: {name}")
    else:
        print("File not found.")
