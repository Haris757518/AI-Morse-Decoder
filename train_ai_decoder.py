import random, math, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from tqdm import tqdm
from secure_core import text_to_morse, morse_to_signal, obfuscate_signal

CLASSES = ["dot", "dash", "short_gap", "letter_gap", "word_gap"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def label_signal(clean_sig):
    """Label clean signal with proper symbol classification"""
    labels = []
    for s, d in clean_sig:
        if s == "on":
            labels.append("dot" if d < 150 else "dash")
        else:
            if d < 200: 
                labels.append("short_gap")
            elif d < 500: 
                labels.append("letter_gap")
            else: 
                labels.append("word_gap")
    return labels

def gen_training_data(samples=5000, seq_len=64):
    """Generate training data with EXTREME noise to train robust decoder"""
    # Extended vocabulary for better generalization
    texts = [
        "sos", "help", "code red", "alpha", "bravo", "charlie", "delta",
        "hi", "hello", "test", "signal", "morse", "data", "secure",
        "go", "stop", "yes", "no", "ok", "run", "wait", "abort",
        "fire", "water", "air", "land", "sky", "sea", "one", "two",
        "three", "four", "five", "roger", "wilco", "copy", "out",
        "emergency", "alert", "warning", "danger", "safe", "clear"
    ]
    
    X, y = [], []
    
    for _ in tqdm(range(samples), desc="🔥 Generating EXTREME noise training data"):
        text = random.choice(texts)
        morse = text_to_morse(text)
        base = random.randint(70, 150)
        clean = morse_to_signal(morse, base_unit=base)
        
        # CRITICAL: Focus on EXTREME noise (150-250% distortion)
        # 70% extreme noise, 30% moderate noise
        if random.random() < 0.7:
            max_scale = random.uniform(2.5, 3.5)  # EXTREME: 150-250% noise
        else:
            max_scale = random.uniform(1.8, 2.5)  # Moderate: 80-150% noise
            
        noisy, _, _ = obfuscate_signal(clean, max_scale=max_scale)
        labels = label_signal(clean)
        
        # Enhanced features: [normalized_duration, on/off_state, duration_variance]
        feats = []
        for i, (s, d) in enumerate(noisy):
            norm_dur = d / 500.0  # Normalize
            state = 1.0 if s == "on" else 0.0
            feats.append([norm_dur, state])
        
        lbls = [CLASS_TO_IDX[l] for l in labels]

        # Sequence handling
        if len(feats) < seq_len:
            pad_len = seq_len - len(feats)
            feats += [[0.0, 0.0]] * pad_len
            lbls += [0] * pad_len
        else:
            feats = feats[:seq_len]
            lbls = lbls[:seq_len]
            
        X.append(feats)
        y.append(lbls)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class RobustMorseDecoder(nn.Module):
    """
    Deep bidirectional GRU with attention mechanism
    Designed to handle EXTREME noise that breaks rule-based decoders
    """
    def __init__(self, input_size=2, hidden_size=192, num_classes=5):
        super().__init__()
        
        # Input processing
        self.input_norm = nn.BatchNorm1d(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Deep bidirectional GRU stack
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True, 
                           bidirectional=True, dropout=0.3)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, 
                           bidirectional=True, dropout=0.3)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, 
                                               dropout=0.2, batch_first=True)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Input normalization (per feature)
        batch_size, seq_len, features = x.shape
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)  # (B, T, F)
        
        # Project input
        x = self.input_proj(x)
        
        # First GRU layer
        out, _ = self.gru1(x)
        out = self.norm1(out)
        
        # Second GRU layer
        out, _ = self.gru2(out)
        out = self.norm2(out)
        
        # Self-attention
        attn_out, _ = self.attention(out, out, out)
        out = out + attn_out  # Residual connection
        
        # Classification
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_robust_model():
    """Train the robust decoder on extreme noise"""
    print("\n" + "="*70)
    print("  🚀 TRAINING ROBUST AI DECODER FOR EXTREME NOISE")
    print("="*70)
    
    # Generate challenging training data
    X, y = gen_training_data(samples=5000, seq_len=64)
    
    print("\n📊 Dataset Statistics:")
    print(f"   Samples: {len(X)}")
    print(f"   Sequence Length: {X.shape[1]}")
    print(f"   Features: {X.shape[2]}")
    print(f"   Classes: {len(CLASSES)}")
    
    # Initialize model
    model = RobustMorseDecoder(hidden_size=192)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model Parameters: {total_params:,}")
    
    # Training setup
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.003, epochs=50, steps_per_epoch=1,
        pct_start=0.3, anneal_strategy='cos'
    )
    loss_fn = nn.CrossEntropyLoss()

    print("\n🎯 Training Progress:")
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        model.train()
        opt.zero_grad()
        
        # Forward pass
        out = model(X)
        loss = loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = out.argmax(-1)
            mask = (y != 0) | (X[:, :, 0] != 0)
            acc = (preds[mask] == y[mask]).float().mean().item()
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': acc,
                'epoch': epoch
            }, "robust_decoder.pth")
            print(f"   Epoch {epoch+1:2d}/50 | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}% ✓ SAVED")
        else:
            patience_counter += 1
            print(f"   Epoch {epoch+1:2d}/50 | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%")
            
        # Early stopping
        if patience_counter >= patience and epoch > 20:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

    print(f"\n✅ Training Complete!")
    print(f"   Best Accuracy: {best_acc*100:.2f}%")
    print(f"   Model saved to: robust_decoder.pth")
    print("="*70 + "\n")

if __name__ == "__main__":
    train_robust_model()