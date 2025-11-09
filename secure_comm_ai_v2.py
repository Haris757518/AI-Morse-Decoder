
import torch
import numpy as np
import statistics
from secure_core import (text_to_morse, morse_to_signal, obfuscate_signal, 
                         show_signals, MORSE_CODE_DICT_REVERSE)

CLASSES = ["dot", "dash", "short_gap", "letter_gap", "word_gap"]

class RobustMorseDecoder(torch.nn.Module):
    """Must match training architecture"""
    def __init__(self, input_size=2, hidden_size=192, num_classes=5):
        super().__init__()
        self.input_norm = torch.nn.BatchNorm1d(input_size)
        self.input_proj = torch.nn.Linear(input_size, hidden_size)
        self.gru1 = torch.nn.GRU(hidden_size, hidden_size, batch_first=True, 
                                 bidirectional=True, dropout=0.3)
        self.gru2 = torch.nn.GRU(hidden_size * 2, hidden_size, batch_first=True, 
                                 bidirectional=True, dropout=0.3)
        self.norm1 = torch.nn.LayerNorm(hidden_size * 2)
        self.norm2 = torch.nn.LayerNorm(hidden_size * 2)
        self.attention = torch.nn.MultiheadAttention(hidden_size * 2, num_heads=4, 
                                                     dropout=0.2, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.4)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.permute(0, 2, 1)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        out, _ = self.gru1(x)
        out = self.norm1(out)
        out, _ = self.gru2(out)
        out = self.norm2(out)
        attn_out, _ = self.attention(out, out, out)
        out = out + attn_out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def load_robust_model(path="robust_decoder.pth"):
    """Load the trained robust model"""
    model = RobustMorseDecoder(hidden_size=192)
    try:
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        acc = checkpoint.get('accuracy', 0) * 100
        print(f"[AI MODEL] ✓ Loaded robust_decoder.pth (Training Acc: {acc:.1f}%)")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("[INFO] Run: python train_robust_decoder.py")
        return None

def standard_decoder(noisy):
    """
    Traditional rule-based decoder using statistical thresholds
    FAILS on extreme noise (>150% distortion)
    """
    if not noisy:
        return "", "ERROR: Empty signal"
    
    try:
        on_durations = [d for s, d in noisy if s == "on"]
        off_durations = [d for s, d in noisy if s == "off"]
        
        if not on_durations or not off_durations:
            return "", "ERROR: Invalid signal"
        
        # Statistical thresholds
        on_median = statistics.median(on_durations)
        off_median = statistics.median(off_durations)
        
        dot_dash_threshold = on_median * 1.5
        letter_gap_threshold = off_median * 2.0
        word_gap_threshold = off_median * 4.0
        
        # Decode
        morse_chars = []
        words = []
        current_char = ""
        
        for state, duration in noisy:
            if state == "on":
                if duration < dot_dash_threshold:
                    current_char += "."
                else:
                    current_char += "-"
            else:
                if duration >= word_gap_threshold:
                    if current_char:
                        morse_chars.append(current_char)
                        current_char = ""
                    if morse_chars:
                        words.append(" ".join(morse_chars))
                        morse_chars = []
                elif duration >= letter_gap_threshold:
                    if current_char:
                        morse_chars.append(current_char)
                        current_char = ""
        
        if current_char:
            morse_chars.append(current_char)
        if morse_chars:
            words.append(" ".join(morse_chars))
        
        # Convert to text
        text_words = []
        for word in words:
            word_text = ""
            for morse_code in word.split():
                char = MORSE_CODE_DICT_REVERSE.get(morse_code, "?")
                word_text += char
            text_words.append(word_text)
        
        morse_display = " / ".join(words)
        text_output = " ".join(text_words)
        
        return morse_display, text_output
    except Exception as e:
        return "", f"ERROR: {str(e)}"

def ai_decoder(noisy, model):
    """AI-powered decoder that handles extreme noise"""
    if not noisy:
        return "", "ERROR: Empty signal"
    
    # Prepare features
    X = np.array([[d / 500.0, 1.0 if s == "on" else 0.0] for s, d in noisy], np.float32)
    original_len = len(X)
    
    # Pad to model's expected length
    if len(X) < 64:
        padding = np.zeros((64 - len(X), 2), dtype=np.float32)
        X = np.vstack([X, padding])
    else:
        X = X[:64]
    
    X = torch.from_numpy(X).unsqueeze(0)

    with torch.no_grad():
        preds = model(X)
        probs = torch.softmax(preds, dim=-1)
        labels = probs.argmax(-1).squeeze(0).cpu().numpy()
        confidences = probs.max(-1).values.squeeze(0).cpu().numpy()

    # Reconstruct with state machine
    morse_chars = []
    words = []
    current_char = ""
    
    for i in range(min(original_len, 64)):
        symbol = CLASSES[labels[i]]
        state = noisy[i][0]
        
        if state == "on":
            if symbol == "dot":
                current_char += "."
            elif symbol == "dash":
                current_char += "-"
        elif state == "off":
            if symbol == "word_gap":
                if current_char:
                    morse_chars.append(current_char)
                    current_char = ""
                if morse_chars:
                    words.append(" ".join(morse_chars))
                    morse_chars = []
            elif symbol == "letter_gap":
                if current_char:
                    morse_chars.append(current_char)
                    current_char = ""
    
    if current_char:
        morse_chars.append(current_char)
    if morse_chars:
        words.append(" ".join(morse_chars))
    
    # Convert to text
    text_words = []
    for word in words:
        word_text = ""
        for morse_code in word.split():
            char = MORSE_CODE_DICT_REVERSE.get(morse_code, "?")
            word_text += char
        text_words.append(word_text)
    
    morse_display = " / ".join(words)
    text_output = " ".join(text_words)
    
    return morse_display, text_output

def calculate_accuracy(original, decoded):
    """Calculate character-level accuracy"""
    if not decoded or "ERROR" in decoded:
        return 0.0
    
    orig = original.upper().replace(" ", "")
    dec = decoded.upper().replace(" ", "")
    
    if not dec:
        return 0.0
    
    matches = sum(c1 == c2 for c1, c2 in zip(orig, dec))
    total = max(len(orig), len(dec))
    return (matches / total) * 100 if total > 0 else 0.0

def run_demo():
    """Demonstrate AI decoder superiority on extreme noise"""
    print("\n" + "="*80)
    print("  Proving AI can decode what traditional methods CANNOT")
    print("="*80)
    
    model = load_robust_model()
    if model is None:
        exit(1)
    
    # Test messages
    test_messages = ["code red", "sos help", "alpha bravo", "secure link"]
    
    print("\n📝 Select test message:")
    for i, msg in enumerate(test_messages, 1):
        print(f"   {i}. {msg}")
    print(f"   5. Custom message")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "5":
        text = input("Enter custom message: ").strip() or "test"
    elif choice in ["1", "2", "3", "4"]:
        text = test_messages[int(choice) - 1]
    else:
        text = "code red"
    
    # Select noise level
    print("\n🎚️  Select noise level:")
    print("   1. Extreme (200% - Standard decoder FAILS)")
    print("   2. Ultra Extreme (250% - Maximum challenge)")
    print("   3. Moderate (100% - Both might work)")
    
    noise_choice = input("\nEnter choice (1-3): ").strip()
    
    if noise_choice == "1":
        max_scale = 3.0  # 200%
        noise_label = "EXTREME (200%)"
    elif noise_choice == "2":
        max_scale = 3.5  # 250%
        noise_label = "ULTRA EXTREME (250%)"
    else:
        max_scale = 2.0  # 100%
        noise_label = "MODERATE (100%)"
    
    # Encode message
    morse = text_to_morse(text)
    clean = morse_to_signal(morse)
    noisy, noise_pct, seed = obfuscate_signal(clean, max_scale=max_scale)
    
    print("\n" + "="*80)
    print("📡 TRANSMISSION DETAILS")
    print("="*80)
    print(f"Original Message:  {text}")
    print(f"Morse Code:        {morse}")
    print(f"Noise Level:       {noise_label}")
    print(f"Encryption Seed:   {seed}")
    print(f"Signal Length:     {len(noisy)} pulses")
    
    # Decode with both methods
    print("\n" + "="*80)
    print("🔍 DECODING RESULTS")
    print("="*80)
    
    std_morse, std_text = standard_decoder(noisy)
    ai_morse, ai_text = ai_decoder(noisy, model)
    
    print(f"\n📊 Standard Decoder (Rule-Based):")
    print(f"   Morse: {std_morse[:60]}{'...' if len(std_morse) > 60 else ''}")
    print(f"   Text:  {std_text}")
    std_acc = calculate_accuracy(text, std_text)
    print(f"   Accuracy: {std_acc:.1f}%")
    
    print(f"\n🤖 AI Decoder (Neural Network):")
    print(f"   Morse: {ai_morse[:60]}{'...' if len(ai_morse) > 60 else ''}")
    print(f"   Text:  {ai_text}")
    ai_acc = calculate_accuracy(text, ai_text)
    print(f"   Accuracy: {ai_acc:.1f}%")
    
    # Compare results
    print("\n" + "="*80)
    print("🏆 PERFORMANCE COMPARISON")
    print("="*80)
    
    improvement = ai_acc - std_acc
    
    if ai_acc >= 90:
        print(f"✅ AI Decoder: EXCELLENT ({ai_acc:.1f}%)")
    elif ai_acc >= 70:
        print(f"⚠️  AI Decoder: GOOD ({ai_acc:.1f}%)")
    else:
        print(f"❌ AI Decoder: POOR ({ai_acc:.1f}%)")
    
    if std_acc >= 90:
        print(f"✅ Standard Decoder: EXCELLENT ({std_acc:.1f}%)")
    elif std_acc >= 70:
        print(f"⚠️  Standard Decoder: GOOD ({std_acc:.1f}%)")
    else:
        print(f"❌ Standard Decoder: FAILED ({std_acc:.1f}%)")
    
    print(f"\n📈 Improvement: {improvement:+.1f}%")
    
    if improvement > 50:
        print("🎯 AI DECODER VASTLY SUPERIOR - Standard decoder completely failed!")
    elif improvement > 20:
        print("🎯 AI DECODER SIGNIFICANTLY BETTER")
    elif improvement > 0:
        print("🎯 AI decoder slightly better")
    else:
        print("🤔 Both performed similarly (try higher noise level)")
    
    print("\n" + "="*80)
    
    # Show waveforms
    show_signals(clean, noisy)

if __name__ == "__main__":
    run_demo()