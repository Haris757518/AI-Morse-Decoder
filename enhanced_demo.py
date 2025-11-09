# final_demo.py - Production-Ready AI Morse Decoder
# Realistic demonstration showing AI superiority under extreme noise

import torch
import numpy as np
import statistics
from secure_core import (text_to_morse, morse_to_signal, obfuscate_signal, 
                         show_signals, MORSE_CODE_DICT_REVERSE)

CLASSES = ["dot", "dash", "short_gap", "letter_gap", "word_gap"]

class RobustMorseDecoder(torch.nn.Module):
    """Deep bidirectional GRU with attention for extreme noise decoding"""
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

def load_model(path="robust_decoder.pth"):
    """Load trained model"""
    model = RobustMorseDecoder(hidden_size=192)
    try:
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"ERROR: Could not load model - {e}")
        print("Please run: python train_robust_decoder.py")
        return None

def standard_decoder(noisy):
    """Traditional rule-based decoder - fails on extreme noise"""
    if not noisy:
        return "", "ERROR"
    
    try:
        on_durations = [d for s, d in noisy if s == "on"]
        off_durations = [d for s, d in noisy if s == "off"]
        
        if not on_durations or not off_durations:
            return "", "ERROR"
        
        on_median = statistics.median(on_durations)
        off_median = statistics.median(off_durations)
        
        dot_dash_threshold = on_median * 1.5
        letter_gap_threshold = off_median * 2.0
        word_gap_threshold = off_median * 4.0
        
        morse_chars = []
        words = []
        current_char = ""
        
        for state, duration in noisy:
            if state == "on":
                current_char += "." if duration < dot_dash_threshold else "-"
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
        
        text_words = []
        for word in words:
            word_text = ""
            for morse_code in word.split():
                char = MORSE_CODE_DICT_REVERSE.get(morse_code, "?")
                word_text += char
            text_words.append(word_text)
        
        return " / ".join(words), " ".join(text_words)
    except:
        return "", "ERROR"

def ai_decoder(noisy, model):
    """AI-powered decoder using deep learning"""
    if not noisy:
        return "", "ERROR"
    
    X = np.array([[d / 500.0, 1.0 if s == "on" else 0.0] for s, d in noisy], np.float32)
    original_len = len(X)
    
    if len(X) < 64:
        padding = np.zeros((64 - len(X), 2), dtype=np.float32)
        X = np.vstack([X, padding])
    else:
        X = X[:64]
    
    X = torch.from_numpy(X).unsqueeze(0)

    with torch.no_grad():
        preds = model(X)
        labels = preds.argmax(-1).squeeze(0).cpu().numpy()

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
    
    text_words = []
    for word in words:
        word_text = ""
        for morse_code in word.split():
            char = MORSE_CODE_DICT_REVERSE.get(morse_code, "?")
            word_text += char
        text_words.append(word_text)
    
    return " / ".join(words), " ".join(text_words)

def calculate_accuracy(original, decoded):
    """Calculate character-level accuracy"""
    if not decoded or "ERROR" in decoded:
        return 0.0
    orig = original.upper().replace(" ", "")
    dec = decoded.upper().replace(" ", "")
    if not dec:
        return 0.0
    matches = sum(c1 == c2 for c1, c2 in zip(orig, dec))
    return (matches / max(len(orig), len(dec))) * 100

def run_production_demo():
    """Production demo - realistic performance without seed manipulation"""
    
    print("\n" + "="*70)
    print("  AI-POWERED MORSE CODE DECODER")
    print("  Deep Learning vs Traditional Rule-Based Decoding")
    print("="*70)
    
    model = load_model()
    if model is None:
        return
    
    print("\nTest Messages:")
    tests = ["code red", "sos help", "alpha bravo", "secure link"]
    for i, msg in enumerate(tests, 1):
        print(f"  {i}. {msg}")
    print("  5. Custom message")
    
    choice = input("\nSelect (1-5): ").strip()
    
    if choice == "5":
        text = input("Enter message: ").strip() or "test"
    elif choice in ["1", "2", "3", "4"]:
        text = tests[int(choice) - 1]
    else:
        text = "code red"
    
    print("\nNoise Level:")
    print("  1. Extreme (200%)")
    print("  2. Moderate (100%)")
    
    noise_choice = input("Select (1-2): ").strip()
    max_scale = 3.0 if noise_choice == "1" else 2.0
    noise_label = "Extreme (200%)" if noise_choice == "1" else "Moderate (100%)"
    
    # Encode with RANDOM seed (realistic scenario)
    morse = text_to_morse(text)
    clean = morse_to_signal(morse)
    noisy, _, seed = obfuscate_signal(clean, max_scale=max_scale)
    
    print("\n" + "="*70)
    print("TRANSMISSION")
    print("="*70)
    print(f"Message: {text}")
    print(f"Noise:   {noise_label}")
    print(f"Seed:    {seed} (random)")
    
    # Decode with both methods
    print("\n" + "="*70)
    print("DECODING")
    print("="*70)
    
    std_morse, std_text = standard_decoder(noisy)
    ai_morse, ai_text = ai_decoder(noisy, model)
    
    std_acc = calculate_accuracy(text, std_text)
    ai_acc = calculate_accuracy(text, ai_text)
    
    print(f"\nStandard Decoder: {std_text}")
    print(f"Accuracy:         {std_acc:.1f}%")
    
    print(f"\nAI Decoder:       {ai_text}")
    print(f"Accuracy:         {ai_acc:.1f}%")
    
    improvement = ai_acc - std_acc
    print(f"\nImprovement:      +{improvement:.1f}%")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    if ai_acc > std_acc + 30:
        print("✓ AI decoder significantly outperforms standard decoder")
    elif ai_acc > std_acc:
        print("✓ AI decoder performs better than standard decoder")
    else:
        print("• Both decoders struggled with this noise level")
    
    if ai_acc < 90:
        print("\nNote: This is a work-in-progress model trained on limited data.")
        print("      Performance improves with:")
        print("      - More training data (currently 5K samples)")
        print("      - Longer training duration (currently 50 epochs)")
        print("      - Larger sequence lengths (currently 64 pulses)")
        print("      - Optimized for short messages (1-3 words)")
    
    print("\n" + "="*70)
    
    show = input("\nShow waveforms? (y/n): ").strip().lower()
    if show == 'y':
        show_signals(clean, noisy)

def run_batch_evaluation():
    """Evaluate on multiple random seeds to show average performance"""
    
    print("\n" + "="*70)
    print("  BATCH EVALUATION - Multiple Random Seeds")
    print("="*70)
    
    model = load_model()
    if model is None:
        return
    
    test_message = input("\nEnter test message (default: code red): ").strip() or "code red"
    
    print("\nNoise Level:")
    print("  1. Extreme (200%)")
    print("  2. Moderate (100%)")
    noise_choice = input("Select (1-2): ").strip()
    max_scale = 3.0 if noise_choice == "1" else 2.0
    noise_label = "Extreme (200%)" if noise_choice == "1" else "Moderate (100%)"
    
    num_trials = int(input("\nNumber of trials (default: 10): ").strip() or "10")
    
    print(f"\n" + "="*70)
    print(f"Running {num_trials} trials with random seeds...")
    print("="*70)
    
    morse = text_to_morse(test_message)
    clean = morse_to_signal(morse)
    
    ai_accuracies = []
    std_accuracies = []
    
    for trial in range(num_trials):
        noisy, _, seed = obfuscate_signal(clean, max_scale=max_scale)
        
        _, std_text = standard_decoder(noisy)
        _, ai_text = ai_decoder(noisy, model)
        
        std_acc = calculate_accuracy(test_message, std_text)
        ai_acc = calculate_accuracy(test_message, ai_text)
        
        ai_accuracies.append(ai_acc)
        std_accuracies.append(std_acc)
        
        print(f"Trial {trial+1:2d}: Seed={seed} | Std={std_acc:5.1f}% | AI={ai_acc:5.1f}% | Δ={ai_acc-std_acc:+5.1f}%")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    avg_ai = sum(ai_accuracies) / len(ai_accuracies)
    avg_std = sum(std_accuracies) / len(std_accuracies)
    
    print(f"\nMessage:               {test_message}")
    print(f"Noise Level:           {noise_label}")
    print(f"Trials:                {num_trials}")
    
    print(f"\nStandard Decoder Avg:  {avg_std:.1f}%")
    print(f"AI Decoder Avg:        {avg_ai:.1f}%")
    print(f"Average Improvement:   +{avg_ai - avg_std:.1f}%")
    
    wins = sum(1 for ai, std in zip(ai_accuracies, std_accuracies) if ai > std)
    print(f"\nAI Wins:               {wins}/{num_trials} ({wins/num_trials*100:.0f}%)")
    
    if avg_ai > avg_std + 20:
        print("\n✓ AI decoder consistently outperforms standard decoder")
    elif avg_ai > avg_std:
        print("\n✓ AI decoder shows improvement over standard decoder")
    else:
        print("\n• Performance varies - model needs more training")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MORSE CODE DECODER DEMONSTRATION")
    print("="*70)
    print("\nSelect mode:")
    print("  1. Single Test (interactive)")
    print("  2. Batch Evaluation (statistical analysis)")
    
    mode = input("\nChoice (1-2): ").strip()
    
    if mode == "2":
        run_batch_evaluation()
    else:
        run_production_demo()