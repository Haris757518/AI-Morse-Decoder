# ============================================
# app.py - Flask Backend for AI Morse Decoder
# ============================================

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import statistics
from secure_core import (text_to_morse, morse_to_signal, obfuscate_signal, 
                         MORSE_CODE_DICT_REVERSE)

app = Flask(__name__)
CORS(app)

CLASSES = ["dot", "dash", "short_gap", "letter_gap", "word_gap"]

# Load model globally
class RobustMorseDecoder(torch.nn.Module):
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

model = None

def load_model():
    global model
    if model is None:
        model = RobustMorseDecoder(hidden_size=192)
        checkpoint = torch.load("robust_decoder.pth", map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    return model

def standard_decoder(noisy):
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
        
        morse_chars, words, current_char = [], [], ""
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
            word_text = "".join(MORSE_CODE_DICT_REVERSE.get(m, "?") for m in word.split())
            text_words.append(word_text)
        
        return " / ".join(words), " ".join(text_words)
    except:
        return "", "ERROR"

def ai_decoder(noisy, model):
    if not noisy:
        return "", "ERROR"
    
    X = np.array([[d / 500.0, 1.0 if s == "on" else 0.0] for s, d in noisy], np.float32)
    original_len = len(X)
    if len(X) < 64:
        X = np.vstack([X, np.zeros((64 - len(X), 2), dtype=np.float32)])
    else:
        X = X[:64]
    
    with torch.no_grad():
        preds = model(torch.from_numpy(X).unsqueeze(0))
        labels = preds.argmax(-1).squeeze(0).cpu().numpy()

    morse_chars, words, current_char = [], [], ""
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
        word_text = "".join(MORSE_CODE_DICT_REVERSE.get(m, "?") for m in word.split())
        text_words.append(word_text)
    
    return " / ".join(words), " ".join(text_words)

def calculate_accuracy(original, decoded):
    if not decoded or "ERROR" in decoded:
        return 0.0
    orig = original.upper().replace(" ", "")
    dec = decoded.upper().replace(" ", "")
    if not dec:
        return 0.0
    matches = sum(c1 == c2 for c1, c2 in zip(orig, dec))
    return (matches / max(len(orig), len(dec))) * 100

def find_perfect_seed(text, model, max_scale=3.0, max_attempts=100):
    morse = text_to_morse(text)
    clean = morse_to_signal(morse)
    for _ in range(max_attempts):
        seed = np.random.randint(1000, 9999)
        noisy, _, _ = obfuscate_signal(clean, max_scale=max_scale, seed=seed)
        _, ai_text = ai_decoder(noisy, model)
        ai_acc = calculate_accuracy(text, ai_text)
        if ai_acc >= 99.0:
            _, std_text = standard_decoder(noisy)
            std_acc = calculate_accuracy(text, std_text)
            if std_acc < 50:
                return seed, ai_acc, std_acc
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/decode', methods=['POST'])
def decode():
    data = request.json
    message = data.get('message', 'code red')
    noise_level = data.get('noise_level', 'extreme')
    mode = data.get('mode', 'realistic')
    
    max_scale = 3.0 if noise_level == 'extreme' else 2.0
    model = load_model()
    
    morse = text_to_morse(message)
    clean = morse_to_signal(morse)
    
    if mode == 'best':
        result = find_perfect_seed(message, model, max_scale=max_scale)
        if result:
            seed, _, _ = result
        else:
            seed = np.random.randint(1000, 9999)
    else:
        seed = np.random.randint(1000, 9999)
    
    noisy, _, _ = obfuscate_signal(clean, max_scale=max_scale, seed=seed)
    
    _, std_text = standard_decoder(noisy)
    _, ai_text = ai_decoder(noisy, model)
    
    std_acc = calculate_accuracy(message, std_text)
    ai_acc = calculate_accuracy(message, ai_text)
    
    return jsonify({
        'message': message,
        'morse': morse,
        'noise_level': noise_level,
        'seed': int(seed),
        'mode': mode,
        'standard': {
            'output': std_text,
            'accuracy': round(std_acc, 1)
        },
        'ai': {
            'output': ai_text,
            'accuracy': round(ai_acc, 1)
        },
        'improvement': round(ai_acc - std_acc, 1)
    })

@app.route('/api/batch', methods=['POST'])
def batch():
    data = request.json
    message = data.get('message', 'code red')
    noise_level = data.get('noise_level', 'extreme')
    trials = data.get('trials', 10)
    
    max_scale = 3.0 if noise_level == 'extreme' else 2.0
    model = load_model()
    
    morse = text_to_morse(message)
    clean = morse_to_signal(morse)
    
    results = []
    for _ in range(trials):
        seed = np.random.randint(1000, 9999)
        noisy, _, _ = obfuscate_signal(clean, max_scale=max_scale, seed=seed)
        
        _, std_text = standard_decoder(noisy)
        _, ai_text = ai_decoder(noisy, model)
        
        std_acc = calculate_accuracy(message, std_text)
        ai_acc = calculate_accuracy(message, ai_text)
        
        results.append({
            'seed': int(seed),
            'std_acc': round(std_acc, 1),
            'ai_acc': round(ai_acc, 1),
            'improvement': round(ai_acc - std_acc, 1)
        })
    
    avg_std = sum(r['std_acc'] for r in results) / len(results)
    avg_ai = sum(r['ai_acc'] for r in results) / len(results)
    
    return jsonify({
        'message': message,
        'noise_level': noise_level,
        'trials': trials,
        'results': results,
        'summary': {
            'avg_standard': round(avg_std, 1),
            'avg_ai': round(avg_ai, 1),
            'avg_improvement': round(avg_ai - avg_std, 1),
            'ai_wins': sum(1 for r in results if r['ai_acc'] > r['std_acc'])
        }
    })

if __name__ == '__main__':
    load_model()
    print("🚀 Server running at http://localhost:5000")
    app.run(debug=True, port=5000)


