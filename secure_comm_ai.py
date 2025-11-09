# secure_comm_ai.py
import torch, numpy as np, pickle
from secure_core import text_to_morse, morse_to_signal, obfuscate_signal, show_signals, MORSE_CODE_DICT_REVERSE

CLASSES = ["dot", "dash", "short_gap", "letter_gap", "word_gap"]

# ------------------ Model ------------------
# ------------------ Model ------------------
class SeqClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(2, 96, batch_first=True, bidirectional=True)  # match training
        self.fc = torch.nn.Linear(96 * 2, 5)  # output size = 5 classes
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


def load_model(path="pulse_seq.pth"):
    model = SeqClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print("[MODEL] Loaded pulse_seq.pth (context-aware)")
    return model


# ------------------ Decoder ------------------
def ai_decode(noisy, model):
    # Convert noisy signal into model input format
    X = np.array([[d / 300.0, 1 if s == "on" else 0] for s, d in noisy], np.float32)
    X = torch.from_numpy(X).unsqueeze(0)  # Add batch dimension => (1, T, 2)

    with torch.no_grad():
        preds = model(X)
        preds = preds.squeeze(0)  # remove batch dim -> (T, 5)
        labels = preds.argmax(1).cpu().numpy()

    morse, cur = "", ""
    for lbl in labels:
        c = CLASSES[lbl]
        if c == "dot": cur += "."
        elif c == "dash": cur += "-"
        elif c == "letter_gap":
            morse += cur + " "
            cur = ""
        elif c == "word_gap":
            morse += cur + " / "
            cur = ""
    if cur:
        morse += cur

    # Convert Morse → Text
    text = ""
    for word in morse.split(" / "):
        for code in word.strip().split():
            text += MORSE_CODE_DICT_REVERSE.get(code, "?")
        text += " "
    return morse.strip(), text.strip()


# ------------------ Main ------------------
if __name__ == "__main__":
    print("==== AI Secure Communication System – Symbol-Aware ====\n")
    model = load_model()

    text = input("Enter message to encrypt: ").strip()
    morse = text_to_morse(text)
    clean = morse_to_signal(morse)
    noisy, noise, seed = obfuscate_signal(clean, max_scale=2.0)
    print(f"[ENCRYPTION] (AI-Learnable) Seed={seed}  Noise≈±{noise}%")

    morse_ai, text_ai = ai_decode(noisy, model)

    print(f"[AI DECODER] Morse reconstructed: {morse_ai}")
    print(f"[AI DECODER OUTPUT] => {text_ai}")

    # Show waveforms
    show_signals(clean, noisy)
    print("\n✅ FINAL DECODED MESSAGE:")
    print(text_ai)
