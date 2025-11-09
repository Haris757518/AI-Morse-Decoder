import pickle
import matplotlib.pyplot as plt

# Morse dictionary (reverse)
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9', '-----': '0'
}

SECRET_KEY = 7  # must match the key in encryption.py


# -------------------- DECODER FUNCTIONS --------------------

def plot_signal(signal, title="Received Encrypted Signal"):
    """Plots received signal waveform"""
    times, values = [], []
    t = 0
    for state, dur in signal:
        times.append(t)
        values.append(1 if state == 'on' else 0)
        t += dur
        times.append(t)
        values.append(1 if state == 'on' else 0)
    plt.figure(figsize=(10, 3))
    plt.plot(times, values, drawstyle='steps-post', color='red')
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Signal (1=ON, 0=OFF)")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def morse_to_text(morse_code):
    """Converts Morse code to readable text (preserves spaces)"""
    words = morse_code.split(' / ')
    decoded_text = []
    for word in words:
        decoded_word = ''
        for code in word.split():
            decoded_word += MORSE_CODE_DICT.get(code, '?')
        decoded_text.append(decoded_word)
    return ' '.join(decoded_text)



# -------------------- NORMAL DECODER (fails) --------------------

def normal_decoder(signal, base_unit=100):
    """Standard Morse decoder (fails for noisy signals)"""
    morse = ''
    current_symbol = ''
    for state, dur in signal:
        if state == 'on':
            if dur < 2 * base_unit:
                current_symbol += '.'
            else:
                current_symbol += '-'
        elif state == 'off':
            if dur > 6 * base_unit:
                morse += current_symbol + ' / '
                current_symbol = ''
            elif dur > 2 * base_unit:
                morse += current_symbol + ' '
                current_symbol = ''
    return morse_to_text(morse)


# -------------------- AI-INSPIRED DECODER (succeeds) --------------------

import statistics

def ai_decoder(signal, key=SECRET_KEY):
    """Final adaptive decoder with dynamic gap classification"""
    # Step 1: Remove secret key offset
    adjusted_signal = []
    for state, dur in signal:
        if state == 'on':
            dur -= (key * 3)
        elif state == 'off':
            dur -= key
        adjusted_signal.append((state, max(dur, 1)))  # avoid negatives

    # Step 2: Estimate base timing
    on_durations = [d for s, d in adjusted_signal if s == 'on']
    off_durations = [d for s, d in adjusted_signal if s == 'off' and d > 0]

    if not on_durations:
        print("No signal detected!")
        return ""

    unit = statistics.median(on_durations)

    # Filter OFF durations to exclude intra-symbol gaps (< unit)
    meaningful_offs = [d for d in off_durations if d > unit * 1.1]
    if meaningful_offs:
        avg_off = statistics.median(meaningful_offs)
    else:
        avg_off = unit * 2.5  # fallback

    print(f"[AI DECODER] Estimated base unit (median): {unit:.2f} ms")
    print(f"[AI DECODER] Median OFF gap (filtered): {avg_off:.2f} ms")

    # Step 3: Dynamic thresholds
    letter_gap = 0.8 * avg_off       # relaxed to handle compressed signals
    word_gap = 2.5 * avg_off

    morse = ''
    current_symbol = ''

    for state, dur in adjusted_signal:
        if state == 'on':
            ratio = dur / unit
            # Dot / Dash
            if ratio < 1.9:
                current_symbol += '.'
            else:
                current_symbol += '-'

        elif state == 'off':
            # Word gaps are much longer now — detect them more clearly
            if dur > (2.5 * avg_off):
                morse += current_symbol + ' / '
                current_symbol = ''
            elif dur > (0.6 * avg_off):
                morse += current_symbol + ' '
                current_symbol = ''


            # Ignore short intra-symbol gaps
    if current_symbol:
        morse += current_symbol

    print(f"[AI DECODER] Reconstructed Morse: {morse}")
    text = morse_to_text(morse)
    print(f"[AI DECODER] Decoded Text: {text}")
    return text




# -------------------- MAIN --------------------

if __name__ == "__main__":
    print("---- AI Secure Communication: Decryption ----")

    # Load encrypted signal saved from encryption.py
    try:
        with open("encrypted_signal.dat", "rb") as f:
            data = pickle.load(f)

        # Handle both old format (list) and new format (dict)
        if isinstance(data, dict):
            encrypted_signal = data.get("signal", [])
            original_case = data.get("original_case", "")
        else:
            encrypted_signal = data
            original_case = ""

        print("[LOADED] Encrypted signal file found.")

    except FileNotFoundError:
        print("❌ No saved signal found! Run encryption.py first and save the output.")
        exit()


    # Show received waveform
    plot_signal(data["signal"], "Received Encrypted Signal")


    # Try normal decoder
    normal_output = normal_decoder(encrypted_signal)
    print("\n[STANDARD DECODER OUTPUT] =>", normal_output or "FAILED TO DECODE")

    # Try AI-based decoder
    ai_output = ai_decoder(encrypted_signal)
    print("\n[AI DECODER OUTPUT] =>", ai_output)
