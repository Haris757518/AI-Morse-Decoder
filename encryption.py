import random
import matplotlib.pyplot as plt

# Morse dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', '0': '-----', ' ': ' '
}

# -------------------- ENCRYPTION FUNCTIONS --------------------

SECRET_KEY = 7  # Only you and your receiver know this

def text_to_morse(text):
    """Converts plain text to Morse code string"""
    morse = []
    for char in text.upper():
        morse.append(MORSE_CODE_DICT.get(char, ''))
    return ' '.join(morse)

def morse_to_signal(morse_code, base_unit=100):
    """Converts Morse code to clean (non-obfuscated) signal"""
    signal = []
    for symbol in morse_code:
        if symbol == '.':
            duration = base_unit
        elif symbol == '-':
            duration = 3 * base_unit
        elif symbol == ' ':
    # Make word gap more distinct and random within range
            signal.append(('off', random.randint(1000, 1300)))  # bigger word gap
            continue
        else:
            signal.append(('off', 3 * base_unit))
            continue

        signal.append(('on', duration))
        signal.append(('off', base_unit))
    return signal

def obfuscate_signal(morse_code, base_unit=100):
    """Adds random jitter + secret key-based encryption"""
    signal = []
    jitter_percent = random.randint(20, 60)  # automatic noise level
    print(f"[ENCRYPTION] Random noise level: ±{jitter_percent}%")

    for symbol in morse_code:
        if symbol == '.':
            duration = base_unit
        elif symbol == '-':
            duration = 3 * base_unit
        elif symbol == ' ':
            signal.append(('off', 7 * base_unit))
            continue
        else:
            signal.append(('off', 3 * base_unit))
            continue

        # Random jitter (noise encryption)
        noise = random.uniform(-jitter_percent, jitter_percent)
        duration = int(duration + duration * noise / 100)

        # Secret key encryption (timing shift)
        duration = duration + (SECRET_KEY * 3)

        signal.append(('on', duration))
        signal.append(('off', base_unit + SECRET_KEY))
    return signal, jitter_percent

def plot_comparison(clean_signal, noisy_signal, jitter):
    """Displays both clean and noisy waveforms side by side"""
    def make_waveform(signal):
        times, values = [], []
        t = 0
        for state, dur in signal:
            times.append(t)
            values.append(1 if state == 'on' else 0)
            t += dur
            times.append(t)
            values.append(1 if state == 'on' else 0)
        return times, values

    t1, v1 = make_waveform(clean_signal)
    t2, v2 = make_waveform(noisy_signal)

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t1, v1, drawstyle='steps-post', color='green')
    plt.title("Clean (Original) Signal Waveform")
    plt.ylabel("Signal (1=ON, 0=OFF)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t2, v2, drawstyle='steps-post', color='red')
    plt.title(f"Encrypted Signal (±{jitter}% Noise + Key-Based Shift)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Signal (1=ON, 0=OFF)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def encrypt_message(text):
    """Full encryption pipeline"""
    morse = text_to_morse(text)
    print(f"[TEXT] {text}")
    print(f"[MORSE] {morse}")

    clean_signal = morse_to_signal(morse)
    noisy_signal, jitter = obfuscate_signal(morse)
    print(f"[ENCRYPTED SIGNAL SAMPLE] {noisy_signal[:10]} ...")
    import pickle
    # Save encrypted signal + original case info
    data = {
        "signal": noisy_signal,
        "original_case": text   # store exact input
    }
    with open("encrypted_signal.dat", "wb") as f:
        pickle.dump(data, f)
    print("[SAVED] Encrypted signal and case info saved to encrypted_signal.dat")


    plot_comparison(clean_signal, noisy_signal, jitter)
    return noisy_signal


# -------------------- RUN & TEST --------------------
if __name__ == "__main__":
    msg = input("Enter message to encrypt: ")
    encrypt_message(msg)
