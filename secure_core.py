# secure_core.py
import random, math, pickle, statistics, os
import matplotlib.pyplot as plt
import numpy as np

# Morse tables
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', '0': '-----'
}
MORSE_CODE_DICT_REVERSE = {v: k for k, v in MORSE_CODE_DICT.items()}
SECRET_KEY = 7

# --- Converters ---
def text_to_morse(text: str) -> str:
    words = []
    for word in text.split(' '):
        chars = ' '.join(MORSE_CODE_DICT.get(c.upper(), '') for c in word if c.upper() in MORSE_CODE_DICT)
        words.append(chars)
    return ' / '.join(words)

def morse_to_signal(morse_code: str, base_unit: int = 100):
    signal = []
    words = morse_code.split(' / ')
    for wi, word in enumerate(words):
        letters = word.split(' ')
        for li, letter in enumerate(letters):
            for ci, char in enumerate(letter):
                dur = base_unit if char == '.' else 3 * base_unit
                signal.append(('on', dur))
                if ci < len(letter) - 1:
                    signal.append(('off', base_unit))
            if li < len(letters) - 1:
                signal.append(('off', 3 * base_unit))
            elif wi < len(words) - 1:
                signal.append(('off', 7 * base_unit))
    return signal

def obfuscate_signal(signal, seed=None, max_scale=2.0):
    noisy = []
    if seed is None:
        seed = random.randint(1000, 9999)
    random.seed(seed)
    for i, (s, d) in enumerate(signal):
        k = ((SECRET_KEY * (i + 1)) % 97) + (seed % 13)
        r = random.random()
        scale = 1.0 + (r ** 1.5) * (max_scale - 1.0)
        jitter = int(math.floor((math.sin(i + seed/100.0) + random.gauss(0, 0.9)) * (abs(k) % 7)))
        if s == 'on':
            nd = int(max(15, (d * scale) + (0.8 * abs(k)) + jitter))
        else:
            nd = int(max(15, (d * scale * 1.05) + (0.5 * abs(k)) + jitter))
        noisy.append((s, nd))
    noise_percent = int(max(0, (max_scale - 1.0) * 100))
    return noisy, noise_percent, seed

# --- Plotting ---
def plot_signal_to_axes(ax, signal, title, color='blue'):
    times, vals = [], []
    t = 0
    for s, d in signal:
        times.append(t)
        vals.append(1 if s == 'on' else 0)
        t += d
        times.append(t)
        vals.append(1 if s == 'on' else 0)
    ax.plot(times, vals, drawstyle='steps-post', color=color)
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

def show_signals(clean, noisy, refined=None):
    nrows = 3 if refined is None else 4
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3*nrows))
    plot_signal_to_axes(axes[0], clean, "1) Clean Original Morse", 'green')
    plot_signal_to_axes(axes[1], noisy, "2) Encrypted (Noisy)", 'red')
    plot_signal_to_axes(axes[2], noisy, "3) Received Distorted", 'orange')
    if refined:
        plot_signal_to_axes(axes[3], refined, "4) AI-Decoded Refined Signal", 'blue')
    plt.tight_layout()
    plt.show()
