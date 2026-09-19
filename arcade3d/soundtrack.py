"""Synthesize the arcade soundtrack from the render's event timeline.

Music: Korobeiniki (the Tetris melody, a public-domain 1861 folk song), arranged here
as chiptune: pulse lead, triangle bass, noise drums, plus a buzzing "fly" voice that
doubles the lead an octave down (a Drosophila wing beat is about 200 Hz). The tempo
climbs while the field thins, cuts when the runner-up dies, and resolves to A major
when the winner is crowned.

Effects follow timeline.json (python capture.py timeline --n N): a swarm buzz that
tracks how many flies are in the air, the opening fly's take-off, line-clear chimes,
a riser into the winner, and a crown sparkle.

  python soundtrack.py timeline.json soundtrack.wav
"""
import json
import sys

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
RNG = np.random.default_rng(2026)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Mix:
    def __init__(self, seconds):
        self.n = int(seconds * SR)
        self.buf = {}

    def add(self, stem, t0, mono, pan=0.0):
        if stem not in self.buf:
            self.buf[stem] = np.zeros((2, self.n))
        i = int(round(t0 * SR))
        if i >= self.n or i + len(mono) <= 0:
            return
        j0, j1 = max(0, -i), min(len(mono), self.n - i)
        left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        seg = mono[j0:j1]
        self.buf[stem][0, i + j0:i + j1] += seg * left
        self.buf[stem][1, i + j0:i + j1] += seg * right


# ---------- oscillators (band-limited, additive) ----------
def harmonics(f):
    return max(1, int(SR / 2 / (max(f, 20) * 1.03)))


def pulse(ph, f, duty=0.25):
    out = np.zeros_like(ph)
    for k in range(1, harmonics(f) + 1):
        out += (2 / (np.pi * k)) * np.sin(np.pi * k * duty) * np.cos(2 * np.pi * k * ph)
    return out


def triangle(ph, f):
    out = np.zeros_like(ph)
    for k in range(1, harmonics(f) + 1, 2):
        out += (8 / np.pi ** 2) * (-1) ** ((k - 1) // 2) * np.sin(2 * np.pi * k * ph) / k ** 2
    return out


def buzz(ph, f, n, bright=1.0, flutter=0.35):
    """Wing-beat buzz: a sawtooth with a nasal 1.5 to 3 kHz formant and some amplitude flutter."""
    out = np.zeros_like(ph)
    for k in range(1, min(harmonics(f), 40) + 1):
        fk = k * f
        w = (1 / k) * (1 + 2.2 * bright * np.exp(-((fk - 2200) / 1100) ** 2))
        out += w * np.sin(2 * np.pi * k * ph + k * 0.7)
    wobble = sosfilt(butter(2, 25, fs=SR, output="sos"), RNG.standard_normal(n))
    wobble /= np.abs(wobble).max() + 1e-9
    return out * (1 + flutter * wobble) * 0.5


def env(n, dur, a=0.005, d=0.08, s=0.6, r=0.06):
    t = np.arange(n) / SR
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = np.clip(1 - (t - dur) / r, 0, 1)
    return e * np.where(t > dur, rel, 1.0)


def note(mix, stem, t0, dur, m, kind, gain, pan=0.0, vib=0.0, adsr=None, bright=1.0, flutter=0.35):
    f = midi(m)
    rel = (adsr or {}).get("r", 0.06)
    n = int((dur + rel + 0.01) * SR)
    t = np.arange(n) / SR
    fr = f * (1 + vib * np.sin(2 * np.pi * 5.5 * t) * smooth(0.1, 0.25, t))
    ph = np.cumsum(fr) / SR
    wave = {"pulse": lambda: pulse(ph, f), "pulse50": lambda: pulse(ph, f, 0.5),
            "tri": lambda: triangle(ph, f), "buzz": lambda: buzz(ph, f, n, bright, flutter)}[kind]()
    mix.add(stem, t0, wave * env(n, dur, **(adsr or {})) * gain, pan)


def noise_hit(n, hp=None, lp=None):
    x = RNG.standard_normal(n)
    if hp:
        x = sosfilt(butter(2, hp, "highpass", fs=SR, output="sos"), x)
    if lp:
        x = sosfilt(butter(2, lp, fs=SR, output="sos"), x)
    return x


def kick(mix, t0, gain):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.035)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    x[:200] += np.linspace(0.6, 0, 200)
    mix.add("drums", t0, x * gain)


def snare(mix, t0, gain):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    x = noise_hit(n, hp=1200, lp=7000) * np.exp(-t / 0.06) * 0.5 + np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05) * 0.7
    mix.add("drums", t0, x * gain)


def hat(mix, t0, gain, open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    x = noise_hit(n, hp=7000) * np.exp(-t / (0.08 if open_ else 0.018))
    mix.add("drums", t0, x * gain, pan=0.25)


def crash(mix, t0, gain, length=2.5):
    n = int(length * SR)
    t = np.arange(n) / SR
    x = noise_hit(n, hp=4000, lp=9000) * np.exp(-t / 0.35)
    mix.add("fx", t0, x * gain, pan=-0.2)
    mix.add("fx", t0 + 0.011, x * gain, pan=0.3)


# ---------- the tune ----------
E5, B4, C5, D5, A4, F5, A5, G5, GS4, GS5 = 76, 71, 72, 74, 69, 77, 81, 79, 68, 80
SECTION_A = [
    (0, 1, E5), (1, .5, B4), (1.5, .5, C5), (2, 1, D5), (3, .5, C5), (3.5, .5, B4),
    (4, 1, A4), (5, .5, A4), (5.5, .5, C5), (6, 1, E5), (7, .5, D5), (7.5, .5, C5),
    (8, 1.5, B4), (9.5, .5, C5), (10, 1, D5), (11, 1, E5),
    (12, 1, C5), (13, 1, A4), (14, 1, A4),
    (16.5, 1, D5), (17.5, .5, F5), (18, 1, A5), (19, .5, G5), (19.5, .5, F5),
    (20, 1.5, E5), (21.5, .5, C5), (22, 1, E5), (23, .5, D5), (23.5, .5, C5),
    (24, 1, B4), (25, .5, B4), (25.5, .5, C5), (26, 1, D5), (27, 1, E5),
    (28, 1, C5), (29, 1, A4), (30, 1, A4),
]
BASS_A = [40, 45, 44, 45, 38, 36, 40, 45]
SECTION_B = [
    (0, 2, E5), (2, 2, C5), (4, 2, D5), (6, 2, B4), (8, 2, C5), (10, 2, A4), (12, 2, GS4), (14, 2, B4),
    (16, 2, E5), (18, 2, C5), (20, 2, D5), (22, 2, B4), (24, 1, C5), (25, 1, E5), (26, 2, A5), (28, 4, GS5),
]
HARM_B = [(0, 2, 72), (2, 2, 69), (4, 2, 71), (6, 2, 68), (8, 2, 69), (10, 2, 64), (12, 2, 64), (14, 2, 68),
          (16, 2, 72), (18, 2, 69), (20, 2, 71), (22, 2, 68), (24, 1, 69), (25, 1, 72), (26, 2, 76), (28, 4, 76)]
BASS_B = [45, 44, 45, 40, 45, 44, 45, 40]


def bars_of(section, length=8):
    bars = [[] for _ in range(length)]
    for b, d, m in section:
        bars[int(b // 4)].append((b % 4, d, m))
    return bars


A_BARS, B_BARS, HB_BARS = bars_of(SECTION_A), bars_of(SECTION_B), bars_of(HARM_B)
SONG = [(A_BARS[i], None, BASS_A[i]) for i in range(8)] + [(B_BARS[i], HB_BARS[i], BASS_B[i]) for i in range(8)]
CADENCE = [(A_BARS[6], None, 40), (A_BARS[7], None, 45)]  # "B B C D E | C A A", the tune's own ending


def tempo_map(T, chord_t):
    """One tempo curve for the whole piece, nudged so the final chord lands on a downbeat at chord_t."""
    t = np.arange(0, T["end"] + 0.001, 0.001)
    bpm = np.interp(t, [0, T["closeEnd"], T["wide"], T["massEnd"], T["runner"], T["arrive"], T["end"]],
                    [150, 154, 166, 180, 184, 168, 138])
    beats = np.concatenate([[0], np.cumsum(bpm[:-1] / 60 * 0.001)])
    b = np.interp(chord_t, t, beats)
    beats *= np.ceil(b / 4) * 4 / b
    return (lambda x: float(np.interp(x, beats, t))), (lambda x: float(np.interp(x, t, beats)))


def music(mix, T):
    """The whole piece as one continuous song: it never stops, it resolves."""
    chord_t = T["end"] - 2.3
    at, beat_of = tempo_map(T, chord_t)
    last = int(round(beat_of(chord_t) / 4))  # bar where the final chord lands
    runner_bar = int(beat_of(T["runner"]) // 4)
    drums_from = T["closeEnd"] - 0.05
    for j in range(last):
        lead, harm, root = CADENCE[j - (last - 2)] if j >= last - 2 else SONG[j % len(SONG)]
        b0 = 4 * j
        building = runner_bar <= j < last - 2
        for b, d, m in lead:
            t0, t1 = at(b0 + b), at(b0 + b + d)
            note(mix, "lead", t0, (t1 - t0) * 0.97, m, "pulse", 0.28, vib=0.004,
                 adsr=dict(a=0.006, d=0.15, s=0.6, r=0.08))
            # Fly voice: the melody an octave down, a soft steady buzz under the lead.
            note(mix, "flyvoice", t0, (t1 - t0) * 0.99, m - 12, "buzz", 0.03, pan=0.3, vib=0.01,
                 adsr=dict(a=0.04, d=0.25, s=0.85, r=0.12), bright=0.5, flutter=0.08)
        for b, d, m in harm or []:
            t0, t1 = at(b0 + b), at(b0 + b + d)
            note(mix, "lead", t0, (t1 - t0) * 0.97, m, "pulse50", 0.07, pan=-0.35,
                 adsr=dict(a=0.01, d=0.3, s=0.5, r=0.1))
        for e in range(8):
            t0, t1 = at(b0 + e * 0.5), at(b0 + e * 0.5 + 0.5)
            note(mix, "bass", t0, (t1 - t0) * 0.9, root + (12 if e % 2 else 0), "tri", 0.17,
                 adsr=dict(a=0.003, d=0.1, s=0.8, r=0.03))
        if building:
            # The swoop to the winner: a snare roll that builds instead of a hard stop.
            span = max(1, last - 2 - runner_bar)
            for q in range(16):
                snare(mix, at(b0 + q * 0.25), 0.04 + 0.12 * ((j - runner_bar) + q / 16) / span)
            continue
        for q in range(8):
            t0 = at(b0 + q * 0.5)
            if t0 >= drums_from:
                lvl = 0.55 + 0.45 * smooth(T["closeEnd"], T["massEnd"], t0)
                if q in (0, 4) or (q == 5 and t0 > T["wide"]):
                    kick(mix, t0, 0.55 * lvl)
                if q in (2, 6):
                    snare(mix, t0, 0.3 * lvl)
                hat(mix, t0, 0.04 * lvl)
            elif t0 >= 1.8 and q % 2 == 0:
                hat(mix, t0, 0.03)
    for k, q in enumerate((3.0, 3.25, 3.5, 3.75)):  # fill into the final chord
        snare(mix, at(4 * (last - 1) + q), 0.14 + 0.06 * k)
    tc = at(4 * last)
    for m in (57, 61, 64, 69, 73, 76):  # A major: A3 C#4 E4 A4 C#5 E5
        note(mix, "lead", tc, 2.6, m, "pulse50" if m > 64 else "pulse", 0.08, pan=RNG.uniform(-0.4, 0.4),
             vib=0.005, adsr=dict(a=0.02, d=1.2, s=0.45, r=1.0))
    note(mix, "bass", tc, 2.6, 33, "tri", 0.26, adsr=dict(a=0.01, d=1.5, s=0.5, r=0.8))
    note(mix, "flyvoice", tc, 2.4, 57, "buzz", 0.03, pan=0.3, vib=0.012,
         adsr=dict(a=0.08, d=1.0, s=0.6, r=1.0), bright=0.5, flutter=0.08)
    kick(mix, tc, 0.6)
    crash(mix, tc, 0.16, 1.6)
    return tc


# ---------- effects ----------
def swarm(mix, T, deaths, end):
    """Many buzzing voices; loudness follows how many flies are in the air."""
    grid = np.arange(0, end, 1 / 200)
    deaths = np.sort(np.asarray(deaths))
    airborne = np.searchsorted(deaths, grid - 0.3) - np.searchsorted(deaths, grid - 2.6)
    level = np.sqrt(airborne / max(airborne.max(), 1))
    level = np.convolve(level, np.hanning(260) / np.hanning(260).sum(), mode="same")
    lvl = np.interp(np.arange(mix.n) / SR, grid, level)
    t = np.arange(mix.n) / SR
    active = lvl > 1e-3
    if not active.any():
        return lambda x: 0.0
    i0, i1 = np.argmax(active), len(active) - np.argmax(active[::-1])
    seg_t = t[i0:i1]
    for v in range(36):
        f0 = RNG.uniform(175, 265)
        wob = 1 + 0.025 * np.sin(2 * np.pi * RNG.uniform(0.2, 0.9) * seg_t + RNG.uniform(0, 6.3)) \
            + 0.01 * np.sin(2 * np.pi * RNG.uniform(3, 7) * seg_t)
        ph = np.cumsum(f0 * wob) / SR
        x = buzz(ph, f0, len(seg_t), bright=0.35, flutter=0.12)
        swell = 0.6 + 0.4 * np.sin(2 * np.pi * RNG.uniform(0.1, 0.5) * seg_t + RNG.uniform(0, 6.3))
        mix.add("swarm", seg_t[0], x * swell * lvl[i0:i1] * 0.021, pan=RNG.uniform(-0.9, 0.9))
    return lambda x: float(np.interp(x, grid, level))


def takeoff(mix, t0, gain, pan_to=0.6, dur=2.4, f0=205):
    n = int(dur * SR)
    t = np.arange(n) / SR
    # Revs up at lift-off, then Doppler-drops and fades as it flies past the camera.
    f = f0 * (1 + 0.18 * smooth(0, 0.25, t) - 0.12 * smooth(0.6, 1.6, t))
    ph = np.cumsum(f) / SR
    x = buzz(ph, f0, n, bright=0.6, flutter=0.15)
    amp = smooth(0, 0.3, t) * (1 - smooth(0.9, dur, t)) * (1 + 0.3 * np.exp(-((t - 0.7) / 0.4) ** 2))
    for k, p in enumerate(np.linspace(0, pan_to, 8)):
        a, b = k * n // 8, (k + 1) * n // 8
        mix.add("fx", t0 + a / SR, x[a:b] * amp[a:b] * gain, pan=p)


def game_over(mix, t0):
    for k, m in enumerate([76, 72, 69, 64, 57]):
        note(mix, "fx", t0 + k * 0.07, 0.09, m, "pulse", 0.1, pan=-0.1)
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    mix.add("fx", t0, np.sin(2 * np.pi * np.cumsum(90 * np.exp(-t / 0.2)) / SR) * np.exp(-t / 0.15) * 0.35)


def chime(mix, t0, root, gain=0.09):
    for k, m in enumerate([root, root + 4, root + 7, root + 12, root + 16]):
        note(mix, "fx", t0 + k * 0.045, 0.16, m, "pulse50", gain, pan=0.2 * (k - 2),
             adsr=dict(a=0.002, d=0.12, s=0.3, r=0.15))


def click(mix, t0, kind, gain):
    """Arcade button clack (kind 0-2) or the softer joystick microswitch (kind 3)."""
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    band = (2500, 5000) if kind < 3 else (1200, 2600)
    x = sosfilt(butter(2, band, "bandpass", fs=SR, output="sos"), RNG.standard_normal(n)) * np.exp(-t / 0.006)
    if kind < 3:
        x = x + np.sin(2 * np.pi * 160 * t) * np.exp(-t / 0.012) * 0.5
    mix.add("fx", t0, x * gain * (1.0 if kind < 3 else 0.6), pan=0.25 if kind < 3 else -0.25)


def riser(mix, t0, t1):
    n = int((t1 - t0) * SR)
    t = np.arange(n) / SR
    k = t / t[-1]
    x = RNG.standard_normal(n)
    out = np.zeros(n)
    blocks = 24
    for b in range(blocks):
        a, e = b * n // blocks, (b + 1) * n // blocks
        fc = 300 * (14 ** (b / blocks))
        out[a:e] = sosfilt(butter(2, [fc * 0.8, fc * 1.25], "bandpass", fs=SR, output="sos"), x[a:e])
    tone = np.sin(2 * np.pi * np.cumsum(110 * 2 ** (2 * k)) / SR) * 0.3
    mix.add("fx", t0, (out * 0.9 + tone) * k ** 2 * 0.25)


def sparkle(mix, t0):
    for i in range(22):
        m = int(RNG.choice([81, 85, 88, 93, 97, 100]))
        note(mix, "fx", t0 + i * 0.035 + RNG.uniform(0, 0.02), 0.05, m, "pulse50", 0.05,
             pan=RNG.uniform(-0.8, 0.8), adsr=dict(a=0.001, d=0.15, s=0.0, r=0.4))


def ambience(mix, T, end):
    n = mix.n
    room = sosfilt(butter(2, 250, fs=SR, output="sos"), RNG.standard_normal(n))
    room /= np.abs(room).max()
    t = np.arange(n) / SR
    mix.add("amb", 0, room * 0.05 * (1 - 0.6 * smooth(T["massEnd"], T["runner"], t)))
    # Distant cabinets chirping while the hall is full.
    for _ in range(90):
        t0 = RNG.uniform(0, T["massEnd"])
        busy = 1 - smooth(T["wide"], T["massEnd"], t0)
        note(mix, "amb", t0, RNG.uniform(0.03, 0.08), int(RNG.integers(72, 96)), "pulse", 0.02 * busy,
             pan=RNG.uniform(-1, 1), adsr=dict(a=0.001, d=0.05, s=0.2, r=0.05))


def reverb(x, seconds=1.4, wet=0.18):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = RNG.standard_normal((2, n)) * np.exp(-t / (seconds / 5))
    ir[:, : int(0.012 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    out = np.stack([fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
    return x + wet * out


def main():
    tl = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "timeline.json"))
    out_path = sys.argv[2] if len(sys.argv) > 2 else "soundtrack.wav"
    T = tl["T"]
    end = T["end"]
    mix = Mix(end + 0.05)

    swarm_level = swarm(mix, T, [d for d in tl["deaths"] if d < T["runner"] + 3], end)
    music(mix, T)
    ambience(mix, T, end)
    takeoff(mix, tl["hero"] + 0.25, 0.07, pan_to=0.2)
    game_over(mix, tl["hero"])
    takeoff(mix, tl["runner"] + 0.25, 0.035, pan_to=-0.5, f0=225)
    game_over(mix, tl["runner"])
    for c in tl["heroClears"]:
        if c < tl["hero"]:
            chime(mix, c, 88)
    for c in tl["winnerClears"]:
        if c > T["runner"]:
            chime(mix, c, 81, 0.11)
    for t0, kind in tl.get("heroPresses", []):
        click(mix, t0, kind, 0.22 * (1 - smooth(T["closeEnd"] - 0.3, T["closeEnd"] + 1.0, t0)))
    for t0, kind in tl.get("winnerPresses", []):
        click(mix, t0, kind, 0.22 * smooth(T["runner"], T["arrive"], t0))
    sparkle(mix, T["crown"] + 0.35)

    gains = dict(lead=1.0, flyvoice=1.0, bass=0.9, drums=0.9, swarm=1.0, fx=1.0, amb=1.0)
    # The music steps back while thousands of flies are in the air.
    tt = np.arange(mix.n) / SR
    duck = 1 - 0.15 * np.array([swarm_level(x) for x in tt[::480]])
    duck = np.interp(tt, tt[::480], duck)
    for k in ("lead", "bass", "drums"):
        mix.buf[k] *= duck
    # Take the fizz off the buzz layers so they sit under the music.
    for k in ("swarm", "flyvoice"):
        mix.buf[k] = sosfilt(butter(2, 2800, fs=SR, output="sos"), mix.buf[k], axis=1)
    total = sum(mix.buf[k] * gains[k] for k in mix.buf)
    total = reverb(total)
    t = np.arange(total.shape[1]) / SR
    total *= smooth(0, 0.12, t) * (1 - smooth(end - 0.8, end, t))  # no click in, gentle fade out
    rms = np.sqrt((total ** 2).mean())
    total *= 10 ** (-16 / 20) / rms
    total = np.tanh(total * 1.1) / np.tanh(1.1)
    total *= 0.93 / np.abs(total).max()
    pcm = (total.T * 32767).astype(np.int16)
    import wave
    with wave.open(out_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {out_path}: {end:.1f}s, stems {sorted(mix.buf)}")


if __name__ == "__main__":
    main()
