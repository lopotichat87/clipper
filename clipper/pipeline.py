"""Pipeline de clipping : téléchargement -> transcription -> sélection des moments forts -> rendu 9:16.

Tout tourne en local et gratuitement :
- yt-dlp pour télécharger
- faster-whisper pour transcrire (mot par mot)
- Ollama (optionnel) pour choisir les meilleurs moments, sinon un score heuristique
- OpenCV pour suivre le visage, Pillow pour les sous-titres, ffmpeg pour encoder
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "travail"   # vidéos sources, audio, transcriptions (cache)
OUT = ROOT / "clips"      # clips finaux
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

OUT_W, OUT_H = 1080, 1920

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
]

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Sentence:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Clip:
    start: float
    end: float
    title: str
    hook: str
    hashtags: list[str]
    score: float
    text: str = ""
    path: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def fmt_time(t: float) -> str:
    t = max(0, int(t))
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# 1. Récupération de la vidéo
# ---------------------------------------------------------------------------

def _ffmpeg_dir() -> Path:
    """yt-dlp cherche un binaire nommé 'ffmpeg' : on lui fournit un lien vers celui d'imageio."""
    d = WORK / "bin"
    d.mkdir(parents=True, exist_ok=True)
    link = d / "ffmpeg"
    if not link.exists():
        link.symlink_to(FFMPEG)
    return d


def download(url: str, progress: Progress = _noop, max_height: int = 1440) -> tuple[Path, str]:
    """Télécharge la meilleure qualité utile (1440p par défaut : un recadrage 9:16 reste net).
    L'AV1 est évité car son décodage logiciel est très lent."""
    import yt_dlp

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total, "Téléchargement de la vidéo")

    h = int(max_height)
    opts = {
        "format": (f"bv*[height<={h}][vcodec!^=av01]+ba[ext=m4a]/bv*[height<={h}][vcodec!^=av01]+ba"
                   f"/b[height<={h}]/bv*+ba/b"),
        "merge_output_format": "mp4/mkv",
        "outtmpl": str(WORK / "%(id)s" / f"source_{h}p.%(ext)s"),
        "ffmpeg_location": str(_ffmpeg_dir()),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        folder = WORK / info["id"]
        pattern = f"source_{h}p.*"
        done = [f for f in folder.glob(pattern) if f.suffix in (".mp4", ".mkv", ".webm")] if folder.exists() else []
        if not done:
            ydl.download([url])
    files = sorted((f for f in folder.glob(pattern) if f.suffix in (".mp4", ".mkv", ".webm")),
                   key=lambda p: p.suffix != ".mp4")
    if not files:
        raise RuntimeError("Le téléchargement a échoué (aucun fichier produit).")
    return files[0], info.get("title") or info["id"]


def import_local(path: str) -> tuple[Path, str]:
    src = Path(path)
    h = hashlib.sha1(f"{src.name}{src.stat().st_size}".encode()).hexdigest()[:12]
    folder = WORK / f"local_{h}"
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"source{src.suffix.lower() or '.mp4'}"
    if not dst.exists():
        shutil.copy(src, dst)
    return dst, src.stem


# ---------------------------------------------------------------------------
# 2. Transcription
# ---------------------------------------------------------------------------

WHISPER_MODELS = {
    "base": "base",
    "small": "small",
    "medium": "medium",
    "turbo": "large-v3-turbo",   # le plus précis utilisable sur un Mac 8 Go (~1,6 Go à télécharger)
}
PROMPTS = {  # amorce une ponctuation et une orthographe propres dans la bonne langue
    "fr": "Bonjour à tous. Aujourd'hui, on parle de ça : c'est vraiment important, non ?",
    "en": "Hello everyone. Today, we're talking about this: it's really important, isn't it?",
    "es": "Hola a todos. Hoy hablamos de esto: es muy importante, ¿verdad?",
}


def transcribe(video: Path, model_size: str = "small", language: str | None = None,
               progress: Progress = _noop, vocabulary: str = "") -> tuple[list[Word], float, str]:
    """Transcription mot par mot, réglée pour coller au plus près de ce qui est dit."""
    vocab = " ".join(vocabulary.replace(",", " ").split())
    key = hashlib.sha1(f"v2|{model_size}|{language}|{vocab}".encode()).hexdigest()[:10]
    cache = video.parent / f"transcription_{model_size}_{language or 'auto'}_{key}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        return [Word(**w) for w in data["words"]], data["duration"], data["language"]

    # Audio nettoyé : coupe les basses parasites, remonte les voix faibles, mono 16 kHz pour Whisper
    wav = video.parent / "audio_clean.wav"
    if not wav.exists():
        progress(0, "Préparation de l'audio")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-vn",
                        "-af", "highpass=f=70,lowpass=f=7800,afftdn=nf=-25,dynaudnorm=f=250:g=11:p=0.9",
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)

    name = WHISPER_MODELS.get(model_size, model_size)
    progress(0, f"Chargement du modèle de transcription « {model_size} » (téléchargé la 1re fois)")
    from faster_whisper import WhisperModel
    model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)

    prompt = PROMPTS.get(language or "", "")
    if vocab:
        prompt = f"{prompt} {vocab}".strip()
    segments, info = model.transcribe(
        str(wav), language=language or None,
        beam_size=5, best_of=5, patience=1.0,
        temperature=[0.0, 0.2, 0.4, 0.6],          # repli si le décodage part en vrille
        condition_on_previous_text=True,
        initial_prompt=prompt or None,
        hotwords=vocab or None,                      # noms propres, marques, jargon
        word_timestamps=True,
        hallucination_silence_threshold=2.0,         # supprime les phrases inventées pendant les silences
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t and w.probability > 0.05:
                words.append(Word(float(w.start), float(w.end), t))
        progress(min(seg.end / max(info.duration, 1), 1), "Transcription")

    cache.write_text(json.dumps({"words": [asdict(w) for w in words],
                                 "duration": info.duration, "language": info.language}))
    return words, info.duration, info.language


def transcript_cache(video: Path, model_size: str, language: str | None, vocabulary: str = "") -> Path:
    vocab = " ".join(vocabulary.replace(",", " ").split())
    key = hashlib.sha1(f"v2|{model_size}|{language}|{vocab}".encode()).hexdigest()[:10]
    return video.parent / f"transcription_{model_size}_{language or 'auto'}_{key}.json"


def join_words(ws) -> str:
    """Recolle les mots proprement : « c 'est » -> « c'est », « mot ? » garde son espace en français."""
    out = ""
    for w in ws:
        t = w.text if hasattr(w, "text") else str(w)
        if not out:
            out = t
        elif t[:1] in "'’-" or t[:1] in ",.…" or out.endswith(("'", "’", "-")):
            out += t
        else:
            out += " " + t
    return out


def to_sentences(words: list[Word], max_len: float = 15.0) -> list[Sentence]:
    """Regroupe les mots en phrases (ponctuation, pause, ou longueur max)."""
    sents: list[Sentence] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.start - w.end) if nxt else 99
        ends = w.text[-1:] in ".!?…" or gap > 0.8 or (cur[-1].end - cur[0].start) > max_len
        if ends:
            sents.append(Sentence(cur[0].start, cur[-1].end, join_words(cur), cur))
            cur = []
    return sents


# ---------------------------------------------------------------------------
# 3. Sélection des moments forts
# ---------------------------------------------------------------------------

HOOK_WORDS = {
    # français
    "pourquoi", "comment", "secret", "jamais", "personne", "erreur", "vérité", "incroyable", "problème",
    "attention", "imagine", "imaginez", "voilà", "écoute", "écoutez", "honnêtement", "sérieusement",
    "argent", "millions", "milliers", "meilleur", "pire", "astuce", "conseil", "histoire", "choqué",
    "arrêtez", "arrête", "faut", "important", "réalité", "personnellement", "raison",
    # anglais
    "why", "how", "secret", "never", "nobody", "mistake", "truth", "crazy", "insane", "problem",
    "imagine", "listen", "honestly", "money", "million", "best", "worst", "tip", "story", "stop",
}
WEAK_STARTS = {"et", "mais", "donc", "alors", "parce", "puis", "ou", "and", "but", "so", "because", "or",
               "euh", "bah", "ben", "um", "uh"}


def _windows(sents: list[Sentence], dmin: float, dmax: float):
    for i in range(len(sents)):
        for j in range(i, len(sents)):
            d = sents[j].end - sents[i].start
            if d > dmax:
                break
            if d >= dmin:
                yield i, j


def _heuristic_score(sents: list[Sentence], i: int, j: int, dmin: float, dmax: float) -> float:
    words = [w for s in sents[i:j + 1] for w in s.words]
    dur = sents[j].end - sents[i].start
    text = " ".join(s.text for s in sents[i:j + 1]).lower()
    first = sents[i].text.lower()
    tokens = re.findall(r"\w+", text)
    first_tokens = re.findall(r"\w+", first)

    rate = len(words) / max(dur, 1)                       # débit de parole (énergie)
    hooks = sum(t in HOOK_WORDS for t in tokens) / max(len(tokens), 1) * 100
    first_hook = 2.0 * any(t in HOOK_WORDS for t in first_tokens) + 2.0 * ("?" in first)
    punch = text.count("!") + text.count("?")
    numbers = min(len(re.findall(r"\d+", text)), 3)
    weak = 3.0 if first_tokens and first_tokens[0] in WEAK_STARTS else 0
    target = (dmin + dmax) / 2
    length_fit = -abs(dur - target) / target * 2
    silence = sum(max(0, sents[k + 1].start - sents[k].end) for k in range(i, j)) / dur * 10

    return rate * 1.5 + hooks * 1.2 + first_hook + punch * 0.6 + numbers * 0.5 - weak + length_fit - silence


def _pick(cands: list[tuple[float, int, int]], n: int, sents: list[Sentence]) -> list[tuple[float, int, int]]:
    """Garde les meilleurs candidats sans chevauchement."""
    chosen: list[tuple[float, int, int]] = []
    for c in sorted(cands, key=lambda c: -c[0]):
        s, e = sents[c[1]].start, sents[c[2]].end
        if all(e <= sents[o[1]].start or s >= sents[o[2]].end for o in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return sorted(chosen, key=lambda c: c[1])


def _auto_hashtags(text: str, k: int = 4) -> list[str]:
    stop = {"cette", "c'est", "parce", "comme", "alors", "quand", "aussi", "votre", "notre", "leurs",
            "avoir", "faire", "était", "there", "about", "would", "their", "which", "where", "these",
            "vraiment", "quelque", "chose", "toujours", "encore", "après", "avant", "juste", "really"}
    counts: dict[str, int] = {}
    for t in re.findall(r"[a-zàâçéèêëîïôûùüÿœ]{5,}", text.lower()):
        if t not in stop:
            counts[t] = counts.get(t, 0) + 1
    top = sorted(counts, key=lambda t: -counts[t])[:k]
    return [f"#{t}" for t in top] + ["#shorts"]


def _title_from(text: str) -> str:
    first = re.split(r"(?<=[.!?…])\s", text.strip())[0]
    return (first[:80] + "…") if len(first) > 80 else first


def ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def _ollama(model: str, prompt: str) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0.3, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.load(r)["response"])


LLM_PROMPT = """Tu es un monteur expert en contenus courts viraux (TikTok, Reels, Shorts).
Voici un extrait de transcription. Chaque ligne commence par son numéro de phrase et son horodatage.

{lines}

Choisis jusqu'à {k} passages qui feraient d'excellents clips autonomes de {dmin} à {dmax} secondes.
Critères : accroche forte dès la première phrase, idée complète compréhensible sans contexte,
émotion / surprise / conseil concret / punchline, ne commence pas au milieu d'une idée.

Réponds UNIQUEMENT en JSON, dans la langue de la transcription pour les textes :
{{"clips": [{{"debut": <numéro de la première phrase>, "fin": <numéro de la dernière phrase>,
"titre": "<titre accrocheur, max 70 caractères>", "accroche": "<pourquoi ce passage marche>",
"hashtags": ["#...", "#..."], "score": <0 à 100>}}]}}"""


def _llm_candidates(sents: list[Sentence], model: str, dmin: float, dmax: float,
                    progress: Progress) -> list[tuple[float, int, int, dict]]:
    out = []
    chunk_secs = 420  # ~7 min de transcription par requête : tient dans le contexte d'un petit modèle
    chunks, cur = [], []
    for idx, s in enumerate(sents):
        cur.append(idx)
        if s.end - sents[cur[0]].start > chunk_secs:
            chunks.append(cur)
            cur = cur[-8:]  # léger recouvrement pour ne pas couper un bon passage
    if cur:
        chunks.append(cur)

    for ci, chunk in enumerate(chunks):
        progress(ci / len(chunks), f"L'IA analyse la vidéo ({ci + 1}/{len(chunks)})")
        lines = "\n".join(f"[{i}] ({fmt_time(sents[i].start)}) {sents[i].text}" for i in chunk)
        try:
            data = _ollama(model, LLM_PROMPT.format(lines=lines, k=3, dmin=int(dmin), dmax=int(dmax)))
        except Exception as e:  # le modèle a répondu n'importe quoi : on passe au bloc suivant
            print(f"[ollama] bloc {ci} ignoré : {e}")
            continue
        for c in data.get("clips", []) if isinstance(data, dict) else []:
            try:
                i, j = int(c["debut"]), int(c["fin"])
            except Exception:
                continue
            if not (chunk[0] <= i <= chunk[-1]):
                continue
            j = max(i, min(j, len(sents) - 1))
            # Ajuste la durée en ajoutant / retirant des phrases entières
            while j + 1 < len(sents) and sents[j].end - sents[i].start < dmin:
                j += 1
            while j > i and sents[j].end - sents[i].start > dmax:
                j -= 1
            dur = sents[j].end - sents[i].start
            if dur < dmin * 0.7 or dur > dmax * 1.15:
                continue
            try:
                score = float(c.get("score", 50))
            except Exception:
                score = 50.0
            out.append((score, i, j, c))
    return out


def find_highlights(words: list[Word], n: int = 5, dmin: float = 25, dmax: float = 60,
                    llm_model: str | None = None, progress: Progress = _noop) -> list[Clip]:
    sents = to_sentences(words)
    if not sents:
        return []

    meta: dict[tuple[int, int], dict] = {}
    cands: list[tuple[float, int, int]] = []
    if llm_model:
        for score, i, j, c in _llm_candidates(sents, llm_model, dmin, dmax, progress):
            cands.append((score, i, j))
            meta[(i, j)] = c

    if len(cands) < n:  # complète (ou remplace) avec l'heuristique
        progress(0.9, "Calcul des moments forts")
        heur = [(_heuristic_score(sents, i, j, dmin, dmax), i, j) for i, j in _windows(sents, dmin, dmax)]
        if not heur:  # vidéo plus courte que la durée min : on prend tout
            heur = [(0.0, 0, len(sents) - 1)]
        # Score lisible sur 100 = rang du passage parmi tous les passages possibles de la vidéo
        ordered = sorted(s for s, _, _ in heur)
        import bisect
        heur = [(round(100 * bisect.bisect_left(ordered, s) / max(len(ordered) - 1, 1), 1), i, j)
                for s, i, j in heur]
        # les candidats IA passent devant (score décalé)
        cands = [(s + 1000, i, j) for s, i, j in cands] + heur

    clips = []
    for score, i, j in _pick(cands, n, sents):
        text = " ".join(s.text for s in sents[i:j + 1])
        c = meta.get((i, j), {})
        tags = c.get("hashtags") if isinstance(c.get("hashtags"), list) else None
        clips.append(Clip(
            start=max(0, sents[i].start - 0.15),
            end=sents[j].end + 0.35,
            title=str(c.get("titre") or _title_from(text)),
            hook=str(c.get("accroche") or ""),
            hashtags=[t if str(t).startswith("#") else f"#{t}" for t in (tags or _auto_hashtags(text))],
            score=round(score - 1000 if score >= 1000 else score, 1),
            text=text,
        ))
    return clips


# ---------------------------------------------------------------------------
# 4. Recadrage 9:16 avec suivi du visage
# ---------------------------------------------------------------------------

def _probe(video: Path) -> tuple[int, int, float]:
    """Lit largeur, hauteur et fps via ffmpeg (pas de ffprobe dans imageio-ffmpeg)."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr
    m = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})", err)
    if not m:
        raise RuntimeError("Impossible de lire la vidéo.")
    w, h = int(m.group(1)), int(m.group(2))
    rot = re.search(r"rotation of (-?\d+)", err) or re.search(r"rotate\s*:\s*(-?\d+)", err)
    if rot and abs(int(float(rot.group(1)))) % 180 == 90:  # vidéo de téléphone tournée
        w, h = h, w
    f = re.search(r"([\d.]+) fps", err)
    fps = float(f.group(1)) if f else 30.0
    return w, h, fps if 1 < fps < 121 else 30.0


def _face_track(video: Path, start: float, dur: float, src_w: int, src_h: int, crop_w: int,
                sample_fps: float = 4.0) -> np.ndarray:
    """Retourne la position x du centre de la caméra, échantillonnée à `sample_fps`."""
    det_w = 640
    det_h = int(round(src_h * det_w / src_w / 2) * 2)
    cmd = [FFMPEG, "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
           "-vf", f"fps={sample_fps},scale={det_w}:{det_h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8)
    n = len(frames) // (det_w * det_h)
    frames = frames[: n * det_w * det_h].reshape(n, det_h, det_w)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    xs = np.full(n, np.nan)
    for k in range(n):
        faces = cascade.detectMultiScale(frames[k], scaleFactor=1.1, minNeighbors=6,
                                         minSize=(det_h // 28, det_h // 28))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            xs[k] = (x + w / 2) * src_w / det_w

    # Plan large (visage trop petit) : on cherche plutôt une silhouette entière
    if n and np.mean(~np.isnan(xs)) < 0.3:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        for k in range(0, n):
            if not np.isnan(xs[k]):
                continue
            rects, weights = hog.detectMultiScale(frames[k], winStride=(8, 8), padding=(8, 8), scale=1.05)
            if len(rects):
                best = int(np.argmax(np.asarray(weights).ravel()))
                if float(np.asarray(weights).ravel()[best]) > 0.5:
                    x, y, w, h = rects[best]
                    xs[k] = (x + w / 2) * src_w / det_w

    center = src_w / 2
    if n == 0 or np.all(np.isnan(xs)):
        return np.full(max(n, 1), center)

    # Bouche les trous (frames sans visage) puis lisse
    idx = np.arange(n)
    good = ~np.isnan(xs)
    xs = np.interp(idx, idx[good], xs[good])
    xs = np.array([np.median(xs[max(0, k - 2): k + 3]) for k in range(n)])  # retire les faux positifs

    # Caméra « posée » : elle ne bouge que si le visage sort d'une zone morte,
    # et coupe net si le sujet change complètement de place (changement d'intervenant)
    dead = crop_w * 0.12
    cam = np.empty(n)
    cam[0] = xs[0]
    for k in range(1, n):
        diff = xs[k] - cam[k - 1]
        if abs(diff) > crop_w * 0.6:
            cam[k] = xs[k]
        elif abs(diff) > dead:
            cam[k] = cam[k - 1] + diff * 0.35
        else:
            cam[k] = cam[k - 1]
    # Lissage : un mouvement de caméra doux, sauf aux coupes franches (changement d'intervenant)
    cuts = [0] + [k for k in range(1, n) if abs(cam[k] - cam[k - 1]) > crop_w * 0.5] + [n]
    smooth = cam.copy()
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = cam[a:b]
        if len(seg) > 3:
            padded = np.pad(seg, 6, mode="edge")
            smooth[a:b] = np.convolve(padded, kernel / kernel.sum(), mode="same")[6:-6]
    half = crop_w / 2
    return np.clip(smooth, half, src_w - half)




# ---------------------------------------------------------------------------
# 5. Styles, polices et sous-titres animés
# ---------------------------------------------------------------------------

_SUPP = "/System/Library/Fonts/Supplemental/"
FONTS: dict[str, tuple[str, int]] = {
    "Arial Black": (_SUPP + "Arial Black.ttf", 0),
    "Impact": (_SUPP + "Impact.ttf", 0),
    "Montserrat-like (Avenir Heavy)": ("/System/Library/Fonts/Avenir Next.ttc", 8),
    "Avenir Bold": ("/System/Library/Fonts/Avenir Next.ttc", 0),
    "Helvetica Black": ("/System/Library/Fonts/HelveticaNeue.ttc", 9),
    "Futura Bold": (_SUPP + "Futura.ttc", 2),
    "Arial Rounded": (_SUPP + "Arial Rounded Bold.ttf", 0),
    "Arial Bold": (_SUPP + "Arial Bold.ttf", 0),
}


def available_fonts() -> list[str]:
    return [name for name, (path, _) in FONTS.items() if Path(path).exists()]


_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font(name: str, size: int):
    key = (name, size)
    if key not in _font_cache:
        path, idx = FONTS.get(name, FONTS["Arial Black"])
        try:
            _font_cache[key] = ImageFont.truetype(path, size, index=idx)
        except Exception:
            _font_cache[key] = ImageFont.load_default(size)
    return _font_cache[key]


def _rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = (hex_color or "#ffffff").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha
    except ValueError:
        return 255, 255, 255, alpha


@dataclass
class CaptionStyle:
    font: str = "Arial Black"
    size: int = 92                 # taille de base en px (sur 1080 de large)
    position: float = 0.70         # hauteur relative du centre des sous-titres (0 = haut, 1 = bas)
    uppercase: bool = True
    words_per_line: int = 3
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFD600"
    highlight_mode: str = "word"   # word | progressive | box | none
    stroke: bool = True
    stroke_color: str = "#000000"
    shadow: bool = True
    pop: bool = True               # le mot prononcé grossit légèrement

    @classmethod
    def from_dict(cls, d: dict | None) -> "CaptionStyle":
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class RenderOptions:
    aspect: str = "9:16"           # 9:16 | 1:1 | 4:5 | 16:9
    framing: str = "face"          # face | center | fit (vidéo entière sur fond flou)
    captions: bool = True
    style: CaptionStyle = field(default_factory=CaptionStyle)
    title_mode: str = "intro"      # none | intro (3 premières secondes) | always
    progress_bar: bool = False     # fine barre de progression en bas du clip
    quality: str = "high"          # high (net, plus lent) | standard (rapide)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RenderOptions":
        d = dict(d or {})
        style = CaptionStyle.from_dict(d.pop("style", None))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(style=style, **known)

    def out_size(self) -> tuple[int, int]:
        return {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350),
                "16:9": (1920, 1080)}.get(self.aspect, (1080, 1920))


def _blend(frame: np.ndarray, pre: np.ndarray, alpha: np.ndarray, x0: int, y0: int) -> None:
    """Colle une image RGBA pré-multipliée sur la frame BGR (en gérant les débordements)."""
    h, w = pre.shape[:2]
    fh, fw = frame.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, fw), min(y0 + h, fh)
    if xa >= xb or ya >= yb:
        return
    p = pre[ya - y0:yb - y0, xa - x0:xb - x0]
    a = alpha[ya - y0:yb - y0, xa - x0:xb - x0]
    region = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (region * (1 - a) + p).astype(np.uint8)


def _to_premult(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(img).astype(np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    return arr[:, :, [2, 1, 0]] * alpha, alpha


class CaptionRenderer:
    def __init__(self, words: list[Word], clip_start: float, clip_end: float, style: CaptionStyle,
                 out_w: int):
        self.style = style
        self.width = out_w
        self.scale = out_w / 1080 if out_w <= 1080 else 1.0
        ws = [Word(w.start - clip_start, w.end - clip_start, w.text) for w in words
              if w.end > clip_start + 0.05 and w.start < clip_end - 0.05]
        self.chunks = self._group(ws, max(1, int(style.words_per_line)))
        self.cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _group(ws: list[Word], max_words: int) -> list[list[Word]]:
        max_chars = 6 * max_words + 4
        chunks, cur = [], []
        for i, w in enumerate(ws):
            cur.append(w)
            nxt = ws[i + 1] if i + 1 < len(ws) else None
            chars = sum(len(x.text) + 1 for x in cur)
            if (len(cur) >= max_words or chars >= max_chars or w.text[-1:] in ".!?,;:…"
                    or nxt is None or nxt.start - w.end > 0.5):
                chunks.append(cur)
                cur = []
        return chunks

    def _fmt(self, t: str) -> str:
        t = re.sub(r"[.,;:]+$", "", t)
        return t.upper() if self.style.uppercase else t

    def _render(self, ci: int, active: int) -> tuple[np.ndarray, np.ndarray]:
        st = self.style
        words = [self._fmt(w.text) for w in self.chunks[ci]]
        mode = st.highlight_mode
        size = int(st.size * self.scale)
        max_w = self.width - int(90 * self.scale)
        while True:
            font = load_font(st.font, size)
            big = load_font(st.font, int(size * 1.12)) if st.pop and mode != "none" else font
            pad = int(size * 0.16) if mode == "box" else 0
            space = font.getlength(" ")
            fonts = [big if k == active else font for k in range(len(words))]
            widths = [f.getlength(t) + (2 * pad if (k == active and mode == "box") else 0)
                      for k, (f, t) in enumerate(zip(fonts, words))]
            total = sum(widths) + space * (len(words) - 1)
            if total <= max_w or size <= 36:
                break
            size -= 4

        stroke = max(3, size // 13) if st.stroke else 0
        h = int(size * 1.5) + stroke * 2 + pad * 2 + 24
        img = Image.new("RGBA", (self.width, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        baseline = h - int(size * 0.32) - stroke - pad - 8
        text_c, hl_c, stroke_c = _rgba(st.text_color), _rgba(st.highlight_color), _rgba(st.stroke_color)

        if st.shadow:  # ombre portée douce, dessinée sur un calque séparé puis floutée
            sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(sh)
            x = (self.width - total) / 2
            for k, (t, f, w) in enumerate(zip(words, fonts, widths)):
                off = pad if (k == active and mode == "box") else 0
                sd.text((x + off + 4, baseline + 6), t, font=f, anchor="ls", fill=(0, 0, 0, 150),
                        stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
                x += w + space
            from PIL import ImageFilter
            img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(6)))
            draw = ImageDraw.Draw(img)

        x = (self.width - total) / 2
        for k, (t, f, w) in enumerate(zip(words, fonts, widths)):
            if mode == "box" and k == active:
                l, tp, r, b = draw.textbbox((x + pad, baseline), t, font=f, anchor="ls")
                draw.rounded_rectangle((l - pad, tp - pad, r + pad, b + pad), radius=int(size * 0.18),
                                       fill=hl_c)
                fill = text_c
                draw.text((x + pad, baseline), t, font=f, anchor="ls", fill=fill,
                          stroke_width=stroke, stroke_fill=stroke_c)
            else:
                if mode == "word":
                    fill = hl_c if k == active else text_c
                elif mode == "progressive":
                    fill = hl_c if k <= active else text_c
                else:
                    fill = text_c
                draw.text((x, baseline), t, font=f, anchor="ls", fill=fill,
                          stroke_width=stroke, stroke_fill=stroke_c)
            x += w + space
        return _to_premult(img)

    def overlay(self, frame: np.ndarray, t: float) -> None:
        for ci, ch in enumerate(self.chunks):
            nxt_start = self.chunks[ci + 1][0].start if ci + 1 < len(self.chunks) else ch[-1].end + 0.4
            end = min(nxt_start, ch[-1].end + 0.6)
            if ch[0].start <= t < end:
                active = max((k for k, w in enumerate(ch) if w.start <= t), default=0)
                key = (ci, active)
                if key not in self.cache:
                    self.cache[key] = self._render(ci, active)
                pre, alpha = self.cache[key]
                y_center = int(frame.shape[0] * min(max(self.style.position, 0.08), 0.92))
                _blend(frame, pre, alpha, 0, y_center - pre.shape[0] // 2)
                return


def _title_overlay(text: str, out_w: int, out_h: int, font_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Titre d'accroche : bloc blanc arrondi, texte noir, en haut du clip."""
    size = int(min(out_w, 1080) * 0.052)
    font = load_font(font_name, size)
    max_w = int(out_w * 0.78)
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    lines = lines[:3]
    pad_x, pad_y, gap = int(size * 0.7), int(size * 0.5), int(size * 0.25)
    line_h = int(size * 1.15)
    box_w = int(max(font.getlength(l) for l in lines)) + pad_x * 2
    box_h = line_h * len(lines) + gap * (len(lines) - 1) + pad_y * 2
    img = Image.new("RGBA", (out_w, box_h + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0 = (out_w - box_w) // 2
    d.rounded_rectangle((x0, 10, x0 + box_w, 10 + box_h), radius=int(size * 0.45), fill=(255, 255, 255, 245))
    y = 10 + pad_y
    for l in lines:
        d.text((out_w / 2, y + line_h / 2), l, font=font, anchor="mm", fill=(15, 15, 15, 255))
        y += line_h + gap
    pre, alpha = _to_premult(img)
    return pre, alpha, int(out_h * 0.09)


# ---------------------------------------------------------------------------
# 6. Rendu final
# ---------------------------------------------------------------------------

_ENCODERS: str | None = None


def _video_encoder(quality: str = "high", fps: float = 30.0) -> list[str]:
    """x264 en qualité « master » : ce fichier sera recompressé par TikTok / YouTube,
    donc on leur envoie un fichier très propre (débit généreux, GOP court, couleurs BT.709)."""
    global _ENCODERS
    if _ENCODERS is None:
        _ENCODERS = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    gop = str(max(int(round(fps)) * 2, 24))
    color = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    if "libx264" in _ENCODERS:
        if quality == "high":
            return ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high", "-level:v", "4.2",
                    "-maxrate", "20M", "-bufsize", "40M", "-g", gop, "-bf", "2", "-tune", "film", *color]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", gop, *color]
    if "h264_videotoolbox" in _ENCODERS:
        return ["-c:v", "h264_videotoolbox", "-b:v", "16M" if quality == "high" else "8M", "-g", gop, *color]
    return ["-c:v", "mpeg4", "-q:v", "2"]


def _sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    """Accentuation douce (unsharp mask) pour compenser l'agrandissement du recadrage."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def render_clip(video: Path, clip: Clip, words: list[Word], out_path: Path,
                opts: RenderOptions | None = None, progress: Progress = _noop) -> Path:
    opts = opts or RenderOptions()
    out_w, out_h = opts.out_size()
    src_w, src_h, fps = _probe(video)
    dur = clip.duration
    ratio = out_w / out_h
    framing = opts.framing

    # Zone à découper dans la source
    if src_w / src_h > ratio:
        crop_w, crop_h = int(round(src_h * ratio / 2) * 2), src_h
    else:
        crop_w, crop_h = src_w, int(round(src_w / ratio / 2) * 2)
    crop_y = (src_h - crop_h) // 2

    upscale = out_w / crop_w
    sharpen_amount = min(0.6, max(0.0, (upscale - 1.0) * 0.45))
    sample_fps = 6.0
    if framing == "face" and crop_w < src_w:
        progress(0, "Suivi du visage")
        cam = _face_track(video, clip.start, dur, src_w, src_h, crop_w, sample_fps)
    else:
        cam = np.array([src_w / 2])

    # Mode « écran entier » : la vidéo complète au centre, sur un fond flouté
    if framing == "fit":
        fit_w = out_w
        fit_h = int(round(src_h * out_w / src_w / 2) * 2)
        if fit_h > out_h:
            fit_h = out_h
            fit_w = int(round(src_w * out_h / src_h / 2) * 2)
        cover = max(out_w / src_w, out_h / src_h)
        cov_w, cov_h = int(src_w * cover) + 2, int(src_h * cover) + 2

    caps = CaptionRenderer(words, clip.start, clip.end, opts.style, out_w) if opts.captions else None
    title = None
    if opts.title_mode in ("intro", "always") and clip.title:
        title = _title_overlay(clip.title, out_w, out_h, opts.style.font)

    dec = subprocess.Popen(
        [FFMPEG, "-loglevel", "error", "-ss", f"{clip.start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    tmp_path = out_path.with_name(out_path.stem + ".tmp.mp4")
    enc = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.5f}", "-i", "-",
         "-ss", f"{clip.start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
         "-map", "0:v", "-map", "1:a?", *_video_encoder(opts.quality, fps), "-pix_fmt", "yuv420p",
         # son au niveau des réseaux sociaux (-14 LUFS), sans saturation
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000",
         "-c:a", "aac", "-b:a", "192k" if opts.quality == "high" else "160k",
         "-shortest", "-movflags", "+faststart", str(tmp_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    enc_err: list[bytes] = []
    threading.Thread(target=lambda: enc_err.append(enc.stderr.read()), daemon=True).start()

    frame_bytes = src_w * src_h * 3
    total = max(int(dur * fps), 1)
    k = 0
    try:
        while True:
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(src_h, src_w, 3)
            t = k / fps

            if framing == "fit":
                small = cv2.resize(frame, (max(cov_w // 10, 2), max(cov_h // 10, 2)), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), 3)
                bg = cv2.resize(small, (cov_w, cov_h), interpolation=cv2.INTER_LINEAR)
                bx, by = (cov_w - out_w) // 2, (cov_h - out_h) // 2
                out = (bg[by:by + out_h, bx:bx + out_w] * 0.55).astype(np.uint8)
                fg = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
                ox, oy = (out_w - fit_w) // 2, (out_h - fit_h) // 2
                out[oy:oy + fit_h, ox:ox + fit_w] = fg
            else:
                cx = float(np.interp(t * sample_fps, np.arange(len(cam)), cam))
                x0 = int(round(min(max(cx - crop_w / 2, 0), src_w - crop_w)))
                crop = frame[crop_y:crop_y + crop_h, x0:x0 + crop_w]
                if crop_w >= out_w:
                    out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
                else:
                    hq = opts.quality == "high"
                    out = cv2.resize(crop, (out_w, out_h),
                                     interpolation=cv2.INTER_LANCZOS4 if hq else cv2.INTER_CUBIC)
                    if hq:
                        out = _sharpen(out, sharpen_amount)

            if title and (opts.title_mode == "always" or t < 3.0):
                pre, alpha, y = title
                _blend(out, pre, alpha, 0, y)
            if caps:
                caps.overlay(out, t)
            if opts.progress_bar:
                bar_h = max(6, out_h // 240)
                out[out_h - bar_h:, : int(out_w * min(t / dur, 1))] = _rgba(opts.style.highlight_color)[2::-1]

            enc.stdin.write(out.tobytes())
            k += 1
            if k % 15 == 0:
                progress(min(k / total, 1), "Montage du clip")
    finally:
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
        dec.wait()
        enc.wait()
    if enc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Erreur ffmpeg : {b''.join(enc_err).decode(errors='ignore')[-500:]}")
    tmp_path.replace(out_path)
    return out_path


def thumbnail(video_path: Path, at: float = 1.0) -> Path:
    thumb = video_path.with_suffix(".jpg")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(video_path),
                    "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "4", str(thumb)])
    return thumb


# ---------------------------------------------------------------------------
# Orchestration (utilisée en ligne de commande ; le serveur web fait la même chose étape par étape)
# ---------------------------------------------------------------------------

def slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_-]+", "-", s)[:50] or "clip"


def run(source: str, *, is_file: bool = False, n_clips: int = 5, dmin: float = 25, dmax: float = 60,
        whisper_model: str = "small", language: str | None = None, llm_model: str | None = None,
        render: RenderOptions | None = None, progress: Progress = _noop) -> tuple[list[Clip], Path]:
    WORK.mkdir(parents=True, exist_ok=True)

    def stage(lo: float, hi: float) -> Progress:
        return lambda p, m: progress(lo + (hi - lo) * max(0.0, min(p, 1.0)), m)

    video, title = import_local(source) if is_file else download(source, stage(0.0, 0.15))
    words, _duration, lang = transcribe(video, whisper_model, language, stage(0.15, 0.55))
    if not words:
        raise RuntimeError("Aucune parole détectée dans la vidéo.")
    clips = find_highlights(words, n_clips, dmin, dmax, llm_model, stage(0.55, 0.65))

    out_dir = OUT / slug(title)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(clips):
        cb = stage(0.65 + 0.35 * i / len(clips), 0.65 + 0.35 * (i + 1) / len(clips))
        path = out_dir / f"{i + 1:02d}_{slug(clip.title)}.mp4"
        render_clip(video, clip, words, path, render, lambda p, m, i=i: cb(p, f"{m} {i + 1}/{len(clips)}"))
        clip.path = str(path)
    progress(1.0, "Terminé")
    return clips, out_dir
