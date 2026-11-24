#!/usr/bin/env python3
# Copyright (c) 2026 Nicholas Ramacciotti
# Licensed under the GNU AGPLv3 License. See LICENSE and NOTICE.txt for details.
# -*- coding: utf-8 -*-
"""
RaceClipper RTX 4080 Super v11 - REGIA COMPOSTA + SMART TRIMMING + NPU REALE + SEO SENZA HASHTAG + TOP 10 MULTI-PASS

Trasforma una live di gara 16:9 (1-3 ore, telecronaca in italiano) nelle 10 migliori clip verticali 9:16
(1080x1920) piu' il rispettivo master 16:9 tagliato al fotogramma. Tutto in locale.

Hardware di riferimento: RTX 4080 Super 16 GB + Intel Core Ultra 7 265K (NPU via OpenVINO) + 48 GB RAM + Ollama.

Catena di lavoro (ogni fase salva il proprio risultato: una run interrotta riparte da dove si era fermata):
  FASE 0  Controlli: FFmpeg, encoder (NVENC o ripiego libx264), Ollama e modelli, OpenVINO/NPU.
  FASE 1  Audio: energia vocale, volume e transienti con passo 0,2 s sull'intera live.
  FASE 2  In parallelo: trascrizione Whisper large-v3 su CUDA (con tempi parola per parola) e scansione
          dell'intera live sulla NPU (webcam del telecronista, grafiche fisse, schermate statiche).
  FASE 3  Agenti di testo (Ollama): contesto gara e piloti, fasi della live, Action Detector sui blocchi di
          telecronaca; in piu' uno scout indipendente sui picchi di voce. Le finestre vicine vengono unite.
  FASE 4  Analisi visiva su CUDA: rilevamento auto, tracciamento con ID persistenti, distanza fra le auto,
          sorpassi (scambio d'ordine lungo la direzione di marcia), contatti, stacchi di camera.
          Smart Action Trimming: innesco e risoluzione sulla curva prossimita' + voce a 0,2 s.
  FASE 5  Verifica 1 (anti falsi positivi) e Verifica 2 (giudice visivo alla cieca + fisica): l'auto da sola
          che passeggia viene scartata qualunque cosa dica il telecronista.
  FASE 6  Punteggio 0-100, scelta delle finaliste, SEO Copywriter (zero hashtag), Verifica 3 (comitato
          editoriale: punteggi, titoli tutti diversi, nomi non inventati, rigenerazione dei testi deboli).
  FASE 7  Regia e render: 9:16 pieno, Reaction Split (webcam sopra), Vista larga + zoom, Dual a due auto;
          zoom punch, micro-shake sugli impatti, transizioni, vignettatura, audio a -14 LUFS in due passate.

Uso rapido:
  python race_clipper_4080_v11.py -i "https://www.youtube.com/watch?v=..." -o "C:/RaceClipper/Output_Gara"
  python race_clipper_4080_v11.py --selftest        (collaudo senza GPU, senza Ollama e senza video reale)
"""

from __future__ import annotations

import os
import re
import gc
import sys
import json
import math
import time
import shutil
import queue
import base64
import hashlib
import logging
import difflib
import argparse
import warnings
import threading
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

import cv2
import numpy as np
import requests

VERSION = "11.1"

# ---------------------------------------------------------------------------------------------------------
# Modelli: per ogni ruolo una lista in ordine di preferenza. Viene usato il primo disponibile; se manca e il
# download e' consentito viene scaricato, altrimenti si passa al successivo. Tutto sovrascrivibile da CLI.
# ---------------------------------------------------------------------------------------------------------
MODEL_PREFS: Dict[str, List[str]] = {
    # Regista: Action Detector, giudice editoriale, comitato di revisione (testo, JSON strutturato).
    "text": ["qwen3.8:27b", "qwen3.6:27b", "qwen3.5:27b", "qwen2.5:32b"],
    # Occhio critico: giudice visivo sui fotogrammi (deve accettare piu' immagini per richiesta).
    "vision": ["qwen3-vl:8b", "qwen3.5:9b", "gemma4:12b", "qwen2.5vl:7b", "minicpm-v:8b"],
    # SEO Copywriter in italiano: famiglia diversa dal revisore, cosi' la revisione e' davvero incrociata.
    "copy": ["gemma4:26b", "gemma4:12b", "gemma3:27b"],
}
YOLO_CUDA_PREFS = ["yolo26x.pt", "yolo11x.pt", "yolov8x.pt"]
# Sulla NPU del 265K (Arrow Lake) Ultralytics dichiara supportato YOLO11/YOLOv8; YOLO26 su NPU richiede
# Core Ultra 2xxV o serie 3xx. Per questo il modello NPU resta YOLO11n.
YOLO_NPU_PREFS = ["yolo11n.pt", "yolov8n.pt"]
WHISPER_MODEL = "large-v3"

OUT_W, OUT_H = 1080, 1920
PERSON_CLASS = 0
VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck

# Soglie di regia e di analisi. Sono raccolte qui per poterle ritoccare dopo le prime prove su gare reali.
TUNING: Dict[str, float] = {
    "bin": 0.2,                 # risoluzione delle curve (s)
    "det_fps": 20.0,            # frequenza del rilevamento auto nell'analisi profonda
    "det_conf": 0.30,           # confidenza minima YOLO (il tracker filtra il rumore)
    "analysis_width": 1280,     # larghezza dei frame di analisi
    "pad_pre": 12.0,            # secondi analizzati prima della finestra candidata
    "pad_post": 15.0,           # secondi analizzati dopo
    "lead_in": 1.8,             # contesto prima dell'innesco (uscita curva / presa di scia)
    "tail_out": 1.2,            # respiro dopo la risoluzione
    "gap_close": 1.0,           # distanza (in larghezze auto) sotto cui e' "lotta ravvicinata"
    "gap_w2w": 0.45,            # sotto cui e' "ruota a ruota"
    "quiet_dip": 3.5,           # buco di attivita' tollerato cercando l'innesco (s)
    "quiet_end": 4.5,           # calma continua che chiude l'azione (s): un rettilineo non chiude un duello
    "chain_s": 15.0,            # un sorpasso/contatto entro questi secondi dal bordo appartiene alla stessa azione
    "grow_post": 30.0,          # se l'azione tocca la fine della finestra analizzata, si analizza cosi' tanto in piu'
    "grow_pre": 20.0,           # idem verso l'inizio
    "punch_amp": 0.10,          # ampiezza dello zoom punch (+10%)
    "shake_s": 0.25,            # durata del micro-shake sull'impatto (s)
    "mode_hold": 2.4,           # tenuta minima di un layout dentro la stessa inquadratura (s)
    "morph_s": 0.40,            # durata dell'animazione fra due layout (s)
    "full_fit": 0.92,           # quota del crop occupabile dal duello prima di cambiare layout
    "vignette": 0.14,           # scurimento massimo ai bordi verticali
}

REACTION_TOP_H = 576            # 30% superiore: webcam del telecronista
DUAL_TOP_H = 960                # split a due auto: 50/50

_PRINT_LOCK = threading.Lock()


def log(msg: str = "") -> None:
    with _PRINT_LOCK:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def warn(msg: str) -> None:
    log(f"    [!] {msg}")


class RaceClipperError(RuntimeError):
    """Errore bloccante con messaggio gia' leggibile dall'utente."""


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def even(v: float) -> int:
    return int(round(v / 2.0)) * 2


def format_time(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:02d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))


def sanitize_filename(text: str, max_len: int = 44) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-\s]", "", strip_accents(text))
    clean = re.sub(r"\s+", "_", clean.strip())
    clean = re.sub(r"_+", "_", clean)
    return clean[:max_len].strip("_") or "Highlight_Gara"


def parse_timestamp_value(val: Any) -> Optional[float]:
    """Accetta 754.2, "754.2s", "12:34", "1:02:03". Restituisce None se non c'e' un numero."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower().replace(",", ".")
    if ":" in s:
        try:
            parts = [float(p) for p in re.findall(r"\d+(?:\.\d+)?", s)]
        except ValueError:
            parts = []
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def viral_tier_label(score: int) -> str:
    if score >= 88:
        return "MUST POST (azione clamorosa verificata video + audio)"
    if score >= 78:
        return "ALTO POTENZIALE (duello o sorpasso solido)"
    if score >= 65:
        return "BUON CONTENUTO (azione di gara interessante)"
    return "SECONDARIO"


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Filtro gaussiano a fase zero con bordi riflessi (la traiettoria non accumula ritardo)."""
    v = np.asarray(values, dtype=np.float64)
    if sigma <= 0.3 or len(v) < 3:
        return v.copy()
    radius = int(max(1, round(sigma * 3)))
    radius = min(radius, len(v) - 1)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(v, (radius, radius), mode="reflect")
    return np.convolve(padded, k, mode="valid")


def fill_nan(values: np.ndarray, default: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).copy()
    ok = ~np.isnan(v)
    if not ok.any():
        v[:] = default
        return v
    idx = np.arange(len(v))
    v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    return v


# ---------------------------------------------------------------------------------------------------------
# Cache di fase: ogni fase salva il risultato con una chiave; se la chiave non cambia, la fase viene saltata.
# ---------------------------------------------------------------------------------------------------------
class StageCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(key: Any) -> str:
        return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

    def load(self, name: str, key: Any) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self.root / f"{name}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("key") == self._key(key):
                return blob.get("data")
        except Exception:
            return None
        return None

    def save(self, name: str, key: Any, data: Any) -> None:
        path = self.root / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"key": self._key(key), "data": data}, f, ensure_ascii=False)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------
# FFmpeg: probe, encoder, lettura e scrittura di frame via pipe
# ---------------------------------------------------------------------------------------------------------
def _no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def log_tail(path: Path, n: int = 12) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def run_ffmpeg(cmd: List[str], log_path: Path, what: str, timeout: Optional[float] = None) -> None:
    """Esegue FFmpeg scrivendo stderr su file: niente pipe da svuotare, niente deadlock, errore sempre visibile."""
    with open(log_path, "ab") as lf:
        lf.write(("\n$ " + " ".join(cmd) + "\n").encode("utf-8", "replace"))
        lf.flush()
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=lf,
                                 timeout=timeout, creationflags=_no_window_flags())
        except FileNotFoundError:
            raise RaceClipperError("FFmpeg non trovato nel PATH.")
        except subprocess.TimeoutExpired:
            raise RaceClipperError(f"FFmpeg non ha risposto in tempo durante: {what}")
    if res.returncode != 0:
        raise RaceClipperError(f"FFmpeg ha fallito ({what}), codice {res.returncode}.\n{log_tail(log_path)}")


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps_num: int
    fps_den: int
    duration: float
    has_audio: bool
    src_fps: float = 0.0

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def fps_str(self) -> str:
        return f"{self.fps_num}/{self.fps_den}"

    @property
    def total_frames(self) -> int:
        return int(self.duration * self.fps)

    def frame_time(self, n: int) -> float:
        return n / self.fps


def ffprobe_info(path: Path) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=_no_window_flags())
    except FileNotFoundError:
        raise RaceClipperError("ffprobe non trovato nel PATH (fa parte di FFmpeg).")
    if res.returncode != 0:
        raise RaceClipperError(f"ffprobe non riesce a leggere {path.name}: {res.stderr.decode('utf-8', 'replace')[:300]}")
    meta = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    streams = meta.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")), None)
    if v is None:
        raise RaceClipperError(f"{path.name} non contiene una traccia video.")

    def _rate(txt: str) -> Tuple[int, int]:
        try:
            a, b = str(txt).split("/")
            a, b = int(a), int(b)
            return (a, b) if a > 0 and b > 0 else (0, 1)
        except Exception:
            return (0, 1)

    num, den = _rate(v.get("avg_frame_rate", "0/0"))
    if num == 0 or num / den > 240:
        num, den = _rate(v.get("r_frame_rate", "0/0"))
    if num == 0:
        num, den = 30, 1
    src_fps = num / den
    # Il lavoro avviene sempre a frame rate costante; oltre i 60 fps si scende a 60 (limite utile dei social).
    if src_fps > 60.5:
        num, den = 60, 1
    duration = 0.0
    for cand in (meta.get("format", {}).get("duration"), v.get("duration")):
        try:
            duration = max(duration, float(cand))
        except (TypeError, ValueError):
            pass
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return VideoInfo(path=path, width=int(v["width"]), height=int(v["height"]), fps_num=num, fps_den=den,
                     duration=duration, has_audio=has_audio, src_fps=src_fps)


def probe_encoder(name: str) -> bool:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x320:r=30:d=0.3",
           "-frames:v", "4", "-c:v", name, "-f", "null", "-"]
    try:
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=40, creationflags=_no_window_flags()).returncode == 0
    except Exception:
        return False


def pick_video_encoder(pref: str) -> str:
    if pref in ("auto", "nvenc") and probe_encoder("h264_nvenc"):
        return "h264_nvenc"
    if pref == "nvenc":
        warn("NVENC non disponibile (driver o sessioni esaurite): uso libx264.")
    if probe_encoder("libx264"):
        return "libx264"
    raise RaceClipperError("Nessun encoder H.264 utilizzabile in FFmpeg (ne' h264_nvenc ne' libx264).")


COLOR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def video_encoder_args(encoder: str, quality: str, fps: float, fast: bool = False) -> List[str]:
    """quality: 'social' per il 9:16, 'master' per il 16:9 destinato alla post-produzione."""
    gop = str(max(12, int(round(fps * 2))))
    if encoder == "h264_nvenc":
        cq, maxrate, buf = ("16", "60M", "120M") if quality == "master" else ("19", "30M", "60M")
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "0",
                "-maxrate", maxrate, "-bufsize", buf, "-spatial-aq", "1", "-temporal-aq", "1",
                "-rc-lookahead", "32", "-bf", "3", "-g", gop, "-profile:v", "high", "-pix_fmt", "yuv420p"]
    crf = "15" if quality == "master" else "17"
    preset = "veryfast" if fast else "slow"
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-g", gop, "-profile:v", "high", "-pix_fmt", "yuv420p"]


class FFmpegReader:
    """
    Legge frame BGR da un intervallo esatto della sorgente: parte dal frame `start_frame` della griglia a
    frame rate costante e ne restituisce `n_frames`. Analisi e render usano lo stesso meccanismo, quindi gli
    indici di frame coincidono fra le due fasi (cosa che OpenCV non garantisce con sorgenti a GOP lungo o VFR).
    La conversione YUV->BGR usa la matrice BT.709 dichiarata in modo esplicito.
    """

    def __init__(self, info: VideoInfo, start_frame: int, n_frames: int, out_w: int, out_h: int,
                 log_path: Path, flags: str = "bicubic"):
        self.w, self.h = int(out_w), int(out_h)
        self.frame_bytes = self.w * self.h * 3
        self.n_frames = int(n_frames)
        self.read_count = 0
        t = max(0.0, (start_frame - 0.4) / info.fps)
        vf = (f"fps={info.fps_str},scale={self.w}:{self.h}:flags={flags}:in_color_matrix=bt709,format=bgr24")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if t > 0:
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-i", str(info.path), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf,
                "-frames:v", str(self.n_frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._log_path = log_path
        self._log = open(log_path, "ab")
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self._log,
                                         bufsize=1024 * 1024 * 8, creationflags=_no_window_flags())
        except FileNotFoundError:
            self._log.close()
            raise RaceClipperError("FFmpeg non trovato nel PATH.")

    def read(self) -> Optional[np.ndarray]:
        if self.read_count >= self.n_frames:
            return None
        buf = bytearray(self.frame_bytes)
        mv = memoryview(buf)
        got = 0
        while got < self.frame_bytes:
            n = self.proc.stdout.readinto(mv[got:])
            if not n:
                break
            got += n
        if got < self.frame_bytes:
            return None
        self.read_count += 1
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            if self.proc.stdout:
                self.proc.stdout.close()
            self.proc.wait(timeout=20)
        except Exception:
            pass
        try:
            self._log.close()
        except Exception:
            pass

    def __enter__(self) -> "FFmpegReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FFmpegWriter:
    """
    Scrive frame grezzi sullo stdin di FFmpeg. stderr va su file (non su pipe): cosi' non esiste un secondo
    canale da svuotare e non puo' crearsi un deadlock. Se FFmpeg muore, l'errore emerge subito con il log:
    su Windows la pipe chiusa solleva OSError (Errno 22), su POSIX BrokenPipeError; qui sono gestiti entrambi.
    """

    def __init__(self, cmd: List[str], log_path: Path, what: str):
        self.what = what
        self.log_path = log_path
        self._log = open(log_path, "ab")
        self._log.write(("\n$ " + " ".join(cmd) + "\n").encode("utf-8", "replace"))
        self._log.flush()
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log,
                                         creationflags=_no_window_flags())
        except FileNotFoundError:
            self._log.close()
            raise RaceClipperError("FFmpeg non trovato nel PATH.")
        self.frames = 0

    def _fail(self, detail: str) -> RaceClipperError:
        try:
            self._log.flush()
        except Exception:
            pass
        return RaceClipperError(f"FFmpeg si e' interrotto durante {self.what} ({detail}).\n{log_tail(self.log_path)}")

    def write(self, frame: np.ndarray) -> None:
        if self.proc.poll() is not None:
            raise self._fail(f"uscito con codice {self.proc.returncode} dopo {self.frames} frame")
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).data)
        except (BrokenPipeError, OSError, ValueError) as e:
            raise self._fail(f"pipe chiusa dopo {self.frames} frame: {e}")
        self.frames += 1

    def close(self, timeout: float = 900.0) -> None:
        err: Optional[RaceClipperError] = None
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            rc = self.proc.wait(timeout=timeout)
            if rc != 0:
                err = self._fail(f"codice di uscita {rc}")
        except subprocess.TimeoutExpired:
            self.proc.kill()
            err = self._fail("nessuna risposta alla chiusura")
        finally:
            try:
                self._log.close()
            except Exception:
                pass
        if err:
            raise err

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        try:
            self._log.close()
        except Exception:
            pass


def extract_still(info: VideoInfo, t: float, out_path: Path, width: int, log_path: Path) -> bool:
    """Estrae un fotogramma JPEG al tempo t (per il giudice visivo e per le anteprime)."""
    h = even(width * info.height / info.width)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", f"{max(0.0, t):.3f}",
           "-i", str(info.path), "-map", "0:v:0", "-frames:v", "1",
           "-vf", f"scale={width}:{h}:flags=bicubic", "-q:v", "3", str(out_path)]
    try:
        run_ffmpeg(cmd, log_path, "estrazione fotogramma", timeout=120)
    except RaceClipperError:
        return False
    return out_path.exists() and out_path.stat().st_size > 0


# ---------------------------------------------------------------------------------------------------------
# FASE 0 - Acquisizione del video
# ---------------------------------------------------------------------------------------------------------
def acquire_video(source: str, temp_dir: Path, max_height: int) -> Path:
    if not re.match(r"^https?://", source):
        video_path = Path(source).expanduser().resolve()
        if not video_path.exists():
            raise RaceClipperError(f"File video non trovato: {video_path}")
        return video_path

    # La cache e' legata all'URL: una gara diversa non riusa mai il file di un'altra.
    tag = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    for existing in sorted(temp_dir.glob(f"source_{tag}.*")):
        if existing.suffix.lower() in (".mkv", ".mp4", ".webm"):
            try:
                ffprobe_info(existing)
                log(f"\n[FASE 0] Video gia' scaricato e integro: {existing.name}")
                return existing
            except RaceClipperError:
                existing.unlink(missing_ok=True)

    log(f"\n[FASE 0] Scarico la live da YouTube (fino a {max_height}p): {source}")
    for stale in temp_dir.glob(f"dl_{tag}.*"):
        stale.unlink(missing_ok=True)
    cmd = [
        "yt-dlp",
        "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        # a parita' di risoluzione e fps preferisce H.264: decodifica piu' leggera durante l'analisi
        "-S", "res,fps,vcodec:h264",
        "--merge-output-format", "mkv",
        "--no-playlist",
        "-o", str(temp_dir / f"dl_{tag}.%(ext)s"),
        source,
    ]
    try:
        res = subprocess.run(cmd)
    except FileNotFoundError:
        raise RaceClipperError("yt-dlp non trovato nel PATH (pip install -U yt-dlp).")
    if res.returncode != 0:
        raise RaceClipperError("yt-dlp non ha completato il download. Aggiornalo (pip install -U yt-dlp) e riprova.")
    done = [p for p in temp_dir.glob(f"dl_{tag}.*") if p.suffix.lower() in (".mkv", ".mp4", ".webm")]
    if not done:
        raise RaceClipperError("Download terminato ma file video non trovato.")
    final = temp_dir / f"source_{tag}{done[0].suffix.lower()}"
    os.replace(done[0], final)   # il nome definitivo compare solo a download completo
    ffprobe_info(final)
    return final


def video_fingerprint(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.name}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------
# FASE 1 - Audio: energia vocale, volume e transienti a passo 0,2 s
# ---------------------------------------------------------------------------------------------------------
@dataclass
class AudioFeatures:
    hop: float
    voice: np.ndarray      # 0..1 energia nella banda della voce, normalizzata sui percentili dell'intera live
    loud: np.ndarray       # 0..1 volume complessivo
    flux: np.ndarray       # z-score robusto dei transienti a banda larga (botti, urti)

    def __len__(self) -> int:
        return len(self.voice)

    def idx(self, t: float) -> int:
        return int(clamp(round(t / self.hop), 0, max(0, len(self.voice) - 1)))

    def window(self, arr: np.ndarray, t0: float, t1: float) -> np.ndarray:
        a, b = self.idx(t0), self.idx(t1)
        return arr[a:max(a + 1, b + 1)]


def load_audio_16k(info: VideoInfo, log_path: Path) -> np.ndarray:
    if not info.has_audio:
        return np.zeros(int(info.duration * 16000), dtype=np.float32)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(info.path), "-map", "0:a:0",
           "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=lf,
                                bufsize=1024 * 1024 * 4, creationflags=_no_window_flags())
        raw = proc.stdout.read()
        rc = proc.wait()
    if rc != 0 or len(raw) < 3200:
        raise RaceClipperError(f"Impossibile estrarre l'audio dalla sorgente.\n{log_tail(log_path)}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    del raw
    audio /= 32768.0
    return audio


def compute_audio_features(x: np.ndarray, sr: int = 16000, hop_s: float = 0.2) -> AudioFeatures:
    hop = int(round(sr * hop_s))
    win = hop * 2
    pad = (win - hop) // 2
    n = max(1, len(x) // hop + 1)
    xp = np.concatenate([np.zeros(pad, np.float32), x.astype(np.float32, copy=False), np.zeros(win + hop, np.float32)])
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    voice_idx = (freqs >= 250) & (freqs <= 3500)
    all_idx = (freqs >= 60) & (freqs <= 7500)
    edges = np.geomspace(80.0, 7800.0, 49)
    starts = np.searchsorted(freqs, edges[:-1])
    last_bin = int(np.searchsorted(freqs, edges[-1]))

    voice_p = np.zeros(n, np.float64)
    tot_p = np.zeros(n, np.float64)
    bands = np.zeros((n, 48), np.float32)
    frames_view = np.lib.stride_tricks.sliding_window_view(xp, win)
    block = 1024
    for s in range(0, n, block):
        e = min(n, s + block)
        fr = frames_view[s * hop:(e - 1) * hop + 1:hop] * window
        p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        voice_p[s:e] = p[:, voice_idx].sum(axis=1)
        tot_p[s:e] = p[:, all_idx].sum(axis=1)
        bands[s:e] = np.add.reduceat(p[:, :last_bin], starts, axis=1)
        del fr, p

    def _norm(power: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
        db = 10.0 * np.log10(power + 1e-12)
        lo, hi = np.percentile(db, lo_q), np.percentile(db, hi_q)
        return np.clip((db - lo) / max(1e-6, hi - lo), 0.0, 1.0).astype(np.float32)

    voice = _norm(voice_p, 30, 98)
    loud = _norm(tot_p, 20, 99)
    eps = 1e-4 * float(np.median(bands.sum(axis=1)) + 1e-12)
    logb = np.log(bands + eps)
    diff = np.diff(logb, axis=0, prepend=logb[:1])
    flux = np.maximum(diff, 0.0).sum(axis=1)
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med)))
    flux_z = ((flux - med) / (1.4826 * mad + 1e-9)).astype(np.float32)
    return AudioFeatures(hop=hop_s, voice=voice, loud=loud, flux=flux_z)


def save_audio_features(path: Path, f: AudioFeatures) -> None:
    np.savez_compressed(path, hop=np.float32(f.hop), voice=f.voice, loud=f.loud, flux=f.flux)


def load_audio_features(path: Path) -> Optional[AudioFeatures]:
    try:
        z = np.load(path)
        return AudioFeatures(hop=float(z["hop"]), voice=z["voice"], loud=z["loud"], flux=z["flux"])
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------------
# FASE 2a - Trascrizione (faster-whisper large-v3 su CUDA, tempi parola per parola)
# ---------------------------------------------------------------------------------------------------------
def _import_torch() -> Any:
    import torch  # noqa: WPS433 (import pesante, solo quando serve)
    if os.name == "nt":
        # CTranslate2 (faster-whisper) cerca cuDNN/cuBLAS: su Windows sono nella cartella lib di torch.
        try:
            lib_dir = Path(torch.__file__).parent / "lib"
            if lib_dir.exists():
                os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(lib_dir))
        except Exception:
            pass
    return torch


def flush_vram() -> None:
    gc.collect()
    try:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def transcribe_audio(audio: np.ndarray, duration: float, language: str, hints: str) -> List[Dict[str, Any]]:
    try:
        torch = _import_torch()
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RaceClipperError(f"Per la trascrizione servono torch e faster-whisper: {e}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    log(f"\n[FASE 2a] Trascrizione {WHISPER_MODEL} su {device.upper()} ({compute_type}) con tempi per parola...")

    glossary = (
        "Telecronaca sportiva in italiano di motorsport e sim racing: qualifiche, pole position, staccata, "
        "sorpasso, incrocio di traiettoria, ruota a ruota, three wide, cordolo, chicane, scia, testacoda, "
        "contatto, drive through, pit stop, settore fucsia, giro veloce, griglia di partenza."
    )
    hints = hints.strip()
    if hints:
        glossary += f" Nomi citati: {hints}."

    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
    kwargs: Dict[str, Any] = dict(
        language=language, beam_size=5, best_of=5, condition_on_previous_text=False, initial_prompt=glossary,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400), word_timestamps=True,
    )
    if hints:
        kwargs["hotwords"] = hints
    try:
        segments_gen, _ = model.transcribe(audio, **kwargs)
    except TypeError:
        kwargs.pop("hotwords", None)      # versioni di faster-whisper precedenti alla 1.0.2
        segments_gen, _ = model.transcribe(audio, **kwargs)

    segments: List[Dict[str, Any]] = []
    total_min = max(1, int(duration // 60))
    next_mark = 600.0
    for seg in segments_gen:
        text = (seg.text or "").strip()
        if text:
            words = []
            for w in (seg.words or []):
                token = (w.word or "").strip()
                if token:
                    words.append({"w": token, "s": round(float(w.start), 2), "e": round(float(w.end), 2)})
            segments.append({"start": round(float(seg.start), 2), "end": round(float(seg.end), 2),
                             "text": text, "words": words})
        if seg.end >= next_mark:
            log(f"    ...trascritti {int(seg.end // 60)} min su {total_min}")
            next_mark += 600.0
    del model
    flush_vram()
    log(f" -> Trascrizione completata: {len(segments)} segmenti. VRAM liberata.")
    return segments


def norm_word(token: str) -> str:
    return re.sub(r"[^a-z0-9àèéìòóùç']", "", token.lower())


def flatten_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lista piatta di parole con tempi. Se un segmento non ha i tempi per parola li distribuisce in modo uniforme."""
    out: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                n = norm_word(w["w"])
                if n:
                    out.append({"s": float(w["s"]), "e": float(w["e"]), "w": n})
        else:
            toks = [norm_word(t) for t in seg["text"].split()]
            toks = [t for t in toks if t]
            if not toks:
                continue
            span = max(0.2, float(seg["end"]) - float(seg["start"]))
            step = span / len(toks)
            for i, t in enumerate(toks):
                out.append({"s": float(seg["start"]) + i * step, "e": float(seg["start"]) + (i + 1) * step, "w": t})
    out.sort(key=lambda w: w["s"])
    return out


def text_in_range(segments: List[Dict[str, Any]], t0: float, t1: float, max_chars: int = 0) -> str:
    parts = [s["text"] for s in segments if s["end"] >= t0 and s["start"] <= t1]
    txt = " ".join(parts).strip()
    if max_chars and len(txt) > max_chars:
        half = max_chars // 2
        txt = txt[:half] + " [...] " + txt[-half:]
    return txt


# Frasi che la telecronaca usa quando succede qualcosa: servono a collocare innesco, contatti, errori e replay.
ATTACK_PATTERNS = [
    r"\bocchio\b", r"\battenzione\b", r"\bci prova\b", r"\bsi butta\b", r"\bsi infila\b", r"\battacc[ao]\b",
    r"\baffianca\w*\b", r"\ball'interno\b", r"\ball'esterno\b", r"\bstaccata\b", r"\bruota a ruota\b",
    r"\bfianco a fianco\b", r"\bsorpass\w+\b", r"\blo passa\b", r"\bla passa\b", r"\bincroci\w*\b",
    r"\bchiude la porta\b", r"\bthree wide\b", r"\btre larghi\b", r"\bin scia\b", r"\bvolata\b", r"\bsemafor\w+\b",
]
CONTACT_PATTERNS = [
    r"\bcontatto\b", r"\bsi toccano\b", r"\btocca\b", r"\btoccat\w+\b", r"\bbotto\b", r"\bbotta\b",
    r"\bsportellata\b", r"\btampon\w+\b", r"\bincidente\b", r"\ba muro\b", r"\bmuro\b", r"\bbarriere\b",
    r"\bimpatto\b", r"\bcolpisce\b", r"\bcrash\b",
]
SPIN_PATTERNS = [
    r"\btestacoda\b", r"\bsi gira\b", r"\bgirato\b", r"\bperde il controllo\b", r"\bperde la macchina\b",
    r"\bla salva\b", r"\bsalvataggio\b", r"\bdi traverso\b", r"\bva lungo\b", r"\bfuori pista\b",
    r"\bnella ghiaia\b", r"\bsull'erba\b", r"\bin ghiaia\b",
]
REPLAY_PATTERNS = [
    r"\breplay\b", r"\brivediamo\w*\b", r"\brivedere\b", r"\briguardiamo\w*\b", r"\brallentatore\b", r"\brallenty\b",
]


class WordIndex:
    """Ricerca di frasi nella telecronaca con il tempo esatto della prima parola."""

    def __init__(self, words: List[Dict[str, Any]]):
        self.words = words
        self.starts = np.array([w["s"] for w in words], dtype=np.float64) if words else np.zeros(0)
        self.ends = np.array([w["e"] for w in words], dtype=np.float64) if words else np.zeros(0)
        pos, chunks, offsets = 0, [], []
        for w in words:
            offsets.append(pos)
            chunks.append(w["w"])
            pos += len(w["w"]) + 1
        self.text = " ".join(chunks)
        self.offsets = np.array(offsets, dtype=np.int64) if offsets else np.zeros(0, np.int64)
        self._cache: Dict[str, np.ndarray] = {}

    def times(self, patterns: Sequence[str], key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        found: List[float] = []
        if len(self.words):
            for pat in patterns:
                for m in re.finditer(pat, self.text):
                    i = int(np.searchsorted(self.offsets, m.start(), side="right") - 1)
                    found.append(float(self.starts[max(0, i)]))
        arr = np.array(sorted(set(found)), dtype=np.float64)
        self._cache[key] = arr
        return arr

    def in_range(self, patterns: Sequence[str], key: str, t0: float, t1: float) -> np.ndarray:
        arr = self.times(patterns, key)
        return arr[(arr >= t0) & (arr <= t1)]

    def gaps(self, t0: float, t1: float, min_gap: float) -> List[Tuple[float, float]]:
        """Pause del parlato (fine parola, inizio parola successiva) dentro l'intervallo."""
        if not len(self.words):
            return [(t0, t1)]
        a = int(np.searchsorted(self.ends, t0 - 3.0))
        b = int(np.searchsorted(self.starts, t1 + 3.0))
        out: List[Tuple[float, float]] = []
        prev_end = self.ends[a - 1] if a > 0 else -1e9
        for i in range(a, min(b + 1, len(self.words))):
            if self.starts[i] - prev_end >= min_gap:
                out.append((float(prev_end), float(self.starts[i])))
            prev_end = max(prev_end, self.ends[i])
        if b >= len(self.words):
            out.append((float(prev_end), 1e9))
        return out

    def word_at(self, t: float) -> Optional[Tuple[float, float]]:
        if not len(self.words):
            return None
        i = int(np.searchsorted(self.starts, t, side="right") - 1)
        if i >= 0 and self.starts[i] <= t <= self.ends[i]:
            return float(self.starts[i]), float(self.ends[i])
        return None

    def rate(self, t0: float, t1: float) -> float:
        if not len(self.words) or t1 <= t0:
            return 0.0
        a = int(np.searchsorted(self.starts, t0))
        b = int(np.searchsorted(self.starts, t1))
        return (b - a) / (t1 - t0)


# ---------------------------------------------------------------------------------------------------------
# FASE 2b - Screener OpenVINO sulla NPU Intel (ripiego: GPU integrata, poi CPU)
# ---------------------------------------------------------------------------------------------------------
Det = Tuple[int, float, float, float, float, float]   # classe, confidenza, x1, y1, x2, y2 (normalizzati 0..1)


def default_models_dir() -> Path:
    env = os.environ.get("RACECLIPPER_MODELS")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "raceclipper_models"


class OpenVINOScreener:
    """
    YOLO11n in formato OpenVINO compilato sulla NPU. Lavora su frame ridotti (640x384) e riconosce persone
    (riquadro webcam del telecronista) e veicoli (attivita' in pista). E' il modello "sempre acceso":
    scansiona l'intera live mentre la RTX trascrive e, durante l'analisi delle candidate, controlla la webcam
    in un thread separato senza toccare la VRAM.
    """

    def __init__(self, models_dir: Path, prefs: Sequence[str] = YOLO_NPU_PREFS,
                 devices: Sequence[str] = ("NPU", "GPU", "CPU"), in_w: int = 640, in_h: int = 384,
                 conf: float = 0.30):
        import openvino as ov
        self.in_w, self.in_h, self.conf = in_w, in_h, conf
        self.device = ""
        self.device_name = ""
        models_dir.mkdir(parents=True, exist_ok=True)
        core = ov.Core()
        available = list(core.available_devices)
        xml = self._ensure_ir(models_dir, prefs)
        model = core.read_model(str(xml))
        self.model_name = xml.stem
        errors: List[str] = []
        for dev in devices:
            if not any(a == dev or a.startswith(dev + ".") for a in available):
                continue
            try:
                compiled = core.compile_model(model, dev, {"CACHE_DIR": str(models_dir / "ov_cache")})
                req = compiled.create_infer_request()
                req.infer({0: np.zeros((1, 3, in_h, in_w), np.float32)})   # prova reale: alcuni driver falliscono qui
                self._compiled, self._req, self.device = compiled, req, dev
                try:
                    self.device_name = str(core.get_property(dev, "FULL_DEVICE_NAME"))
                except Exception:
                    self.device_name = dev
                break
            except Exception as e:
                errors.append(f"{dev}: {str(e)[:120]}")
        if not self.device:
            raise RuntimeError("nessun device OpenVINO utilizzabile (" + "; ".join(errors) + ")")
        self._lock = threading.Lock()

    def _ensure_ir(self, models_dir: Path, prefs: Sequence[str]) -> Path:
        last_err: Optional[Exception] = None
        for name in prefs:
            stem = Path(name).stem
            ir_dir = models_dir / f"{stem}_{self.in_h}x{self.in_w}_openvino_model"
            xmls = sorted(ir_dir.glob("*.xml")) if ir_dir.exists() else []
            if xmls:
                return next((p for p in xmls if p.stem.endswith("_fp16")), xmls[0])
            try:
                _import_torch()
                from ultralytics import YOLO
                log(f"    Esporto {name} in OpenVINO ({self.in_w}x{self.in_h}, forma statica: richiesta dalla NPU)...")
                yolo = YOLO(str(models_dir / name))
                exported = Path(str(yolo.export(format="openvino", imgsz=[self.in_h, self.in_w], dynamic=False,
                                                batch=1, verbose=False)))
                if exported.is_file():
                    exported = exported.parent
                if exported.resolve() != ir_dir.resolve():
                    if ir_dir.exists():
                        shutil.rmtree(ir_dir, ignore_errors=True)
                    shutil.move(str(exported), str(ir_dir))
                xmls = sorted(p for p in ir_dir.glob("*.xml") if not p.stem.endswith("_fp16"))
                if xmls:
                    # pesi compressi a FP16 (precisione nativa della NPU) con l'API OpenVINO: indipendente
                    # dalla versione di Ultralytics (il flag `half` del v10 oggi e' deprecato)
                    try:
                        import openvino as ov
                        fp16 = xmls[0].with_name(xmls[0].stem + "_fp16.xml")
                        ov.save_model(ov.Core().read_model(str(xmls[0])), str(fp16), compress_to_fp16=True)
                        return fp16
                    except Exception:
                        return xmls[0]
            except Exception as e:
                last_err = e
        raise RuntimeError(f"export OpenVINO non riuscito: {last_err}")

    def infer(self, frame_bgr: np.ndarray) -> List[Det]:
        h, w = frame_bgr.shape[:2]
        r = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        img = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame_bgr
        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        ox, oy = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = img
        blob = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        with self._lock:
            self._req.infer({0: blob})
            out = np.array(self._req.get_output_tensor(0).data, copy=True)
        return self._decode(out, r, ox, oy, w, h)

    def _decode(self, out: np.ndarray, r: float, ox: int, oy: int, w: int, h: int) -> List[Det]:
        wanted = (PERSON_CLASS,) + VEHICLE_CLASSES
        boxes: List[List[float]] = []
        scores: List[float] = []
        classes: List[int] = []
        o = out[0]
        if o.ndim == 2 and o.shape[1] == 6 and o.shape[0] != 6:
            # testa end-to-end (YOLO26 / YOLOv10): righe [x1, y1, x2, y2, conf, classe]
            for x1, y1, x2, y2, c, k in o:
                if c >= self.conf and int(k) in wanted:
                    boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                    scores.append(float(c))
                    classes.append(int(k))
        else:
            p = o.T if o.shape[0] < o.shape[1] else o        # (N, 4 + classi)
            cls_scores = p[:, 4:]
            sub = cls_scores[:, list(wanted)]
            best = sub.argmax(axis=1)
            conf = sub[np.arange(len(sub)), best]
            keep = conf >= self.conf
            for (cx, cy, bw, bh), c, k in zip(p[keep, :4], conf[keep], best[keep]):
                boxes.append([float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)])
                scores.append(float(c))
                classes.append(int(wanted[int(k)]))
        if not boxes:
            return []
        dets: List[Det] = []
        # NMS separata per persone e veicoli (fra veicoli e' agnostica alla classe: car/truck sullo stesso box)
        for group in ((PERSON_CLASS,), VEHICLE_CLASSES):
            idx = [i for i, k in enumerate(classes) if k in group]
            if not idx:
                continue
            keep = cv2.dnn.NMSBoxes([boxes[i] for i in idx], [scores[i] for i in idx], self.conf, 0.5)
            for j in np.array(keep).reshape(-1):
                i = idx[int(j)]
                x, y, bw, bh = boxes[i]
                x1 = clamp((x - ox) / r / w, 0.0, 1.0)
                y1 = clamp((y - oy) / r / h, 0.0, 1.0)
                x2 = clamp((x + bw - ox) / r / w, 0.0, 1.0)
                y2 = clamp((y + bh - oy) / r / h, 0.0, 1.0)
                if x2 - x1 > 0.004 and y2 - y1 > 0.004:
                    dets.append((classes[i], scores[i], x1, y1, x2, y2))
        return dets


def _cluster_person_boxes(boxes: List[Tuple[int, float, float, float, float]], n_samples: int,
                          min_presence: float) -> List[Dict[str, Any]]:
    """Trova fino a 2 posizioni in cui una persona resta ferma per gran parte della live: le webcam."""
    regions: List[Dict[str, Any]] = []
    remaining = list(boxes)
    for _ in range(2):
        if len(remaining) < 12:
            break
        arr = np.array(remaining, dtype=np.float64)
        cx, cy = (arr[:, 1] + arr[:, 3]) / 2, (arr[:, 2] + arr[:, 4]) / 2
        heat, xe, ye = np.histogram2d(cx, cy, bins=[32, 18], range=[[0, 1], [0, 1]])
        ix, iy = np.unravel_index(int(np.argmax(heat)), heat.shape)
        px, py = (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2
        member = (np.abs(cx - px) < 0.06) & (np.abs(cy - py) < 0.09)
        if member.sum() < 12:
            break
        # ricentra sul baricentro reale e riseleziona
        px, py = float(np.median(cx[member])), float(np.median(cy[member]))
        member = (np.abs(cx - px) < 0.06) & (np.abs(cy - py) < 0.09)
        samples = set(arr[member, 0].astype(int).tolist())
        presence = len(samples) / max(1, n_samples)
        spread = float(np.std(cx[member]) + np.std(cy[member]))
        if presence >= min_presence and spread < 0.06:
            box = [float(np.percentile(arr[member, 1], 20)), float(np.percentile(arr[member, 2], 20)),
                   float(np.percentile(arr[member, 3], 80)), float(np.percentile(arr[member, 4], 80))]
            regions.append({"person": box, "presence": round(presence, 3), "center": [px, py]})
        remaining = [b for b, m in zip(remaining, member) if not m]
    return regions


def _refine_rect_by_edges(ex: np.ndarray, ey: np.ndarray, person: List[float]) -> Tuple[List[float], bool]:
    """
    Allarga il box della persona fino ai bordi fissi del riquadro webcam. ex/ey sono le medie nel tempo dei
    gradienti orizzontali/verticali: il video di gara cambia e si annulla, la cornice di un riquadro no.
    """
    hh, ww = ex.shape
    x1, y1, x2, y2 = person
    bw, bh = x2 - x1, y2 - y1
    r0, r1 = int(clamp(y1, 0, 1) * hh), max(int(clamp(y2, 0, 1) * hh), int(clamp(y1, 0, 1) * hh) + 2)
    c0, c1 = int(clamp(x1, 0, 1) * ww), max(int(clamp(x2, 0, 1) * ww), int(clamp(x1, 0, 1) * ww) + 2)
    col_profile = ex[r0:r1, :].mean(axis=0)
    row_profile = ey[:, c0:c1].mean(axis=1)
    med_c, med_r = float(np.median(ex)) + 1e-6, float(np.median(ey)) + 1e-6

    def nearest_edge(profile: np.ndarray, start: int, stop: int, step: int, med: float) -> Optional[int]:
        idx = list(range(start, stop, step))
        if not idx:
            return None
        vals = profile[idx]
        peak = float(vals.max())
        if peak < 3.0 * med:
            return None
        thr = max(2.5 * med, 0.6 * peak)
        for i, v in zip(idx, vals):
            if v >= thr:
                return i
        return None

    left = nearest_edge(col_profile, c0, max(-1, int((x1 - 1.1 * bw) * ww) - 1), -1, med_c)
    right = nearest_edge(col_profile, min(c1, ww - 1), min(ww, int((x2 + 1.1 * bw) * ww) + 1), 1, med_c)
    top = nearest_edge(row_profile, r0, max(-1, int((y1 - 0.9 * bh) * hh) - 1), -1, med_r)
    bottom = nearest_edge(row_profile, max(0, min(r1, hh - 1) - int(0.1 * bh * hh)),
                          min(hh, int((y2 + 0.6 * bh) * hh) + 1), 1, med_r)
    found = sum(e is not None for e in (left, right, top, bottom))
    # dove il bordo non si trova: margine standard attorno alla persona
    rx1 = left / ww if left is not None else x1 - 0.45 * bw
    rx2 = (right + 1) / ww if right is not None else x2 + 0.45 * bw
    ry1 = top / hh if top is not None else y1 - 0.22 * bh
    ry2 = (bottom + 1) / hh if bottom is not None else y2 + 0.05 * bh
    rect = [clamp(rx1, 0, 1), clamp(ry1, 0, 1), clamp(rx2, 0, 1), clamp(ry2, 0, 1)]
    return rect, found >= 3


def run_global_scan(info: VideoInfo, screener: Any, log_path: Path, step_s: float = 2.0,
                    quiet: bool = False) -> Dict[str, Any]:
    """
    Scansione dell'intera live a bassa risoluzione (solo keyframe: e' veloce anche su 3 ore). Produce:
      - posizione della webcam del telecronista (persona stabile + cornice fissa)
      - maschera delle grafiche fisse (HUD, classifica, loghi): sostituisce i margini fissi del v10
      - linea del tempo di movimento e veicoli: individua schermate statiche e persone a tutto schermo
    """
    sw, sh = 640, even(640 * info.height / info.width)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-skip_frame", "nokey",
           "-i", str(info.path), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", f"fps=1/{step_s:g},scale={sw}:{sh}:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_bytes = sw * sh * 3
    motion: List[float] = []
    veh: List[int] = []
    big_person: List[int] = []
    person_boxes: List[Tuple[int, float, float, float, float]] = []
    gw, gh = 160, even(160 * info.height / info.width)
    static_count = np.zeros((gh, gw), np.float32)
    dyn_frames = 0
    ex_sum = np.zeros((gh * 2, gw * 2), np.float32)
    ey_sum = np.zeros((gh * 2, gw * 2), np.float32)
    prev_gray: Optional[np.ndarray] = None
    last_motion = 0.0
    preview_frame: Optional[np.ndarray] = None
    k = 0
    t_start = time.time()
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=lf,
                                bufsize=1024 * 1024 * 4, creationflags=_no_window_flags())
        try:
            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
                dets = screener.infer(frame)
                n_veh = 0
                has_big = 0
                for cls, conf, x1, y1, x2, y2 in dets:
                    if cls == PERSON_CLASS:
                        bh = y2 - y1
                        if 0.05 <= bh <= 0.75 and conf >= 0.35:
                            person_boxes.append((k, x1, y1, x2, y2))
                        if bh > 0.45:
                            has_big = 1
                    elif cls in VEHICLE_CLASSES:
                        n_veh += 1
                veh.append(n_veh)
                big_person.append(has_big)
                gray = cv2.cvtColor(cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    d = cv2.absdiff(gray, prev_gray)
                    m = float(d.mean())
                    if m < 0.02:
                        m = last_motion       # frame duplicato dal filtro fps (GOP lungo): non e' un'informazione
                    else:
                        last_motion = m
                        if m > 4.0:
                            static_count += (d < 7)
                            dyn_frames += 1
                    motion.append(m)
                else:
                    motion.append(0.0)
                prev_gray = gray
                g2 = cv2.cvtColor(cv2.resize(frame, (gw * 2, gh * 2), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2GRAY).astype(np.float32)
                ex_sum += np.abs(cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3))
                ey_sum += np.abs(cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3))
                if preview_frame is None and any(d[0] == PERSON_CLASS for d in dets) and k * step_s > 120:
                    preview_frame = frame.copy()
                k += 1
                if not quiet and k % 900 == 0:
                    log(f"    ...scansione NPU a {format_time(k * step_s)}")
        finally:
            try:
                proc.stdout.close()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
    n = max(1, k)
    overlay = np.zeros((gh, gw), bool)
    if dyn_frames >= 20:
        ratio = static_count / dyn_frames
        mask = (ratio > 0.80).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if mask.mean() < 0.55:      # oltre meta' schermo "fisso" significa stima inaffidabile: meglio non mascherare
            overlay = mask.astype(bool)

    webcams: List[Dict[str, Any]] = []
    for region in _cluster_person_boxes(person_boxes, n, min_presence=0.20):
        rect, framed = _refine_rect_by_edges(ex_sum / n, ey_sum / n, region["person"])
        if (rect[2] - rect[0]) < 0.06 or (rect[3] - rect[1]) < 0.06:
            continue
        if (rect[2] - rect[0]) > 0.6 and (rect[3] - rect[1]) > 0.6:
            continue                 # persona a tutto schermo per tutta la live: non e' un riquadro webcam
        webcams.append({"rect": [round(v, 4) for v in rect], "person": [round(v, 4) for v in region["person"]],
                        "presence": region["presence"], "framed": bool(framed)})

    # Zone non di gara viste dall'immagine: schermata ferma senza auto, oppure persone a tutto schermo senza auto.
    zones: List[List[float]] = []
    mot = np.array(motion, dtype=np.float32)
    vh = np.array(veh, dtype=np.int32)
    bp = np.array(big_person, dtype=np.int32)
    for cond, min_len in (((mot < 2.0) & (vh == 0), 20.0), ((bp == 1) & (vh == 0), 30.0)):
        run_start = None
        for i, flag in enumerate(list(cond) + [False]):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                if (i - run_start) * step_s >= min_len:
                    zones.append([run_start * step_s, i * step_s])
                run_start = None

    preview_b64 = ""
    if preview_frame is not None and webcams:
        pv = preview_frame.copy()
        for wc in webcams:
            x1, y1, x2, y2 = wc["rect"]
            cv2.rectangle(pv, (int(x1 * sw), int(y1 * sh)), (int(x2 * sw), int(y2 * sh)), (0, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", pv, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            preview_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "step": step_s, "samples": k, "seconds": round(time.time() - t_start, 1),
        "device": getattr(screener, "device", "?"),
        "motion": [round(float(v), 2) for v in motion], "veh": [int(v) for v in veh],
        "webcams": webcams, "static_zones": zones,
        "overlay_shape": [gh, gw], "overlay": np.packbits(overlay).tolist(),
        "preview_b64": preview_b64,
    }


def unpack_overlay(scan: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not scan or not scan.get("overlay"):
        return None
    gh, gw = scan["overlay_shape"]
    bits = np.unpackbits(np.array(scan["overlay"], dtype=np.uint8))[:gh * gw]
    mask = bits.reshape(gh, gw).astype(bool)
    return mask if mask.any() else None


def parse_box_arg(text: str) -> Optional[List[float]]:
    """--webcam-box "x,y,w,h" in percentuale del frame (es. "74,4,24,26") -> [x1, y1, x2, y2] normalizzati."""
    try:
        x, y, w, h = [float(v) for v in re.split(r"[,;\s]+", text.strip()) if v]
    except ValueError:
        raise RaceClipperError('--webcam-box: usa "x,y,larghezza,altezza" in percentuale, es. "74,4,24,26"')
    if max(x + w, y + h) <= 1.0001:      # gia' frazioni 0..1
        x, y, w, h = x * 100, y * 100, w * 100, h * 100
    rect = [clamp(x / 100, 0, 1), clamp(y / 100, 0, 1), clamp((x + w) / 100, 0, 1), clamp((y + h) / 100, 0, 1)]
    if rect[2] - rect[0] < 0.03 or rect[3] - rect[1] < 0.03:
        raise RaceClipperError("--webcam-box: riquadro troppo piccolo.")
    return rect


class WebcamWatcher:
    """
    Thread che usa la NPU durante l'analisi profonda: riceve un frame ogni mezzo secondo e verifica se nel
    riquadro webcam c'e' davvero una persona. Lavora in parallelo alla RTX, che intanto traccia le auto.
    """

    def __init__(self, screener: Any, rect: List[float]):
        self.screener, self.rect = screener, rect
        self.q: "queue.Queue[Optional[Tuple[int, np.ndarray]]]" = queue.Queue(maxsize=8)
        self.hits: List[Tuple[int, bool]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        x1, y1, x2, y2 = self.rect
        while True:
            item = self.q.get()
            if item is None:
                return
            idx, frame = item
            present = False
            try:
                for cls, conf, a1, b1, a2, b2 in self.screener.infer(frame):
                    if cls != PERSON_CLASS:
                        continue
                    cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
                    if x1 - 0.02 <= cx <= x2 + 0.02 and y1 - 0.02 <= cy <= y2 + 0.02:
                        present = True
                        break
            except Exception:
                present = True      # un errore di inferenza non deve far sparire la webcam dal montaggio
            self.hits.append((idx, present))

    def push(self, idx: int, frame: np.ndarray) -> None:
        try:
            self.q.put_nowait((idx, frame))
        except queue.Full:
            pass                    # la NPU e' indietro: salta un campione, non rallenta la RTX

    def finish(self) -> List[Tuple[int, bool]]:
        self.q.put(None)
        self.thread.join(timeout=60)
        return sorted(self.hits)


# ---------------------------------------------------------------------------------------------------------
# Agenti Ollama: schema JSON imposto, coercizione dei tipi, validazione, retry con correzione
# ---------------------------------------------------------------------------------------------------------
def extract_json(text: str) -> Optional[Any]:
    """Estrae il JSON anche se il modello lo incornicia in ```json ... ``` o aggiunge testo attorno."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    closer = "}" if t[start] == "{" else "]"
    end = t.rfind(closer)
    while end > start:
        try:
            return json.loads(t[start:end + 1])
        except Exception:
            end = t.rfind(closer, start, end)
    return None


_TRUE = {"true", "vero", "si", "sì", "yes", "1"}
_FALSE = {"false", "falso", "no", "0"}


def coerce_validate(value: Any, schema: Dict[str, Any], path: str = "risposta") -> Tuple[Any, List[str]]:
    """
    Porta la risposta del modello dentro lo schema: "8/10" -> 8, "si" -> True, "a, b" -> ["a", "b"],
    numeri fuori intervallo riportati nei limiti. Cio' che non e' recuperabile (campo mancante, tipo
    sbagliato, valore fuori elenco) diventa un errore leggibile da rimandare al modello.
    """
    errors: List[str] = []
    typ = schema.get("type")

    if typ == "object":
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
        if not isinstance(value, dict):
            return None, [f"{path}: atteso un oggetto JSON"]
        out: Dict[str, Any] = {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        for key, sub in props.items():
            if key in value and value[key] is not None:
                v, errs = coerce_validate(value[key], sub, f"{path}.{key}")
                errors.extend(errs)
                out[key] = v
            elif key in required:
                errors.append(f"{path}.{key}: campo obbligatorio mancante")
            elif "default" in sub:
                out[key] = sub["default"]
        return out, errors

    if typ == "array":
        if isinstance(value, str):
            value = [p.strip() for p in re.split(r"[,;\n]", value) if p.strip()]
        elif isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return None, [f"{path}: attesa una lista"]
        items_schema = schema.get("items", {})
        out_list = []
        for i, item in enumerate(value):
            v, errs = coerce_validate(item, items_schema, f"{path}[{i}]")
            if errs and items_schema.get("type") == "string":
                continue                      # elementi non testuali in una lista di stringhe: si scartano
            errors.extend(errs)
            out_list.append(v)
        if "maxItems" in schema:
            out_list = out_list[: int(schema["maxItems"])]
        if len(out_list) < int(schema.get("minItems", 0)):
            errors.append(f"{path}: servono almeno {schema['minItems']} elementi")
        return out_list, errors

    if typ in ("integer", "number"):
        num: Optional[float]
        if isinstance(value, bool):
            num = None
        elif isinstance(value, (int, float)):
            num = float(value)
        elif isinstance(value, str):
            num = parse_timestamp_value(value) if schema.get("x-time") else None
            if num is None:
                m = re.search(r"-?\d+(?:[.,]\d+)?", value)
                num = float(m.group(0).replace(",", ".")) if m else None
        else:
            num = None
        if num is None or math.isnan(num) or math.isinf(num):
            return None, [f"{path}: atteso un numero, ricevuto {json.dumps(value, ensure_ascii=False)[:40]}"]
        if "minimum" in schema:
            num = max(float(schema["minimum"]), num)
        if "maximum" in schema:
            num = min(float(schema["maximum"]), num)
        return (int(round(num)) if typ == "integer" else float(num)), errors

    if typ == "boolean":
        if isinstance(value, bool):
            return value, errors
        if isinstance(value, (int, float)):
            return bool(value), errors
        s = str(value).strip().lower()
        if s in _TRUE:
            return True, errors
        if s in _FALSE:
            return False, errors
        return None, [f"{path}: atteso true o false"]

    if typ == "string":
        if isinstance(value, (list, dict)):
            if isinstance(value, list) and all(isinstance(x, (str, int, float)) for x in value):
                value = ", ".join(str(x) for x in value)
            else:
                return None, [f"{path}: attesa una stringa"]
        s = str(value).strip()
        if "enum" in schema:
            norm = re.sub(r"[\s\-]+", "_", strip_accents(s).upper())
            table = {re.sub(r"[\s\-]+", "_", strip_accents(e).upper()): e for e in schema["enum"]}
            if norm in table:
                return table[norm], errors
            for key, original in table.items():
                if key in norm or (len(norm) >= 4 and norm in key):
                    return original, errors
            return None, [f"{path}: valore non ammesso, usa uno fra {', '.join(schema['enum'])}"]
        if len(s) < int(schema.get("minLength", 0)):
            errors.append(f"{path}: testo vuoto o troppo corto")
        return s, errors

    return value, errors


def wire_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Schema inviato a Ollama: solo le parole chiave che ogni versione sa tradurre in grammatica.
    Intervalli, lunghezze e valori predefiniti restano compito di coerce_validate."""
    out: Dict[str, Any] = {}
    for key in ("type", "enum", "required"):
        if key in schema:
            out[key] = schema[key]
    if "properties" in schema:
        out["properties"] = {k: wire_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = wire_schema(schema["items"])
    return out


class OllamaClient:
    CTX_LADDER = (4096, 8192, 12288, 16384, 24576, 32768)

    def __init__(self, base_url: str, allow_pull: bool = True, timeout: float = 900.0):
        self.base = base_url.rstrip("/")
        self.allow_pull = allow_pull
        self.timeout = timeout
        self.session = requests.Session()
        self._no_think: set = set()
        self._plain_json: set = set()
        self.stats = {"calls": 0, "retries": 0, "failed": 0}

    # -- server e modelli ---------------------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            return self.session.get(self.base + "/api/tags", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def ensure_server(self) -> None:
        if self.ping():
            return
        log(" -> Ollama non risponde: avvio 'ollama serve' e attendo...")
        try:
            subprocess.Popen(["ollama", "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=_no_window_flags())
        except FileNotFoundError:
            raise RaceClipperError("Ollama non e' installato o non e' nel PATH (https://ollama.com/download).")
        for _ in range(45):
            time.sleep(1.0)
            if self.ping():
                return
        raise RaceClipperError(f"Il server Ollama non si e' avviato su {self.base}.")

    def list_models(self) -> List[str]:
        try:
            r = self.session.get(self.base + "/api/tags", timeout=10)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    @staticmethod
    def _matches(wanted: str, installed: str) -> bool:
        if wanted == installed:
            return True
        if ":" not in wanted:
            return installed.split(":")[0] == wanted
        return False

    def has_model(self, name: str, installed: Optional[List[str]] = None) -> bool:
        installed = self.list_models() if installed is None else installed
        return any(self._matches(name, m) for m in installed)

    def pull(self, name: str) -> bool:
        log(f" -> Scarico il modello '{name}' da Ollama (solo la prima volta)...")
        try:
            r = self.session.post(self.base + "/api/pull", json={"model": name, "stream": True},
                                  stream=True, timeout=(10, 3600))
            if r.status_code != 200:
                warn(f"download di '{name}' rifiutato: {r.text[:200]}")
                return False
            last_pct = -10
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("error"):
                    warn(f"download di '{name}' fallito: {str(ev['error'])[:200]}")
                    return False
                total, done = ev.get("total"), ev.get("completed")
                if total and done:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 10:
                        log(f"    {name}: {pct}%")
                        last_pct = pct
            return self.has_model(name)
        except requests.RequestException as e:
            warn(f"download di '{name}' interrotto: {e}")
            return False

    def resolve_model(self, role: str, prefs: Sequence[str], override: str = "") -> str:
        """Primo modello utilizzabile nell'ordine di preferenza. Mai varianti cloud: il tool resta 100% locale."""
        candidates = [override] if override else [p for p in prefs if "cloud" not in p]
        installed = self.list_models()

        def installed_name(name: str, pool: List[str]) -> str:
            """Nome completo di tag come lo conosce Ollama ('qwen3.8' -> 'qwen3.8:27b')."""
            exact = [m for m in pool if m == name]
            if exact:
                return exact[0]
            same = sorted(m for m in pool if self._matches(name, m))
            return same[0] if same else ""

        for name in candidates:
            found = installed_name(name, installed)
            if found:
                return found
            if self.allow_pull and self.pull(name):
                return installed_name(name, self.list_models()) or name
        if override:
            raise RaceClipperError(f"Modello '{override}' ({role}) non disponibile in Ollama.")
        for name in candidates:           # ultimo tentativo: qualunque modello della lista gia' installato
            found = installed_name(name, self.list_models())
            if found:
                return found
        raise RaceClipperError(
            f"Nessun modello disponibile per il ruolo '{role}'. Installa uno fra: {', '.join(candidates)} "
            f"(es. ollama pull {candidates[0]}) oppure indica un modello con l'opzione dedicata.")

    def loaded_models(self) -> List[str]:
        try:
            r = self.session.get(self.base + "/api/ps", timeout=10)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def unload(self, model: str) -> None:
        try:
            self.session.post(self.base + "/api/generate", json={"model": model, "prompt": "", "keep_alive": 0},
                              timeout=30)
        except requests.RequestException:
            pass

    def unload_all(self, wait_s: float = 25.0) -> None:
        """Regola d'oro sulla VRAM: prima di caricare YOLO/Whisper o un altro LLM, Ollama deve essere vuoto."""
        loaded = self.loaded_models()
        for m in loaded:
            self.unload(m)
        t0 = time.time()
        while loaded and time.time() - t0 < wait_s:
            time.sleep(0.5)
            loaded = self.loaded_models()
        flush_vram()

    # -- chiamata strutturata -----------------------------------------------------------------------------
    def _think_value(self, model: str) -> Any:
        if model in self._no_think:
            return None
        return "low" if model.startswith("gpt-oss") else False

    def _ctx_for(self, estimate: int, floor: int = 0) -> int:
        for c in self.CTX_LADDER:
            if c >= max(estimate, floor):
                return c
        return self.CTX_LADDER[-1]

    def chat_json(self, model: str, system: str, user: str, schema: Dict[str, Any], *,
                  images: Optional[List[str]] = None, temperature: float = 0.2, num_predict: int = 900,
                  num_ctx: int = 0, keep_alive: Any = "20m", retries: int = 3, label: str = "") -> Optional[Dict[str, Any]]:
        """
        Restituisce un dict conforme allo schema oppure None (mai un valore di ripiego "ottimista").
        Gestisce: schema non supportato dal server (ripiega su format=json), prompt troncato dal contesto,
        risposta tagliata da num_predict, JSON non valido, tipi e campi sbagliati (retry con l'elenco errori).
        """
        base_messages: List[Dict[str, Any]] = [{"role": "system", "content": system},
                                               {"role": "user", "content": user}]
        if images:
            base_messages[1]["images"] = images
        messages = list(base_messages)
        estimate = int((len(system) + len(user)) / 2.6) + len(images or []) * 800 + num_predict + 300
        ctx = self._ctx_for(estimate, num_ctx)
        self.stats["calls"] += 1
        validation_round = 0
        wire = wire_schema(schema)
        for _ in range(retries + 4):
            fmt: Any = "json" if model in self._plain_json else wire
            payload: Dict[str, Any] = {
                "model": model, "messages": messages, "stream": False, "format": fmt, "keep_alive": keep_alive,
                "options": {"temperature": temperature, "num_ctx": ctx, "num_predict": num_predict},
            }
            think = self._think_value(model)
            if think is not None:
                payload["think"] = think
            try:
                r = self.session.post(self.base + "/api/chat", json=payload, timeout=(10, self.timeout))
            except requests.RequestException as e:
                warn(f"{label or model}: richiesta a Ollama fallita ({str(e)[:100]}), riprovo")
                self.stats["retries"] += 1
                time.sleep(2.0)
                continue
            if r.status_code != 200:
                body = r.text[:400].lower()
                if "think" in body and model not in self._no_think:
                    self._no_think.add(model)
                    continue
                if isinstance(fmt, dict) and any(k in body for k in ("format", "schema", "grammar")):
                    self._plain_json.add(model)
                    continue
                warn(f"{label or model}: Ollama ha risposto {r.status_code}: {r.text[:160]}")
                self.stats["retries"] += 1
                time.sleep(1.0)
                continue
            try:
                data = r.json()
            except ValueError:
                self.stats["retries"] += 1
                continue
            msg = data.get("message") or {}
            raw = (msg.get("content") or "").strip()
            if not raw and msg.get("thinking"):
                raw = str(msg["thinking"]).strip()
            if int(data.get("prompt_eval_count") or 0) >= ctx - 32 and ctx < self.CTX_LADDER[-1]:
                ctx = self._ctx_for(ctx + 1)          # il prompt ha riempito il contesto: Ollama lo ha troncato
                self.stats["retries"] += 1
                continue
            if data.get("done_reason") == "length" and num_predict < 6000:
                num_predict = int(num_predict * 1.8)   # risposta tagliata: piu' spazio e si ripete
                ctx = self._ctx_for(ctx + num_predict // 2)
                self.stats["retries"] += 1
                continue
            obj = extract_json(raw)
            if obj is None:
                errors = ["la risposta non e' JSON valido"]
            else:
                obj, errors = coerce_validate(obj, schema)
            if not errors:
                return obj
            validation_round += 1
            self.stats["retries"] += 1
            if validation_round > retries:
                break
            fix = ("La tua risposta non rispetta lo schema richiesto. Errori:\n- " + "\n- ".join(errors[:8]) +
                   "\nRispondi di nuovo con il SOLO oggetto JSON completo e corretto: numeri come numeri "
                   "(non stringhe), tutti i campi obbligatori presenti, nessun testo fuori dal JSON.")
            messages = base_messages + [{"role": "assistant", "content": raw[:6000]},
                                        {"role": "user", "content": fix}]
        self.stats["failed"] += 1
        warn(f"{label or model}: nessuna risposta valida dopo i tentativi previsti")
        return None


def schema_hint(schema: Dict[str, Any]) -> str:
    """Lo schema viene anche scritto nel prompt: aiuta i modelli e resta valido se il server usa format=json."""
    return json.dumps(schema, ensure_ascii=False)


def image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------------------------------------
# Stato condiviso della run
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    args: argparse.Namespace
    out_dir: Path
    work_dir: Path
    logs_dir: Path
    cache: StageCache
    info: Optional[VideoInfo] = None
    encoder: str = "libx264"
    fast_encode: bool = False
    llm: Optional[OllamaClient] = None
    models: Dict[str, str] = field(default_factory=dict)
    feats: Optional[AudioFeatures] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    words: Optional[WordIndex] = None
    vocab: Dict[str, List[str]] = field(default_factory=dict)
    scan: Optional[Dict[str, Any]] = None
    overlay: Optional[np.ndarray] = None
    webcam: Optional[Dict[str, Any]] = None
    screener: Any = None
    detector_factory: Optional[Callable[[], Any]] = None
    race: Dict[str, Any] = field(default_factory=dict)
    blacklist: List[Tuple[float, float, str]] = field(default_factory=list)
    phases: List[Dict[str, Any]] = field(default_factory=list)

    def log_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


# ---------------------------------------------------------------------------------------------------------
# Vocabolario della telecronaca: serve a distinguere un nome "sentito davvero" da un nome inventato
# ---------------------------------------------------------------------------------------------------------
def build_vocab(words: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    vocab: Dict[str, set] = {}
    for w in words:
        t = strip_accents(w["w"]).replace("'", "")
        if len(t) >= 3:
            vocab.setdefault(t[0], set()).add(t)
    return {k: sorted(v) for k, v in vocab.items()}


def name_supported(token: str, vocab: Dict[str, List[str]], allowed: set) -> bool:
    """Vero se il nome compare nella telecronaca (anche con l'ortografia incerta di Whisper) o nei dati dell'utente."""
    t = strip_accents(token.lower()).replace("'", "")
    t = re.sub(r"[^a-z0-9]", "", t)
    if len(t) < 3 or t in allowed:
        return True
    bucket = vocab.get(t[0], [])
    if t in bucket:
        return True
    near = [w for w in bucket if abs(len(w) - len(t)) <= 2]
    return bool(difflib.get_close_matches(t, near, n=1, cutoff=0.8))


def phrase_supported(phrase: str, vocab: Dict[str, List[str]], allowed: set) -> bool:
    toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ0-9']+", phrase) if len(t) >= 4 and not t.isdigit()]
    if not toks:
        return bool(phrase.strip())
    hits = sum(name_supported(t, vocab, allowed) for t in toks)
    return hits >= max(1, math.ceil(len(toks) * 0.6))


# ---------------------------------------------------------------------------------------------------------
# Zone escluse: spot, interviste, schermate fisse
# ---------------------------------------------------------------------------------------------------------
DEFAULT_AD_KEYWORDS = [
    "hai mai provato l'adrenalina", "box 33", "box33", "sette postazioni professionali",
    "7 postazioni professionali", "eventi aziendali o tra amici", "il motore è acceso", "sei pronto a partire",
]
DEFAULT_INTERVIEW_KEYWORDS = [
    "fase finale di questo evento", "quindi appunto le interviste", "prossima intervista",
    "ci tenevo a ringraziare", "i nostri sponsor", "riavviare il computer", "domanda per te",
    "ti lasciamo andare", "grazie mille per l'intervista", "cominciamo quindi da subito con il primo in classifica",
]


def load_keyword_file(path: str) -> Tuple[List[str], List[str]]:
    ads, interviews = list(DEFAULT_AD_KEYWORDS), list(DEFAULT_INTERVIEW_KEYWORDS)
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            ads += [str(x).lower() for x in blob.get("spot", blob.get("ads", []))]
            interviews += [str(x).lower() for x in blob.get("interviste", blob.get("interviews", []))]
        except Exception as e:
            warn(f"file parole chiave non leggibile ({e}): uso l'elenco predefinito")
    return ads, interviews


def merge_zones(zones: List[Tuple[float, float, str]], gap: float = 10.0) -> List[Tuple[float, float, str]]:
    merged: List[Tuple[float, float, str]] = []
    for a, b, why in sorted(zones):
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + gap:
            pa, pb, pwhy = merged[-1]
            merged[-1] = (pa, max(pb, b), pwhy if why in pwhy else f"{pwhy}+{why}")
        else:
            merged.append((a, b, why))
    return merged


def rule_blacklist(segments: List[Dict[str, Any]], skip_intro: float, ads: List[str],
                   interviews: List[str]) -> List[Tuple[float, float, str]]:
    zones: List[Tuple[float, float, str]] = []
    if skip_intro > 0:
        zones.append((0.0, skip_intro, "intro"))
    counts: Dict[str, int] = {}
    for s in segments:
        norm = re.sub(r"[^a-z0-9]", "", s["text"].lower())
        if len(norm) >= 22:
            counts[norm[:35]] = counts.get(norm[:35], 0) + 1
    repeated = {k for k, c in counts.items() if c >= 3}
    for s in segments:
        low = s["text"].lower()
        prefix = re.sub(r"[^a-z0-9]", "", low)[:35]
        if prefix in repeated or any(k in low for k in ads):
            zones.append((max(0.0, s["start"] - 8.0), s["end"] + 8.0, "spot"))
        elif any(k in low for k in interviews):
            zones.append((max(0.0, s["start"] - 8.0), s["end"] + 8.0, "intervista"))
    return merge_zones(zones)


def subtract_zones(a: float, b: float, zones: List[Tuple[float, float, str]]) -> List[Tuple[float, float]]:
    """Parti di [a, b] che restano fuori dalle zone escluse."""
    pieces = [(a, b)]
    for za, zb, _ in zones:
        nxt: List[Tuple[float, float]] = []
        for pa, pb in pieces:
            if zb <= pa or za >= pb:
                nxt.append((pa, pb))
                continue
            if za > pa:
                nxt.append((pa, za))
            if zb < pb:
                nxt.append((zb, pb))
        pieces = nxt
    return [(pa, pb) for pa, pb in pieces if pb - pa > 0.05]


def zone_at(t: float, zones: List[Tuple[float, float, str]]) -> str:
    for a, b, why in zones:
        if a <= t <= b:
            return why
    return ""


# ---------------------------------------------------------------------------------------------------------
# AGENTE 1 - Contesto gara (pista, campionato, auto, piloti) e fasi della live
# ---------------------------------------------------------------------------------------------------------
CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "circuit": {"type": "string", "default": ""},
        "championship": {"type": "string", "default": ""},
        "category": {"type": "string", "default": ""},
        "cars": {"type": "array", "items": {"type": "string"}, "maxItems": 12, "default": []},
        "drivers": {"type": "array", "items": {"type": "string"}, "maxItems": 40, "default": []},
    },
    "required": ["circuit", "championship", "drivers"],
}


def extract_race_context(ctx: Ctx) -> Dict[str, Any]:
    args = ctx.args
    user_drivers = [d.strip() for d in (args.piloti or "").split(",") if d.strip()]
    race = {"circuit": (args.pista or "").strip(), "championship": (args.campionato or "").strip(),
            "category": "", "cars": [], "drivers": list(user_drivers)}
    segs = [s for s in ctx.segments if not zone_at(s["start"], ctx.blacklist)]
    if not segs:
        return race
    # Estratti da tutta la gara (il v10 leggeva solo i primi minuti).
    n_ex = 10
    excerpts: List[str] = []
    for i in range(n_ex):
        k = int(i * len(segs) / n_ex)
        chunk, size = [], 0
        for s in segs[k:k + 60]:
            chunk.append(s["text"])
            size += len(s["text"])
            if size > 1100:
                break
        if chunk:
            excerpts.append(f"[{format_time(segs[k]['start'])}] " + " ".join(chunk))
    system = ("Sei un analista di telecronache di motorsport e sim racing. Estrai solo cio' che e' scritto negli "
              "estratti. Se un dato non compare lascia la stringa vuota o la lista vuota. Non inventare nomi. "
              "La trascrizione e' automatica: i nomi propri possono avere piccoli errori di ortografia.")
    user = (
        "ESTRATTI DELLA TELECRONACA (da tutta la diretta):\n" + "\n".join(excerpts) +
        f"\n\nINDICAZIONI DELL'UTENTE: pista='{race['circuit']}', campionato='{race['championship']}', "
        f"piloti='{', '.join(user_drivers)}'\n\n"
        "Compila: circuit (nome del circuito), championship (campionato o evento), category (categoria o simulatore, "
        "es. GT3, Assetto Corsa Competizione), cars (modelli di auto citati), drivers (cognomi dei piloti citati "
        "piu' volte come concorrenti in pista).\n"
        f"Rispondi SOLO con JSON conforme a questo schema: {schema_hint(CONTEXT_SCHEMA)}")
    res = ctx.llm.chat_json(ctx.models["text"], system, user, CONTEXT_SCHEMA, temperature=0.0,
                            num_predict=700, label="contesto gara")
    if res:
        allowed = {strip_accents(x.lower()) for x in user_drivers}
        if not race["circuit"] and res.get("circuit") and phrase_supported(res["circuit"], ctx.vocab, allowed):
            race["circuit"] = res["circuit"].strip()
        if not race["championship"] and res.get("championship") and \
                phrase_supported(res["championship"], ctx.vocab, allowed):
            race["championship"] = res["championship"].strip()
        if res.get("category") and phrase_supported(res["category"], ctx.vocab, allowed):
            race["category"] = res["category"].strip()
        race["cars"] = [c.strip() for c in res.get("cars", []) if c.strip() and phrase_supported(c, ctx.vocab, allowed)][:10]
        seen = {strip_accents(d.lower()) for d in race["drivers"]}
        for d in res.get("drivers", []):
            d = d.strip()
            key = strip_accents(d.lower())
            # un pilota vale solo se il suo cognome compare davvero nella telecronaca (anche con la grafia di Whisper)
            if d and key not in seen and name_supported(d.split()[-1], ctx.vocab, set()):
                race["drivers"].append(d)
                seen.add(key)
    return race


PHASES = ["GARA", "QUALIFICA", "PARTENZA", "GIRO_FORMAZIONE", "PRE_GARA", "PUBBLICITA", "INTERVISTE",
          "PAUSA_CLASSIFICA", "POST_GARA"]
HARD_NON_RACE = {"PUBBLICITA", "INTERVISTE", "PAUSA_CLASSIFICA", "POST_GARA"}
PHASE_SCHEMA = {
    "type": "object",
    "properties": {"blocks": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "phase": {"type": "string", "enum": PHASES}},
        "required": ["id", "phase"]}}},
    "required": ["blocks"],
}


def classify_phases(ctx: Ctx, block_s: float = 120.0) -> List[Dict[str, Any]]:
    """Fasi della live a blocchi di 2 minuti: testo della telecronaca + cio' che la NPU ha visto nel blocco."""
    total = ctx.info.duration
    blocks: List[Dict[str, Any]] = []
    scan = ctx.scan or {}
    step = float(scan.get("step", 2.0))
    veh = np.array(scan.get("veh", []), dtype=np.float32)
    mot = np.array(scan.get("motion", []), dtype=np.float32)
    t = 0.0
    while t < total:
        t1 = min(total, t + block_s)
        txt = text_in_range(ctx.segments, t, t1)
        a, b = int(t / step), int(math.ceil(t1 / step))
        veh_ratio = float((veh[a:b] >= 1).mean()) if len(veh[a:b]) else -1.0
        static_ratio = float((mot[a:b] < 2.0).mean()) if len(mot[a:b]) else -1.0
        blocks.append({"id": len(blocks), "t0": t, "t1": t1, "text": txt, "veh_ratio": veh_ratio,
                       "static_ratio": static_ratio, "phase": "GARA"})
        t = t1
    system = ("Classifichi le fasi di una diretta di gara (motorsport / sim racing) commentata in italiano. "
              "Per ogni blocco scegli UNA fase. GARA e QUALIFICA solo quando le auto sono in pista a tempo. "
              "GIRO_FORMAZIONE comprende outlap e giri di riscaldamento. PAUSA_CLASSIFICA e' la lettura di "
              "classifiche o schermate ferme. INTERVISTE e POST_GARA sono le chiacchiere dopo la bandiera a scacchi.")
    batch = 10
    for i in range(0, len(blocks), batch):
        part = blocks[i:i + batch]
        if all(len(b["text"]) < 20 for b in part):
            continue
        lines = []
        for b in part:
            txt = b["text"]
            if len(txt) > 620:
                txt = txt[:360] + " [...] " + txt[-240:]
            vis = ""
            if b["veh_ratio"] >= 0:
                vis = f" | auto visibili {int(b['veh_ratio'] * 100)}% | schermo fermo {int(b['static_ratio'] * 100)}%"
            lines.append(f"BLOCCO {b['id']} [{format_time(b['t0'])}-{format_time(b['t1'])}{vis}]: {txt or '(silenzio)'}")
        user = ("\n".join(lines) + f"\n\nFasi ammesse: {', '.join(PHASES)}.\n"
                f"Rispondi SOLO con JSON conforme a: {schema_hint(PHASE_SCHEMA)} (un elemento per ogni BLOCCO, stesso id).")
        res = ctx.llm.chat_json(ctx.models["text"], system, user, PHASE_SCHEMA, temperature=0.0,
                                num_predict=80 + 24 * len(part), label=f"fasi {i // batch + 1}")
        if not res:
            continue
        by_id = {int(x["id"]): x["phase"] for x in res.get("blocks", []) if "id" in x and "phase" in x}
        for b in part:
            if b["id"] in by_id:
                b["phase"] = by_id[b["id"]]
    return [{k: v for k, v in b.items() if k != "text"} for b in blocks]


def phase_zones(phases: List[Dict[str, Any]], total: float) -> List[Tuple[float, float, str]]:
    """
    Una fase "non di gara" diventa zona esclusa solo se e' corroborata: blocco vicino con la stessa natura
    oppure conferma visiva della NPU. Giro di formazione e pre-gara NON vengono esclusi qui: la partenza
    lanciata arriva subito dopo e la decisione spetta alla Verifica 1 sulla singola clip.
    """
    zones: List[Tuple[float, float, str]] = []
    n = len(phases)
    for i, b in enumerate(phases):
        if b["phase"] not in HARD_NON_RACE:
            continue
        neighbour = any(0 <= j < n and phases[j]["phase"] in HARD_NON_RACE for j in (i - 1, i + 1))
        visual = 0 <= b.get("veh_ratio", -1) < 0.25
        if neighbour or visual:
            zones.append((b["t0"], b["t1"], b["phase"].lower()))
    # coda della diretta: dopo due blocchi consecutivi di interviste/post gara nell'ultimo 40% non si torna in pista
    for i in range(n - 1):
        if phases[i]["t0"] > total * 0.6 and phases[i]["phase"] in ("INTERVISTE", "POST_GARA") and \
                phases[i + 1]["phase"] in ("INTERVISTE", "POST_GARA"):
            later_race = sum(p["phase"] in ("GARA", "PARTENZA", "QUALIFICA") for p in phases[i + 2:])
            if later_race <= 1:
                zones.append((phases[i]["t0"], total + 10.0, "post_gara"))
                break
    return zones


def phase_at(t: float, phases: List[Dict[str, Any]]) -> str:
    for b in phases:
        if b["t0"] <= t < b["t1"]:
            return b["phase"]
    return "GARA"


# ---------------------------------------------------------------------------------------------------------
# AGENTE 2 - Action Detector sui blocchi di telecronaca + scout indipendente sui picchi di voce
# ---------------------------------------------------------------------------------------------------------
ACTION_TYPES = ["CONTATTO", "PARTENZA", "SORPASSO", "DUELLO", "ERRORE_SALVATAGGIO", "POLE_LAP", "VOLATA"]
SCOUT_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "maxItems": 4, "items": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 3},
            "start": {"type": "number", "x-time": True},
            "end": {"type": "number", "x-time": True},
            "action_type": {"type": "string", "enum": ACTION_TYPES},
            "drivers": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "default": []},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "continues_after": {"type": "boolean", "default": False},
        },
        "required": ["summary", "start", "end", "action_type", "confidence"]}}},
    "required": ["clips"],
}
TYPE_BONUS = {"CONTATTO": 0.08, "PARTENZA": 0.06, "VOLATA": 0.05, "ERRORE_SALVATAGGIO": 0.04, "SORPASSO": 0.03}


def scout_with_llm(ctx: Ctx, chunk_s: float = 240.0, overlap_s: float = 30.0) -> List[Dict[str, Any]]:
    segs = ctx.segments
    if not segs:
        return []
    total = ctx.info.duration
    race = ctx.race
    where = " ".join(x for x in (race.get("championship"), "a " + race["circuit"] if race.get("circuit") else "") if x)
    system = (
        f"Sei il regista di una diretta di gara{(' (' + where + ')') if where else ''}. Leggi un blocco di telecronaca "
        "in italiano e individui SOLO i momenti in cui in pista succede qualcosa di concreto. "
        "CERCA, in ordine di priorita': 1) contatti, incidenti, uscite di pista, testacoda, salvataggi; "
        "2) partenze con bagarre alla prima curva; 3) sorpassi, duelli ruota a ruota, incroci di traiettoria, "
        "tre auto affiancate, difese dure, volate al traguardo; 4) giro da pole all'ultimo secondo. "
        "SCARTA: chiacchiere su assetto o regolamento, lettura di distacchi e classifiche, giri di formazione, "
        "pubblicita', interviste. Se nel blocco non succede nulla restituisci una lista vuota. "
        "Per ogni momento indica inizio e fine dell'azione INTERA cosi' come emergono dalla telecronaca, in secondi "
        "(i numeri fra parentesi quadre): dalla preparazione (si avvicina, prende la scia) fino a quando l'esito e' "
        "chiaro. Un duello che prosegue per piu' curve, con sorpasso e contro-sorpasso, e' UN solo momento, anche "
        "se dura un minuto o piu': non spezzarlo. Il taglio preciso verra' fatto dopo sui dati video. "
        "Se l'azione sta ancora proseguendo quando il blocco finisce, imposta continues_after a true. "
        "confidence da 0 a 100 = quanto sei sicuro che sia azione vera e non commento."
    )
    out: List[Dict[str, Any]] = []
    start = max(0.0, segs[0]["start"] - 1.0)
    idx = 0
    n_chunks = max(1, int(math.ceil((total - start) / (chunk_s - overlap_s))))
    while start < total:
        end = start + chunk_s
        idx += 1
        free = sum(b - a for a, b in subtract_zones(start, min(end, total), ctx.blacklist))
        chunk = [s for s in segs if s["end"] >= start and s["start"] <= end and not zone_at(s["start"], ctx.blacklist)]
        if free >= 30.0 and len(chunk) >= 4:
            lines = []
            for s in chunk:
                v = int(round(100 * float(ctx.feats.window(ctx.feats.voice, s["start"], s["end"]).max())))
                hits = len(ctx.words.in_range(ATTACK_PATTERNS + CONTACT_PATTERNS + SPIN_PATTERNS, "action",
                                              s["start"] - 0.05, s["end"]))
                tag = "AZIONE+URLO" if (v >= 75 and hits) else "BAGARRE" if hits >= 2 else ""
                lines.append(f"[{s['start']:.1f}-{s['end']:.1f} | voce {v}{' | ' + tag if tag else ''}] {s['text']}")
            user = (
                f"FASE STIMATA DEL BLOCCO: {phase_at((start + end) / 2, ctx.phases)}\n"
                f"PILOTI NOTI: {', '.join(race.get('drivers', [])[:30]) or 'non disponibili'}\n"
                f"RICHIESTA DELL'UTENTE: {ctx.args.prompt}\n\nTRASCRIZIONE:\n" + "\n".join(lines) +
                f"\n\nRispondi SOLO con JSON conforme a: {schema_hint(SCOUT_SCHEMA)}\n"
                "Scrivi 'summary' (cosa succede, in una frase) PRIMA dei numeri. Da 0 a 3 momenti.")
            res = ctx.llm.chat_json(ctx.models["text"], system, user, SCOUT_SCHEMA, temperature=0.1,
                                    num_predict=900, label=f"scout blocco {idx}/{n_chunks}")
            for c in (res or {}).get("clips", []):
                st, et = float(c["start"]), float(c["end"])
                if st < start - 20 or st > end + 20 or st <= 0:
                    continue             # tempi fuori dal blocco letto: allucinazione
                if et <= st:
                    et = st + 12.0
                et = min(et, st + max(150.0, float(ctx.args.max_dur)), total)
                out.append({
                    "start": round(max(0.0, st), 1), "end": round(et, 1), "types": [c["action_type"]],
                    "drivers": [d for d in c.get("drivers", []) if d], "summaries": [c["summary"]],
                    "confidence": int(c["confidence"]), "sources": ["telecronaca"],
                    "continues_after": bool(c.get("continues_after", False)) or et >= end - 4.0,
                    "story": [round(max(0.0, st), 1), round(et, 1)],
                })
            if idx % 10 == 0:
                log(f"    ...letti {idx}/{n_chunks} blocchi, {len(out)} momenti segnalati")
        start += chunk_s - overlap_s
    return out


def scout_audio_peaks(ctx: Ctx) -> List[Dict[str, Any]]:
    """Scout indipendente dal testo: i picchi di voce del telecronista fuori dalle zone escluse."""
    f = ctx.feats
    if f is None or len(f) < 50:
        return []
    k = max(1, int(round(2.0 / f.hop)))
    sm = np.convolve(f.voice, np.ones(k) / k, mode="same")
    valid = np.ones(len(sm), bool)
    for a, b, _ in ctx.blacklist:
        valid[f.idx(a):f.idx(b) + 1] = False
    if valid.sum() < 50:
        return []
    thr = float(np.percentile(sm[valid], 93))
    score = np.where(valid, sm, 0.0)
    max_n = max(8, int(ctx.info.duration / 240))
    guard = int(round(25.0 / f.hop))
    out: List[Dict[str, Any]] = []
    for _ in range(max_n):
        i = int(np.argmax(score))
        if score[i] < max(thr, 0.35):
            break
        t = i * f.hop
        out.append({"start": round(max(0.0, t - 9.0), 1), "end": round(min(ctx.info.duration, t + 7.0), 1),
                    "types": ["PICCO_VOCE"], "drivers": [], "summaries": [], "confidence": 40,
                    "sources": ["picco voce"], "continues_after": False})
        score[max(0, i - guard):i + guard] = 0.0
    return out


def merge_candidates(cands: List[Dict[str, Any]], max_len: float = 150.0,
                     near_gap: float = 6.0) -> List[Dict[str, Any]]:
    """
    Unisce le finestre sovrapposte o contigue (il v10 le scartava): un duello che attraversa il confine fra
    due blocchi di telecronaca resta un'unica candidata, e il trimming trovera' i confini reali.
    """
    out: List[Dict[str, Any]] = []
    for c in sorted(cands, key=lambda x: (x["start"], x["end"])):
        if out:
            p = out[-1]
            gap = c["start"] - p["end"]
            join_gap = max(15.0, near_gap) if p.get("continues_after") else near_gap
            union_len = max(p["end"], c["end"]) - p["start"]
            if gap <= join_gap and union_len <= max_len:
                p["end"] = max(p["end"], c["end"])
                p["types"] = list(dict.fromkeys(p["types"] + c["types"]))
                p["drivers"] = list(dict.fromkeys(p["drivers"] + c["drivers"]))
                p["summaries"] = list(dict.fromkeys(p["summaries"] + c["summaries"]))[:4]
                p["sources"] = list(dict.fromkeys(p["sources"] + c["sources"]))
                p["confidence"] = max(p["confidence"], c["confidence"])
                p["continues_after"] = bool(c.get("continues_after")) if c["end"] >= p["end"] else p["continues_after"]
                p["merged"] = p.get("merged", 1) + 1
                if c.get("story"):
                    ps = p.get("story")
                    p["story"] = [min(ps[0], c["story"][0]), max(ps[1], c["story"][1])] if ps else list(c["story"])
                if "prior" in p or "prior" in c:
                    p["prior"] = max(p.get("prior", 0.0), c.get("prior", 0.0))
                    p["action_type"] = next((t for t in p["types"] if t != "PICCO_VOCE"), "PICCO_VOCE")
                continue
            if gap < 0 and (min(p["end"], c["end"]) - c["start"]) > 0.6 * (c["end"] - c["start"]):
                continue                 # quasi tutta dentro la precedente, che pero' e' gia' al limite di durata
        out.append(dict(c))
    return out


def rank_candidates(ctx: Ctx, cands: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Punteggio preliminare: le migliori N vanno all'analisi profonda (il v10 teneva le prime 28 in ordine di tempo)."""
    f = ctx.feats
    sorted_voice = np.sort(f.voice) if f is not None and len(f) else np.zeros(1)
    kept: List[Dict[str, Any]] = []
    for c in cands:
        pieces = subtract_zones(c["start"], c["end"], ctx.blacklist)
        if not pieces:
            continue
        a, b = max(pieces, key=lambda p: p[1] - p[0])
        if b - a < 6.0:
            continue
        c = dict(c, start=round(a, 1), end=round(b, 1))
        if c.get("story"):
            s0, s1 = max(a, c["story"][0]), min(b, c["story"][1])
            c["story"] = [round(s0, 1), round(s1, 1)] if s1 - s0 >= 1.0 else None
        vpk = float(f.window(f.voice, a, b).max()) if f is not None and len(f) else 0.0
        vrank = float(np.searchsorted(sorted_voice, vpk) / max(1, len(sorted_voice)))
        kw = len(ctx.words.in_range(ATTACK_PATTERNS + CONTACT_PATTERNS + SPIN_PATTERNS, "action", a, b))
        kw_n = min(1.0, kw / 4.0)
        bonus = max([TYPE_BONUS.get(t, 0.0) for t in c["types"]] + [0.0])
        if "telecronaca" in c["sources"] and "picco voce" in c["sources"]:
            bonus += 0.05                 # due scout indipendenti sullo stesso momento
        c["voice_peak"] = round(vpk, 3)
        c["prior"] = round(0.50 * c["confidence"] / 100.0 + 0.30 * vrank + 0.20 * kw_n + bonus, 4)
        c["action_type"] = next((t for t in c["types"] if t != "PICCO_VOCE"), "PICCO_VOCE")
        c["phase_hint"] = phase_at((a + b) / 2, ctx.phases)
        kept.append(c)
    kept.sort(key=lambda x: x["prior"], reverse=True)
    kept = kept[:limit]
    kept.sort(key=lambda x: x["start"])
    for i, c in enumerate(kept, start=1):
        c["id"] = i
    return kept


# ---------------------------------------------------------------------------------------------------------
# AGENTE 3 - Continuita': dove comincia e dove finisce DAVVERO l'azione, letta per intero
# ---------------------------------------------------------------------------------------------------------
CONTINUITY_SCHEMA = {
    "type": "object",
    "properties": {
        "story": {"type": "string", "minLength": 10},
        "action_start": {"type": "number", "x-time": True},
        "action_end": {"type": "number", "x-time": True},
        "resolved": {"type": "boolean"},
    },
    "required": ["story", "action_start", "action_end", "resolved"],
}
ACTION_PRIORITY = ["CONTATTO", "PARTENZA", "SORPASSO", "VOLATA", "ERRORE_SALVATAGGIO", "DUELLO", "POLE_LAP"]


def refine_story_windows(ctx: Ctx, cands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, float]:
    """
    Per ogni candidata rilegge la telecronaca attorno al momento (45 s prima, 75 s dopo) e stabilisce dove
    l'azione inizia e dove si conclude nella sua interezza: preparazione, manovra, contro-sorpasso, conseguenze.
    Lo scout legge blocchi di 4 minuti e tende a segnare il solo momento clou; questo agente guarda una sola
    azione alla volta e ne segue lo sviluppo anche oltre il confine del blocco. La finestra puo' solo allargarsi:
    a togliere i tempi morti pensa poi il taglio sui dati video.
    """
    total = ctx.info.duration
    system = (
        "Sei il montatore di un canale di motorsport. Ti viene indicata un'azione gia' individuata in una diretta "
        "di gara e la telecronaca dei secondi attorno. Devi dire dove l'azione comincia davvero e dove si conclude "
        "davvero, considerandola PER INTERO, perche' il montaggio non deve tagliarla a meta' ne' spezzarla. "
        "Fanno parte della stessa azione: la preparazione (si avvicina, prende la scia, mette pressione, studia "
        "l'avversario), i tentativi andati a vuoto, il sorpasso o il contatto, l'eventuale contro-sorpasso o "
        "incrocio di traiettoria, il duello che prosegue nelle curve successive, le conseguenze immediate "
        "(chi resta davanti, auto in testacoda che riparte, posizioni perse, bandiere), fino a quando l'esito e' "
        "chiaro e il telecronista passa ad altro. Se lo stesso duello continua senza interruzione anche per un "
        "minuto o piu', e' un'unica azione. NON fanno parte dell'azione: chiacchiere generiche, lettura di "
        "classifiche e distacchi, altri duelli che non c'entrano, pubblicita', interviste. "
        "Rispondi con i secondi letti fra parentesi quadre."
    )
    out: List[Dict[str, Any]] = []
    widened, added = 0, 0.0
    for i, c in enumerate(cands, start=1):
        c = dict(c)
        est = c.get("story") or [c["start"], c["end"]]
        lo, hi = max(0.0, c["start"] - 45.0), min(total, c["end"] + 75.0)
        pieces = [pz for pz in subtract_zones(lo, hi, ctx.blacklist) if pz[0] <= c["end"] and pz[1] >= c["start"]]
        if pieces:
            lo, hi = max(pieces, key=lambda pz: min(pz[1], c["end"]) - max(pz[0], c["start"]))
        segs = [sg for sg in ctx.segments if sg["end"] >= lo and sg["start"] <= hi]
        if len(segs) < 2:
            out.append(c)
            continue
        lines = [f"[{sg['start']:.1f}-{sg['end']:.1f}] {sg['text']}" for sg in segs]
        user = (
            f"AZIONE INDIVIDUATA: {'; '.join(c.get('summaries', [])[:2]) or 'picco di voce del telecronista'}\n"
            f"FINESTRA STIMATA: da {est[0]:.1f} a {est[1]:.1f} secondi\n"
            f"PILOTI NOTI: {', '.join(ctx.race.get('drivers', [])[:30]) or 'non disponibili'}\n\n"
            "TELECRONACA:\n" + "\n".join(lines) + "\n\n"
            "Compila in quest'ordine:\n"
            "story: che cosa succede dall'inizio alla conclusione, in una o due frasi.\n"
            "action_start: secondo in cui l'azione comincia (compresa la preparazione).\n"
            "action_end: secondo in cui l'azione e' conclusa e il telecronista passa ad altro.\n"
            "resolved: false se alla fine del testo l'azione e' ancora in corso.\n"
            f"Rispondi SOLO con JSON conforme a: {schema_hint(CONTINUITY_SCHEMA)}")
        res = ctx.llm.chat_json(ctx.models["text"], system, user, CONTINUITY_SCHEMA, temperature=0.0,
                                num_predict=400, label=f"continuita' {i}/{len(cands)}")
        if res:
            a, b = float(res["action_start"]), float(res["action_end"])
            # tempi plausibili: dentro il testo letto e a cavallo della finestra stimata
            if lo - 1.0 <= a < b <= hi + 1.0 and a <= est[1] and b >= est[0]:
                story = [round(max(lo, min(est[0], a)), 1), round(min(hi, max(est[1], b)), 1)] if c.get("story") \
                    else [round(max(lo, a), 1), round(min(hi, b), 1)]
                gain = (est[0] - story[0]) + (story[1] - est[1]) if c.get("story") else 0.0
                if gain > 1.0:
                    widened += 1
                    added += gain
                c["story"] = story
                c["start"] = round(min(c["start"], story[0]), 1)
                c["end"] = round(max(c["end"], story[1]), 1)
                c["continues_after"] = (not res["resolved"]) and b >= hi - 5.0
                if res["story"] not in c.get("summaries", []):
                    c["summaries"] = ([res["story"]] + list(c.get("summaries", [])))[:4]
        out.append(c)
        if i % 10 == 0:
            log(f"    ...continuita' verificata su {i}/{len(cands)} azioni")
    return out, widened, added


def combine_cands(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Due candidate che dopo il taglio risultano la stessa azione (o due azioni attaccate) diventano una."""
    c = dict(a)
    for key in ("types", "drivers", "summaries", "sources"):
        c[key] = list(dict.fromkeys(list(a.get(key, [])) + list(b.get(key, []))))
    c["summaries"] = c["summaries"][:4]
    c["confidence"] = max(a.get("confidence", 0), b.get("confidence", 0))
    c["prior"] = max(a.get("prior", 0.0), b.get("prior", 0.0))
    c["voice_peak"] = max(a.get("voice_peak", 0.0), b.get("voice_peak", 0.0))
    real = [t for t in c["types"] if t in ACTION_PRIORITY]
    c["action_type"] = min(real, key=ACTION_PRIORITY.index) if real else "PICCO_VOCE"
    stories = [x["story"] for x in (a, b) if x.get("story")]
    c["story"] = [min(x[0] for x in stories), max(x[1] for x in stories)] if stories else None
    c["continues_after"] = bool(b.get("continues_after"))
    c["merged"] = a.get("merged", 1) + b.get("merged", 1)
    return c


# ---------------------------------------------------------------------------------------------------------
# FASE 4 - Analisi visiva su CUDA: rilevamento, tracciamento, fisica dell'azione
# ---------------------------------------------------------------------------------------------------------
class CudaVehicleDetector:
    """YOLO (26x, ripiego 11x / v8x) su CUDA in FP16, a lotti. Restituisce box normalizzati 0..1."""

    def __init__(self, models_dir: Path, prefs: Sequence[str], imgsz: int = 1280, conf: float = 0.30):
        torch = _import_torch()
        from ultralytics import YOLO
        models_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.name = ""
        last_err: Optional[Exception] = None
        for name in prefs:
            try:
                self.model = YOLO(str(models_dir / name) if not Path(name).is_absolute() else name)
                self.name = Path(name).stem
                break
            except Exception as e:          # pesi non scaricabili o architettura sconosciuta a questa versione
                last_err = e
        if self.model is None:
            raise RaceClipperError(f"Nessun modello YOLO caricabile ({', '.join(prefs)}): {last_err}")
        self.cuda = bool(torch.cuda.is_available())
        self.device: Any = 0 if self.cuda else "cpu"
        self.imgsz, self.conf = imgsz, conf
        self.batch = 8 if self.cuda else 2
        # FP16 su CUDA: Ultralytics 8.4 usa `quantize=16`, le versioni precedenti `half=True`
        self._precision: Dict[str, Any] = {"quantize": 16} if self.cuda else {}

    def detect(self, frames: List[np.ndarray]) -> List[List[Det]]:
        try:
            results = self.model.predict(frames, classes=list(VEHICLE_CLASSES), conf=self.conf, imgsz=self.imgsz,
                                         device=self.device, verbose=False, **self._precision)
        except Exception:
            if "quantize" not in self._precision:
                raise
            self._precision = {"half": True}
            results = self.model.predict(frames, classes=list(VEHICLE_CLASSES), conf=self.conf, imgsz=self.imgsz,
                                         device=self.device, verbose=False, **self._precision)
        out: List[List[Det]] = []
        for res in results:
            dets: List[Det] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xy = boxes.xyxyn.cpu().numpy()
                cf = boxes.conf.cpu().numpy()
                cl = boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), c, k in zip(xy, cf, cl):
                    dets.append((int(k), float(c), float(x1), float(y1), float(x2), float(y2)))
            out.append(dets)
        return out

    def close(self) -> None:
        self.model = None
        flush_vram()


class Track:
    __slots__ = ("id", "x1", "y1", "x2", "y2", "vx", "vy", "grow", "hits", "miss", "conf")

    def __init__(self, tid: int, det: Det):
        self.id = tid
        _, self.conf, self.x1, self.y1, self.x2, self.y2 = det
        self.vx = self.vy = self.grow = 0.0
        self.hits, self.miss = 1, 0.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.w * self.h

    def box(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class SimpleTracker:
    """
    Tracker a ID persistenti (IoU sul box predetto + distanza dei centri). Il v10 non tracciava: "leader" e
    "secondo" erano i due box piu' grandi di ogni frame e gli scambi di identita' diventavano falsi sorpassi.
    """

    def __init__(self, max_miss_s: float = 0.6):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_miss = max_miss_s

    def reset(self) -> None:
        self.tracks = []

    def update(self, dets: List[Det], dt: float) -> List[Track]:
        dt = max(dt, 1e-3)
        preds = [[t.x1 + t.vx * dt, t.y1 + t.vy * dt, t.x2 + t.vx * dt, t.y2 + t.vy * dt] for t in self.tracks]
        pairs: List[Tuple[float, int, int]] = []
        for i, t in enumerate(self.tracks):
            pcx, pcy = (preds[i][0] + preds[i][2]) / 2, (preds[i][1] + preds[i][3]) / 2
            size = max(t.w, t.h, 0.02)
            for j, d in enumerate(dets):
                iou = _iou(preds[i], d[2:6])
                if iou > 0.10:
                    pairs.append((iou, i, j))
                else:
                    dist = math.hypot((d[2] + d[4]) / 2 - pcx, (d[3] + d[5]) / 2 - pcy) / size
                    if dist < 1.2:
                        pairs.append((0.08 * (1.0 - dist / 1.2), i, j))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            t, d = self.tracks[i], dets[j]
            ncx, ncy = (d[2] + d[4]) / 2, (d[3] + d[5]) / 2
            nvx, nvy = (ncx - t.cx) / dt, (ncy - t.cy) / dt
            old_size = math.sqrt(max(1e-8, t.area))
            new_size = math.sqrt(max(1e-8, (d[4] - d[2]) * (d[5] - d[3])))
            t.grow = 0.7 * t.grow + 0.3 * ((new_size - old_size) / old_size / dt)
            a = 0.5 if t.hits >= 2 else 1.0
            t.vx, t.vy = (1 - a) * t.vx + a * nvx, (1 - a) * t.vy + a * nvy
            t.conf, t.x1, t.y1, t.x2, t.y2 = d[1], d[2], d[3], d[4], d[5]
            t.hits += 1
            t.miss = 0.0
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t.miss += dt
                t.x1, t.y1, t.x2, t.y2 = preds[i]
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        for j, d in enumerate(dets):
            if j not in used_d:
                self.tracks.append(Track(self.next_id, d))
                self.next_id += 1
        return [t for t in self.tracks if t.hits >= 3 and t.miss == 0.0]


def pair_gap_norm(a: Track, b: Track, ar: float) -> float:
    """Distanza fra i bordi dei due box in larghezze auto (0 = si toccano). ar = altezza/larghezza del frame."""
    hgap = max(0.0, max(a.x1, b.x1) - min(a.x2, b.x2))
    vgap = max(0.0, max(a.y1, b.y1) - min(a.y2, b.y2)) * ar
    return math.hypot(hgap, vgap) / max(0.01, (a.w + b.w) / 2)


class ActionPhysics:
    """Stato della "telemetria visiva" lungo la finestra: coppia in lotta, sorpassi, contatti fra box."""

    def __init__(self, ar: float):
        self.ar = ar
        self.tracker = SimpleTracker()
        self.pair: Optional[Tuple[int, int]] = None
        self.lead: Optional[int] = None
        self.dir: Optional[Tuple[float, float]] = None
        self.order: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.overtakes: List[float] = []
        self.touches: List[float] = []

    def cut(self) -> None:
        self.tracker.reset()
        self.pair = self.lead = None
        self.dir = None
        self.order = {}

    def step(self, t: float, dets: List[Det], dt: float, bg: Tuple[float, float, bool]) -> Dict[str, Any]:
        tracks = self.tracker.update(dets, dt)
        sample: Dict[str, Any] = {"n": len(tracks), "gap": float("nan"), "A": None, "B": None, "F": None}
        if not tracks:
            self.pair = self.lead = None
            return sample
        by_id = {tr.id: tr for tr in tracks}
        top = sorted(tracks, key=lambda tr: tr.area, reverse=True)[:6]

        # auto di riferimento con isteresi: la camera non salta da un'auto all'altra a ogni frame
        best = max(top, key=lambda tr: tr.area * (1.0 + min(tr.hits, 40) / 80.0))
        if self.lead in by_id and by_id[self.lead].area >= 0.6 * best.area:
            lead = by_id[self.lead]
        else:
            lead = best
        self.lead = lead.id

        # coppia in lotta: la piu' vicina fra le auto principali, a profondita' simile
        best_pair: Optional[Tuple[Track, Track]] = None
        best_gap = 1e9
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                ratio = max(a.area, b.area) / max(1e-6, min(a.area, b.area))
                if ratio > 6.0:
                    continue
                g = pair_gap_norm(a, b, self.ar)
                if g < best_gap:
                    best_gap, best_pair = g, (a, b)
        if self.pair and self.pair[0] in by_id and self.pair[1] in by_id:
            pa, pb = by_id[self.pair[0]], by_id[self.pair[1]]
            g_prev = pair_gap_norm(pa, pb, self.ar)
            if best_pair is None or g_prev <= best_gap * 1.4 + 0.3:
                best_pair, best_gap = (pa, pb), g_prev
        if best_pair is not None:
            a, b = sorted(best_pair, key=lambda tr: tr.cx)      # A = auto a sinistra, B = auto a destra
            self.pair = (a.id, b.id)
            sample["gap"] = float(best_gap)
            sample["A"], sample["B"] = a.box(), b.box()
            ratio = max(a.area, b.area) / max(1e-6, min(a.area, b.area))
            if (_iou(a.box(), b.box()) > 0.03 or best_gap < 0.06) and ratio < 3.0:
                self.touches.append(t)
            self._check_overtake(t, a, b, best_gap, bg)
        else:
            self.pair = None
        if best_pair is not None and best_gap < 3.0:
            a, b = best_pair
            sample["F"] = [min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2)]
        else:
            sample["F"] = lead.box()
        return sample

    def _check_overtake(self, t: float, a: Track, b: Track, gap: float, bg: Tuple[float, float, bool]) -> None:
        """
        Sorpasso = scambio d'ordine lungo la direzione di marcia. La direzione e' la velocita' delle auto
        rispetto allo sfondo (se la camera segue le auto, loro sono ferme nell'immagine e lo sfondo scorre).
        Se il moto nel piano e' troppo piccolo (auto in avvicinamento o in allontanamento) si usa la profondita':
        chi cresce sta arrivando, e davanti c'e' chi e' piu' in basso; viceversa se rimpicciolisce.
        """
        ar = self.ar
        if a.id > b.id:
            a, b = b, a                     # ordine per ID, non per lato: nel sorpasso laterale i lati si scambiano
        bvx, bvy, bok = bg
        rvx = (a.vx + b.vx) / 2 - (bvx if bok else 0.0)
        rvy = ((a.vy + b.vy) / 2 - (bvy if bok else 0.0)) * ar
        speed = math.hypot(rvx, rvy)
        d: Optional[Tuple[float, float]] = None
        if speed > 0.04:
            d = (rvx / speed, rvy / speed)
        else:
            grow = (a.grow + b.grow) / 2
            if abs(grow) > 0.08:
                d = (0.0, 1.0 if grow > 0 else -1.0)
        if d is None:
            return
        if self.dir is not None:
            mx, my = 0.7 * self.dir[0] + 0.3 * d[0], 0.7 * self.dir[1] + 0.3 * d[1]
            norm = math.hypot(mx, my)
            if norm < 0.3:                 # direzione appena invertita: stima instabile, si riparte
                self.dir = d
                self.order = {}
                return
            d = (mx / norm, my / norm)
        self.dir = d
        mean_w = max(0.01, (a.w + b.w) / 2)
        s = ((a.cx - b.cx) * d[0] + (a.cy - b.cy) * ar * d[1]) / mean_w       # > 0: A e' davanti
        key = (a.id, b.id)
        st = self.order.setdefault(key, {"sign": 0, "pend": 0, "n": 0, "last_s": s, "last_t": -1e9})
        jump = abs(s - st["last_s"])
        st["last_s"] = s
        if jump > 0.9:                      # salto discontinuo: e' cambiata la stima, non la posizione in pista
            st["pend"], st["n"] = 0, 0
            return
        if abs(s) < 0.25:
            return
        sign = 1 if s > 0 else -1
        if sign == st["pend"]:
            st["n"] += 1
        else:
            st["pend"], st["n"] = sign, 1
        if st["n"] >= 2 and sign != st["sign"]:
            if st["sign"] != 0 and gap < 2.0 and t - st["last_t"] > 3.0:
                self.overtakes.append(t)
                st["last_t"] = t
            st["sign"] = sign


def estimate_bg_flow(prev_gray: np.ndarray, gray: np.ndarray, boxes: List[Sequence[float]],
                     overlay_small: Optional[np.ndarray], dt: float) -> Tuple[float, float, bool]:
    """Moto dello sfondo (frazioni di frame al secondo) con flusso ottico sparso, escluse auto e grafiche fisse."""
    h, w = prev_gray.shape
    mask = np.full((h, w), 255, np.uint8)
    if overlay_small is not None:
        mask[overlay_small] = 0
    for x1, y1, x2, y2 in boxes:
        mask[max(0, int(y1 * h) - 2):int(y2 * h) + 3, max(0, int(x1 * w) - 2):int(x2 * w) + 3] = 0
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=140, qualityLevel=0.01, minDistance=7, mask=mask)
    if pts is None or len(pts) < 12:
        return 0.0, 0.0, False
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    if nxt is None:
        return 0.0, 0.0, False
    good = st.reshape(-1) == 1
    if good.sum() < 10:
        return 0.0, 0.0, False
    d = (nxt.reshape(-1, 2) - pts.reshape(-1, 2))[good]
    med = np.median(d, axis=0)
    return float(med[0] / w / max(dt, 1e-3)), float(med[1] / h / max(dt, 1e-3)), True


def keep_vehicle(det: Det, overlay: Optional[np.ndarray], webcam_rect: Optional[List[float]]) -> bool:
    _, _, x1, y1, x2, y2 = det
    w, h = x2 - x1, y2 - y1
    if w < 0.015 or h < 0.02 or w > 0.70 or h > 0.75:
        return False
    if y2 > 0.97 and w > 0.45:
        return False                         # cofano/abitacolo della propria auto nelle viste onboard
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if webcam_rect and webcam_rect[0] <= cx <= webcam_rect[2] and webcam_rect[1] <= cy <= webcam_rect[3]:
        return False
    if overlay is not None:
        gh, gw = overlay.shape
        patch = overlay[int(y1 * gh):max(int(y1 * gh) + 1, int(y2 * gh)), int(x1 * gw):max(int(x1 * gw) + 1, int(x2 * gw))]
        if patch.size and patch.mean() > 0.6:
            return False                     # icone e miniature dentro le grafiche fisse (mappa, classifica)
    return True


def analysis_window(ctx: Ctx, cand: Dict[str, Any], extra_pre: float = 0.0, extra_post: float = 0.0) -> Tuple[int, int]:
    """
    Finestra da analizzare (in frame sorgente): la candidata piu' margini, senza entrare nelle zone escluse.
    extra_pre/extra_post allargano la finestra quando il taglio scopre che l'azione prosegue oltre il bordo.
    """
    info = ctx.info
    w0 = max(0.0, cand["start"] - TUNING["pad_pre"] - extra_pre)
    w1 = cand["end"] + TUNING["pad_post"] + extra_post
    if not ctx.args.no_replay_join:
        rp = ctx.words.in_range(REPLAY_PATTERNS, "replay", cand["end"] - 1.0, cand["end"] + 45.0)
        if len(rp):
            w1 = max(w1, float(rp[0]) + 16.0)
    w1 = min(info.duration - 0.1, w1, cand["start"] + max(170.0, float(ctx.args.max_dur) + 90.0))
    for a, b, _ in ctx.blacklist:
        if b <= cand["start"]:
            w0 = max(w0, b)
        elif a >= cand["end"]:
            w1 = min(w1, a)
    n0 = int(math.floor(w0 * info.fps))
    n1 = int(math.ceil(w1 * info.fps))
    return n0, max(n0 + int(info.fps * 4), n1)


def analyze_candidate(ctx: Ctx, cand: Dict[str, Any], detector: Any, extra_pre: float = 0.0,
                      extra_post: float = 0.0) -> Dict[str, Any]:
    info = ctx.info
    fps = info.fps
    n0, n1 = analysis_window(ctx, cand, extra_pre, extra_post)
    n_frames = n1 - n0
    aw = int(TUNING["analysis_width"])
    ah = even(aw * info.height / info.width)
    ar = info.height / info.width
    det_step = max(1, int(round(fps / TUNING["det_fps"])))
    webcam_rect = ctx.webcam["rect"] if ctx.webcam else None
    overlay = ctx.overlay
    fw, fh = 320, even(320 * ar)
    overlay_flow = None
    if overlay is not None:
        overlay_flow = cv2.resize(overlay.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST).astype(bool)

    physics = ActionPhysics(ar)
    watcher = WebcamWatcher(ctx.screener, webcam_rect) if (ctx.screener is not None and webcam_rect) else None
    watch_step = max(1, int(round(fps * 0.5)))
    batch = int(getattr(detector, "batch", 8))

    cuts: List[int] = []
    motion = np.zeros(n_frames, np.float32)
    luma = np.zeros((n_frames, 4), np.float32)
    s_f: List[int] = []
    s_n: List[int] = []
    s_gap: List[float] = []
    s_a: List[Optional[List[float]]] = []
    s_b: List[Optional[List[float]]] = []
    s_focus: List[Optional[List[float]]] = []

    pending: List[Tuple[int, np.ndarray, bool]] = []
    prev_hist = None
    prev_thumb: Optional[np.ndarray] = None
    prev_flow_gray: Optional[np.ndarray] = None
    last_sample_f = -1
    last_cut = -10 ** 9
    cut_since_sample = False
    # Per ogni inquadratura: dove l'immagine cambia nel tempo. Parte bassa ferma + parte alta in movimento
    # = camera onboard (abitacolo fisso, pista che scorre): la regia la tratta in modo diverso.
    shots: List[Dict[str, Any]] = []
    shot_acc = np.zeros((90, 160), np.float32)
    shot_cnt = 0
    shot_start = 0

    def close_shot(end_f: int) -> None:
        nonlocal shot_acc, shot_cnt, shot_start
        onboard = False
        if shot_cnt >= max(3, int(fps * 0.5)):
            mean_d = shot_acc / shot_cnt
            low, top = mean_d[56:, :], mean_d[:45, :]
            onboard = float((low < 4.0).mean()) > 0.6 and float((top >= 4.0).mean()) > 0.4
        if end_f > shot_start:
            shots.append({"start": shot_start, "end": end_f, "onboard": bool(onboard)})
        shot_acc = np.zeros((90, 160), np.float32)
        shot_cnt = 0
        shot_start = end_f

    def flush() -> None:
        nonlocal prev_flow_gray, last_sample_f
        if not pending:
            return
        results = detector.detect([p[1] for p in pending])
        for (fi, frame, after_cut), dets in zip(pending, results):
            dets = [d for d in dets if keep_vehicle(d, overlay, webcam_rect)]
            gray = cv2.cvtColor(cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            dt = (fi - last_sample_f) / fps if last_sample_f >= 0 else det_step / fps
            if after_cut:
                physics.cut()
                prev_flow_gray = None
            bg = (0.0, 0.0, False)
            if prev_flow_gray is not None:
                bg = estimate_bg_flow(prev_flow_gray, gray, [d[2:6] for d in dets], overlay_flow, dt)
            sample = physics.step(n0 / fps + fi / fps, dets, dt, bg)
            s_f.append(fi)
            s_n.append(sample["n"])
            s_gap.append(sample["gap"])
            s_a.append(sample["A"])
            s_b.append(sample["B"])
            s_focus.append(sample["F"])
            prev_flow_gray = gray
            last_sample_f = fi
        pending.clear()

    got = 0
    with FFmpegReader(info, n0, n_frames, aw, ah, ctx.log_path("analisi"), flags="bilinear") as reader:
        for i in range(n_frames):
            frame = reader.read()
            if frame is None:
                break
            got = i + 1
            thumb = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            gray_t = hsv[:, :, 2]
            is_cut = False
            if prev_hist is not None and prev_thumb is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                d = cv2.absdiff(gray_t, prev_thumb)
                diff = float(d.mean())
                motion[i] = diff
                if (i - last_cut) > int(fps * 0.4) and ((corr < 0.55 and diff > 14.0) or corr < 0.30 or diff > 55.0):
                    is_cut = True
                    cuts.append(i)
                    last_cut = i
                    close_shot(i)
                else:
                    shot_acc += d
                    shot_cnt += 1
            prev_hist, prev_thumb = hist, gray_t
            p5, p50, p95 = np.percentile(gray_t, (5, 50, 95))
            luma[i] = (p5, p50, p95, float(hsv[:, :, 1].mean()))
            if is_cut:
                cut_since_sample = True
            if i % det_step == 0 or is_cut:
                pending.append((i, frame, cut_since_sample))
                cut_since_sample = False
                if len(pending) >= batch:
                    flush()
            if watcher is not None and i % watch_step == 0:
                watcher.push(i, cv2.resize(frame, (640, even(640 * ar)), interpolation=cv2.INTER_AREA))
        flush()
    close_shot(got)
    webcam_hits = watcher.finish() if watcher is not None else []

    def boxes(lst: List[Optional[List[float]]]) -> List[List[float]]:
        return [[round(v, 4) for v in b] if b else [] for b in lst]

    return {
        "n0": n0, "n_frames": got, "fps": fps,
        "cuts": cuts, "shots": shots,
        "s_f": s_f, "s_n": s_n, "s_gap": [None if (g != g) else round(g, 3) for g in s_gap],
        "s_a": boxes(s_a), "s_b": boxes(s_b), "s_focus": boxes(s_focus),
        "overtakes": [round(t, 2) for t in physics.overtakes],
        "touches": [round(t, 2) for t in physics.touches],
        "webcam_hits": [[int(f), int(p)] for f, p in webcam_hits],
        "motion": [round(float(v), 1) for v in motion[:got]],
        "luma": [[int(v) for v in row] for row in luma[:got:max(1, int(fps // 2))]],
        "detector": getattr(detector, "name", "?"),
    }


# ---------------------------------------------------------------------------------------------------------
# Curve a 0,2 s: prossimita' fisica + energia vocale + parole d'innesco
# ---------------------------------------------------------------------------------------------------------
def build_curves(ctx: Ctx, an: Dict[str, Any]) -> Dict[str, Any]:
    fps = an["fps"]
    binw = TUNING["bin"]
    w0 = an["n0"] / fps
    dur = an["n_frames"] / fps
    nb = max(1, int(math.ceil(dur / binw)))
    prox = np.zeros(nb, np.float32)
    ncars = np.zeros(nb, np.float32)
    gapmin = np.full(nb, np.nan, np.float32)
    for f, n, g in zip(an["s_f"], an["s_n"], an["s_gap"]):
        b = min(nb - 1, int(f / fps / binw))
        ncars[b] = max(ncars[b], n)
        if g is not None:
            prox[b] = max(prox[b], math.exp(-g / 0.7))
            gapmin[b] = g if np.isnan(gapmin[b]) else min(gapmin[b], g)
    feats = ctx.feats
    a = feats.idx(w0)
    voice = np.zeros(nb, np.float32)
    flux = np.zeros(nb, np.float32)
    seg_v = feats.voice[a:a + nb]
    seg_f = feats.flux[a:a + nb]
    voice[:len(seg_v)] = seg_v
    flux[:len(seg_f)] = seg_f

    def kernel(times: np.ndarray, half: float, height: float) -> np.ndarray:
        out = np.zeros(nb, np.float32)
        hb = max(1, int(round(half / binw)))
        for t in times:
            c = int((t - w0) / binw)
            for b in range(max(0, c - hb), min(nb, c + hb + 1)):
                out[b] = max(out[b], height * (1.0 - abs(b - c) / (hb + 1)))
        return out

    w1 = w0 + dur
    kw = np.maximum(kernel(ctx.words.in_range(ATTACK_PATTERNS, "attack", w0, w1), 1.5, 0.8),
                    np.maximum(kernel(ctx.words.in_range(CONTACT_PATTERNS, "contact", w0, w1), 1.5, 1.0),
                               kernel(ctx.words.in_range(SPIN_PATTERNS, "spin", w0, w1), 1.5, 1.0)))

    # Contatti: i box che si toccano non bastano (in fila indiana si sovrappongono sempre per prospettiva).
    # Serve un secondo indizio: botto nell'audio oppure parola di contatto nella telecronaca subito dopo.
    contacts: List[float] = []
    contact_words = ctx.words.in_range(CONTACT_PATTERNS, "contact", w0, w1)

    def transient_near(t: float) -> Optional[float]:
        """Istante del botto audio piu' forte attorno a t (se supera la soglia), per centrare shake e zoom."""
        a = feats.idx(t - 0.5)
        seg = feats.flux[a:feats.idx(t + 0.8) + 1]
        if len(seg) and float(seg.max()) >= 3.5:
            return a * feats.hop + int(np.argmax(seg)) * feats.hop
        return None

    for t in sorted(an["touches"]):
        said = bool(((contact_words >= t - 0.5) & (contact_words <= t + 3.5)).any())
        bang = transient_near(t)
        if (bang is not None or said) and (not contacts or t - contacts[-1] > 3.0):
            contacts.append(bang if bang is not None else t)   # un contatto dura qualche frame: un solo evento
    for i in np.where(flux >= 5.0)[0]:          # urto senza seconda auto (muro): botto forte + parola
        t = w0 + i * binw
        said = bool(((contact_words >= t - 0.5) & (contact_words <= t + 3.5)).any())
        if said and all(abs(t - c) > 3.0 for c in contacts):
            contacts.append(t)
    contacts = sorted(contacts)[:3]
    overtakes = sorted(an["overtakes"])
    events = np.maximum(kernel(np.array(overtakes), 1.5, 0.5), kernel(np.array(contacts), 1.5, 0.6))

    k = max(1, int(round(1.0 / binw)))
    box = np.ones(k) / k
    prox_s = np.convolve(prox, box, mode="same")
    voice_s = np.convolve(voice, np.ones(3) / 3, mode="same")
    presence = float((ncars >= 1).mean())
    if presence >= 0.15:
        act = 0.50 * prox_s + 0.30 * voice_s + 0.20 * kw + events
        basis = "prossimita' + voce"
    else:
        act = 0.65 * voice_s + 0.35 * kw + events      # nessuna auto tracciabile (es. onboard): guida la voce
        basis = "voce (auto non tracciabili)"
    return {"w0": w0, "nb": nb, "bin": binw, "prox": prox, "ncars": ncars, "gap": gapmin, "voice": voice,
            "flux": flux, "kw": kw, "act": act.astype(np.float32), "contacts": contacts, "overtakes": overtakes,
            "spin_words": [float(t) for t in ctx.words.in_range(SPIN_PATTERNS, "spin", w0, w1)],
            "presence": presence, "basis": basis}


# ---------------------------------------------------------------------------------------------------------
# Smart Action Trimming: innesco e risoluzione misurati, ingresso e uscita agganciati a stacchi e pause
# ---------------------------------------------------------------------------------------------------------
def _snap_in(t: float, cuts_t: List[float], words: WordIndex, fps: float) -> Tuple[float, str]:
    near = [c for c in cuts_t if t - 1.2 <= c <= t + 1.5]
    if near:
        c = min(near, key=lambda x: abs(x - t))
        return c + 1.0 / fps, "stacco di camera"          # primo frame pieno della nuova inquadratura
    gaps = [g for g in words.gaps(t - 1.0, t + 0.6, 0.22) if t - 1.0 <= g[1] <= t + 0.6]
    if gaps:
        g = min(gaps, key=lambda x: abs(x[1] - t))
        return max(g[0], g[1] - 0.12), "pausa del parlato"
    w = words.word_at(t)
    if w:
        return w[0] - 0.06, "inizio parola"
    return t, "curva"


def _snap_out(t: float, cuts_t: List[float], words: WordIndex, fps: float) -> Tuple[float, str]:
    how = "curva"
    gaps = [g for g in words.gaps(t - 0.8, t + 1.6, 0.28) if t - 0.8 <= g[0] <= t + 1.6]
    if gaps:
        g = min(gaps, key=lambda x: abs(x[0] - t))
        t, how = min(g[1], g[0] + 0.18), "fine frase"
    else:
        w = words.word_at(t)
        if w:
            t, how = w[1] + 0.10, "fine parola"
    near = [c for c in cuts_t if t - 0.9 < c <= t + 0.5]
    if near:
        return min(near) - 2.0 / fps, "prima dello stacco"  # mai due frame dell'inquadratura successiva in coda
    return t, how


def smart_trim(ctx: Ctx, cand: Dict[str, Any], an: Dict[str, Any], cv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Taglio sull'azione INTERA, non sul singolo momento clou. Tre fonti, unite:
      1) nucleo fisico: isteresi sulla curva di azione attorno al clou (un rettilineo di qualche secondo
         in cui le auto si allungano non chiude il duello);
      2) finestra narrativa: inizio e fine della storia secondo la telecronaca e l'Agente Continuita';
         se ne tolgono solo i bordi in cui non c'e' nulla da vedere;
      3) concatenazione: sorpassi, contro-sorpassi, contatti ed errori a ridosso dei bordi fanno parte
         della stessa azione e la allargano.
    Se l'azione tocca il bordo della finestra analizzata lo segnala (open_start / open_end): chi chiama
    analizza altro video e ripete il taglio. La durata e' libera fino a --max-dur.
    """
    fps = an["fps"]
    binw, nb, w0 = cv["bin"], cv["nb"], cv["w0"]
    w_end = w0 + an["n_frames"] / fps
    act = cv["act"].astype(np.float64)
    min_dur, max_dur = float(ctx.args.min_dur), float(ctx.args.max_dur)

    def b_of(t: float) -> int:
        return int(clamp(round((t - w0) / binw), 0, nb - 1))

    weight = np.full(nb, 0.55)
    weight[b_of(cand["start"] - 3.0):b_of(cand["end"] + 3.0) + 1] = 1.0
    in_hint = lambda t: cand["start"] - 4.0 <= t <= cand["end"] + 6.0   # noqa: E731
    contacts = [t for t in cv["contacts"] if in_hint(t)]
    overtakes = [t for t in cv["overtakes"] if in_hint(t)]
    if contacts:
        climax_t, climax_kind = contacts[0], "contatto"
    elif overtakes:
        climax_t, climax_kind = max(overtakes, key=lambda t: act[b_of(t)]), "sorpasso"
    else:
        climax_t, climax_kind = w0 + int(np.argmax(act * weight)) * binw, "picco di azione"
    pb = b_of(climax_t)
    one = max(1, int(round(1.0 / binw)))
    peak = float(act[max(0, pb - one):pb + one + 1].max())
    base = float(np.percentile(act, 20))
    contrast = peak - base
    flat = contrast < 0.08
    active = act >= base + 0.22 * contrast
    dip = int(round(TUNING["quiet_dip"] / binw))
    quiet = int(round(TUNING["quiet_end"] / binw))

    def walk_back(b0: int) -> int:
        sb_, gap_, b_ = b0, 0, b0
        while b_ > 0:
            b_ -= 1
            if active[b_]:
                sb_, gap_ = b_, 0
            else:
                gap_ += 1
                if gap_ > dip:
                    break
        return sb_

    def walk_fwd(b0: int) -> int:
        eb_, gap_, b_ = b0, 0, b0
        while b_ < nb - 1:
            b_ += 1
            if active[b_]:
                eb_, gap_ = b_, 0
            else:
                gap_ += 1
                if gap_ >= quiet:
                    break
        return eb_

    parts: List[str] = []
    if flat:
        sb, eb = b_of(cand["start"]), b_of(cand["end"])
        parts.append("finestra della telecronaca (curva piatta)")
        if not (cand["start"] <= climax_t <= cand["end"]):
            climax_t, climax_kind = (cand["start"] + cand["end"]) / 2, "centro della finestra"   # il picco era rumore
    else:
        # 1) nucleo fisico attorno al clou
        sb, eb = walk_back(pb), walk_fwd(pb)
        parts.append(cv["basis"])
        # 2) finestra narrativa: si tolgono solo i bordi senza auto e senza attivita'
        story = cand.get("story")
        ns, ne = (b_of(story[0]), b_of(story[1])) if story else (0, 0)
        if ne > ns:
            if cv["presence"] >= 0.15:
                shown = np.convolve((cv["ncars"] >= 1).astype(np.float64), np.ones(2 * one + 1), mode="same") > 0
                keep = shown | (act >= base + 0.10 * contrast)
            else:
                keep = np.ones(nb, bool)            # auto non tracciabili (onboard): ci si fida della telecronaca
            idx = np.flatnonzero(keep[ns:ne + 1])
            if len(idx):
                n_first, n_last = ns + int(idx[0]), ns + int(idx[-1])
                if n_first < sb - one or n_last > eb + one:
                    parts.append("storia della telecronaca")
                sb, eb = min(sb, n_first), max(eb, n_last)
        # 3) concatenazione degli eventi vicini ai bordi (contro-sorpasso, secondo contatto, testacoda)
        chain = TUNING["chain_s"]
        marks = sorted(set(cv["overtakes"]) | set(cv["contacts"]) | set(cv["spin_words"]))
        chained = 0
        for _ in range(8):
            t_a, t_b = w0 + sb * binw, w0 + (eb + 1) * binw
            grown = False
            for t in marks:
                if t_b < t <= t_b + chain:
                    eb, grown = max(eb, walk_fwd(b_of(t))), True
                elif t_a - chain <= t < t_a:
                    sb, grown = min(sb, walk_back(b_of(t))), True
            if not grown:
                break
            chained += 1
        if chained:
            parts.append("eventi concatenati")
    t_trig, t_res = w0 + sb * binw, w0 + (eb + 1) * binw

    cover = cand.get("cover")               # clip gia' tagliate e poi riunite: l'unione va coperta per intero
    if cover:
        t_trig = min(t_trig, float(cover[0]) + TUNING["lead_in"])
        t_res = max(t_res, float(cover[1]) - TUNING["tail_out"])
        parts.append("frammenti riuniti")

    # L'azione arriva al bordo della finestra analizzata? Allora probabilmente prosegue oltre.
    episode = t_res - t_trig
    open_start = (not flat) and t_trig <= w0 + 2.0 and episode < max_dur
    open_end = (not flat) and t_res >= w_end - 2.0 and episode < max_dur

    t_in = t_trig - TUNING["lead_in"]
    t_out = t_res + TUNING["tail_out"]
    truncated = False
    if t_out - t_in > max_dur:
        # azione piu' lunga del massimo: tiene la finestra con piu' azione che contiene il momento clou
        s_lo = max(t_in, climax_t - (max_dur - 5.0))
        s_hi = max(s_lo, min(climax_t - 3.0, t_out - max_dur))
        best_s, best_v = s_lo, -1.0
        s = s_lo
        while s <= s_hi + 1e-6:
            v = float(act[b_of(s):b_of(s + max_dur) + 1].sum())
            if v > best_v:
                best_s, best_v = s, v
            s += 0.4
        t_in, t_out = best_s, best_s + max_dur
        truncated = True
        parts.append("ridotta alla durata massima")
    if t_out - t_in < min_dur:
        need = min_dur - (t_out - t_in)
        t_in -= need * 0.65
        t_out += need * 0.35
    t_in = clamp(t_in, w0 + 0.05, w_end - 2.0)
    t_out = clamp(t_out, t_in + 2.0, w_end - 0.05)

    cuts_t = [w0 + c / fps for c in an["cuts"]]
    t_in2, how_in = _snap_in(t_in, cuts_t, ctx.words, fps)
    t_out2, how_out = _snap_out(t_out, cuts_t, ctx.words, fps)
    t_in2 = clamp(t_in2, w0 + 0.02, w_end - 2.0)
    t_out2 = clamp(t_out2, t_in2 + 2.0, w_end - 0.02)
    if t_out2 - t_in2 > max_dur + 1.5:
        t_out2 = t_in2 + max_dur
        how_out = "durata massima"
    if t_in2 > climax_t - 1.0:              # l'aggancio non deve mai tagliare via il momento clou
        t_in2, how_in = max(w0 + 0.02, min(t_in, climax_t - 1.5)), "curva"
    if t_out2 < climax_t + 1.0:
        t_out2, how_out = min(w_end - 0.02, max(t_out, climax_t + 1.5)), "curva"

    n_lo, n_hi = an["n0"], an["n0"] + an["n_frames"]
    n_in = int(clamp(round(t_in2 * fps), n_lo, n_hi - 2))
    n_out = int(clamp(round(t_out2 * fps), n_in + int(fps), n_hi))
    segments = [[n_in, n_out]]
    replay = _find_replay(ctx, an, cv, n_out / fps, max_dur + 12.0 - (n_out - n_in) / fps)
    if replay:
        segments.append(replay)
    dur = sum(b - a for a, b in segments) / fps
    return {
        "segments": segments, "t_in": round(n_in / fps, 3), "t_out": round(n_out / fps, 3), "dur": round(dur, 2),
        "climax_t": round(float(climax_t), 2), "climax_kind": climax_kind,
        "trigger_t": round(float(t_trig), 2), "resolution_t": round(float(t_res), 2),
        "method": " + ".join(parts), "snap_in": how_in, "snap_out": how_out, "replay": bool(replay),
        "open_start": bool(open_start), "open_end": bool(open_end), "truncated": truncated,
        "story_window": cand.get("story"),
    }


def _find_replay(ctx: Ctx, an: Dict[str, Any], cv: Dict[str, Any], t_out: float, budget: float) -> Optional[List[int]]:
    """Replay immediato dell'azione: parola chiave nella telecronaca + stacco di camera + auto visibili."""
    if ctx.args.no_replay_join or budget < 4.0:
        return None
    fps = an["fps"]
    w0 = cv["w0"]
    w_end = w0 + an["n_frames"] / fps
    kws = ctx.words.in_range(REPLAY_PATTERNS, "replay", t_out - 1.0, min(w_end, t_out + 45.0))
    if not len(kws):
        return None
    tk = float(kws[0])
    cuts_t = [w0 + c / fps for c in an["cuts"]]
    starts = [c for c in cuts_t if c >= t_out + 0.5 and tk - 6.0 <= c <= tk + 8.0]
    if not starts:
        return None
    c0 = min(starts, key=lambda c: abs(c - tk))
    later = [c for c in cuts_t if c >= c0 + 3.5]
    c1 = min(later[0] if later else c0 + 12.0, c0 + 12.0, c0 + budget, w_end - 0.2)
    if c1 - c0 < 3.5:
        return None
    b0 = int(clamp((c0 - w0) / cv["bin"], 0, cv["nb"] - 1))
    b1 = int(clamp((c1 - w0) / cv["bin"], b0 + 1, cv["nb"]))
    if float((cv["ncars"][b0:b1] >= 1).mean()) < 0.5:
        return None
    return [int(round(c0 * fps)) + 1, int(round(c1 * fps)) - 1]


def clip_metrics(ctx: Ctx, an: Dict[str, Any], cv: Dict[str, Any], trim: Dict[str, Any]) -> Dict[str, Any]:
    """Telemetria visiva e audio della clip tagliata (segmento principale)."""
    fps = an["fps"]
    binw, w0, nb = cv["bin"], cv["w0"], cv["nb"]
    t_in, t_out = trim["t_in"], trim["t_out"]
    b0 = int(clamp((t_in - w0) / binw, 0, nb - 1))
    b1 = int(clamp(math.ceil((t_out - w0) / binw), b0 + 1, nb))
    nc = cv["ncars"][b0:b1]
    gap = cv["gap"][b0:b1]
    gap_ok = np.nan_to_num(gap, nan=99.0)
    presence = float((nc >= 1).mean())
    multi = float((nc >= 2).mean())
    pack = float((nc >= 3).mean())
    battle = float((gap_ok < TUNING["gap_close"]).mean())
    w2w = float((gap_ok < TUNING["gap_w2w"]).mean())
    # Durate assolute: una clip lunga con dentro 8 s di ruota a ruota non deve valere meno di un frammento
    # di 10 s che contiene solo quegli 8 s (le percentuali da sole premiavano i tagli corti).
    battle_s = float((gap_ok < TUNING["gap_close"]).sum()) * binw
    w2w_s = float((gap_ok < TUNING["gap_w2w"]).sum()) * binw
    pack_s = float((nc >= 3).sum()) * binw
    overtakes = [t for t in cv["overtakes"] if t_in <= t <= t_out]
    contacts = [t for t in cv["contacts"] if t_in <= t <= t_out]
    spins = [t for t in cv["spin_words"] if t_in - 0.5 <= t <= t_out]
    f0 = int(clamp(round(t_in * fps) - an["n0"], 0, an["n_frames"] - 1))
    f1 = int(clamp(round(t_out * fps) - an["n0"], f0 + 1, an["n_frames"]))
    mot = np.array(an["motion"][f0:f1], dtype=np.float32)
    static_ratio = float((mot < 0.8).mean()) if len(mot) else 0.0
    feats = ctx.feats
    v = feats.window(feats.voice, t_in, t_out)
    all_sorted = np.sort(feats.voice)
    v_peak = float(v.max()) if len(v) else 0.0
    v_rank = float(np.searchsorted(all_sorted, v_peak) / max(1, len(all_sorted)))
    hook = float(feats.window(feats.voice, t_in, t_in + 3.0).max())
    rate = ctx.words.rate(t_in, t_out)
    hits = [p for f, p in an["webcam_hits"] if f0 <= f < f1]
    if not ctx.webcam:
        webcam_ratio = 0.0
    elif hits:
        webcam_ratio = float(np.mean(hits))
    else:
        webcam_ratio = 1.0                    # riquadro noto ma nessun controllo NPU disponibile: lo si considera presente
    step = max(1, int(fps // 2))
    lum = np.array(an["luma"][f0 // step:max(f0 // step + 1, f1 // step)], dtype=np.float32)
    luma = [float(x) for x in lum.mean(axis=0)] if len(lum) else [16.0, 110.0, 220.0, 90.0]

    pts = 0.0
    if presence >= 0.2:
        pts = (9.0 * min(1.0, max(w2w / 0.30, w2w_s / 5.0)) + 5.0 * min(1.0, max(battle / 0.50, battle_s / 10.0)) +
               min(8.0, 4.0 * len(overtakes)) + min(5.0, 5.0 * len(contacts) + 2.0 * min(1, len(spins))) +
               3.0 * min(1.0, max(pack / 0.25, pack_s / 4.0)))
    return {
        "car_presence": round(presence, 3), "multi_car_ratio": round(multi, 3), "pack_3plus_ratio": round(pack, 3),
        "close_battle_ratio": round(battle, 3), "wheel_to_wheel_ratio": round(w2w, 3),
        "close_battle_seconds": round(battle_s, 1), "wheel_to_wheel_seconds": round(w2w_s, 1),
        "clip_seconds": round(t_out - t_in, 1),
        "overtakes": len(overtakes), "contacts": len(contacts), "error_words": len(spins),
        "overtake_times": [round(t - t_in, 2) for t in overtakes],
        "contact_times": [round(t - t_in, 2) for t in contacts],
        "static_ratio": round(static_ratio, 3), "voice_peak": round(v_peak, 3), "voice_rank": round(v_rank, 3),
        "hook_voice": round(hook, 3), "speech_rate": round(rate, 2), "webcam_ratio": round(webcam_ratio, 3),
        "luma": [round(x, 1) for x in luma], "physics_score_30": round(min(30.0, pts), 1),
    }


# ---------------------------------------------------------------------------------------------------------
# VERIFICA 2 - Giudice visivo (alla cieca) e giudice editoriale
# ---------------------------------------------------------------------------------------------------------
SCENE_TYPES = ["AZIONE_IN_PISTA", "REPLAY", "GRAFICA_O_CLASSIFICA", "PUBBLICITA", "INTERVISTA_O_WEBCAM",
               "MENU_O_BOX", "ALTRO"]
VIS_ACTIONS = ["SORPASSO", "DUELLO_RAVVICINATO", "CONTATTO_O_INCIDENTE", "TESTACODA_O_FUORIPISTA",
               "PARTENZA_O_GRUPPO", "AUTO_SOLITARIA", "NESSUNA_AZIONE"]
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 10},
        "scene_type": {"type": "string", "enum": SCENE_TYPES},
        "cars_visible_max": {"type": "integer", "minimum": 0, "maximum": 60},
        "cars_close_max": {"type": "integer", "minimum": 0, "maximum": 60},
        "action": {"type": "string", "enum": VIS_ACTIONS},
        "action_visible": {"type": "boolean"},
        "visual_score": {"type": "integer", "minimum": 0, "maximum": 20},
        "on_screen_text": {"type": "string", "default": ""},
    },
    "required": ["description", "scene_type", "cars_visible_max", "cars_close_max", "action", "action_visible",
                 "visual_score"],
}
VISION_SYSTEM = (
    "Sei un commissario di gara che esamina fotogrammi di una diretta di motorsport o sim racing. Descrivi solo "
    "cio' che si vede. Non dedurre nulla che non sia visibile e non farti influenzare da cio' che ti aspetteresti. "
    "I fotogrammi sono in ordine cronologico e appartengono alla stessa clip; l'etichetta gialla indica "
    "numero e secondo. Ignora mappe, miniature e riquadri webcam quando conti le auto."
)
VISION_USER = (
    "Esamina i fotogrammi e compila i campi in quest'ordine.\n"
    "description: che cosa si vede, fotogramma per fotogramma (posizione delle auto, distanze, erba, ghiaia, fumo, "
    "auto di traverso, grafiche a tutto schermo).\n"
    "scene_type: AZIONE_IN_PISTA, REPLAY, GRAFICA_O_CLASSIFICA, PUBBLICITA, INTERVISTA_O_WEBCAM, MENU_O_BOX o ALTRO.\n"
    "cars_visible_max: numero massimo di auto distinte in un singolo fotogramma.\n"
    "cars_close_max: nel fotogramma piu' affollato, quante auto sono a meno di una lunghezza l'una dall'altra "
    "(0 se nessuna e' vicina a un'altra).\n"
    "action: SORPASSO, DUELLO_RAVVICINATO, CONTATTO_O_INCIDENTE, TESTACODA_O_FUORIPISTA, PARTENZA_O_GRUPPO; "
    "AUTO_SOLITARIA se c'e' una sola auto senza avversari vicini e senza errori; NESSUNA_AZIONE per schermate, "
    "classifiche, interviste.\n"
    "action_visible: true solo se l'azione si vede davvero nei fotogrammi.\n"
    "visual_score da 0 a 20: 16-20 duello ravvicinato, sorpasso, contatto o gruppo compatto ben visibili; "
    "8-15 auto che si inseguono a breve distanza; 0-7 auto solitaria o nessuna azione.\n"
    "on_screen_text: testo leggibile delle grafiche (giro, posizioni, nomi), se c'e'.\n"
)


def vlm_frame_times(trim: Dict[str, Any], info: VideoInfo) -> List[float]:
    t_in, t_out, c = trim["t_in"], trim["t_out"], trim["climax_t"]
    raw = [t_in + 0.4, c - 3.0, c - 1.5, c - 0.5, c, c + 0.7, c + 1.8, t_out - 0.5]
    span = t_out - t_in
    if span > 30.0:                       # azione lunga: due sguardi anche lontano dal momento clou
        raw += [t_in + span / 3.0, t_in + 2.0 * span / 3.0]
    main = len(raw)
    if len(trim["segments"]) > 1:
        a, b = trim["segments"][1]
        raw.append((a + b) / 2 / info.fps)
    out: List[float] = []
    for t in sorted(clamp(x, t_in + 0.2, max(t_in + 0.3, t_out - 0.2)) if i < main else x for i, x in enumerate(raw)):
        if not out or t - out[-1] >= 0.3:
            out.append(round(t, 2))
    return out


def prepare_vlm_frames(ctx: Ctx, item: Dict[str, Any], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    t_in = item["trim"]["t_in"]
    for k, t in enumerate(vlm_frame_times(item["trim"], ctx.info), start=1):
        p = out_dir / f"cand{item['id']:02d}_f{k}.jpg"
        if not extract_still(ctx.info, t, p, 960, ctx.log_path("fotogrammi")):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        label = f"F{k}  {t - t_in:+.1f}s"
        cv2.rectangle(img, (8, 8), (190, 46), (0, 0, 0), -1)
        cv2.putText(img, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        paths.append(p)
    return paths


def make_grid(paths: List[Path], out_path: Path) -> Optional[Path]:
    """Mosaico per i modelli che accettano una sola immagine per richiesta."""
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [cv2.resize(i, (640, 360)) for i in imgs if i is not None]
    if not imgs:
        return None
    while len(imgs) % 3:
        imgs.append(np.zeros_like(imgs[0]))
    rows = [np.hstack(imgs[i:i + 3]) for i in range(0, len(imgs), 3)]
    cv2.imwrite(str(out_path), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return out_path


def vision_judge(ctx: Ctx, item: Dict[str, Any], frames: List[Path], last: bool) -> Optional[Dict[str, Any]]:
    """Giudizio alla cieca: il modello visivo non riceve ne' i numeri dei sensori ne' la telecronaca."""
    if not frames:
        return None
    model = ctx.models["vision"]
    user = VISION_USER + f"Rispondi SOLO con JSON conforme a: {schema_hint(VISION_SCHEMA)}"
    single = "llama3.2-vision" in model or getattr(ctx, "_vision_single", False)
    res = None
    if not single:
        res = ctx.llm.chat_json(model, VISION_SYSTEM, f"Fotogrammi: {len(frames)}.\n" + user, VISION_SCHEMA,
                                images=[image_to_b64(p) for p in frames], temperature=0.0, num_predict=700,
                                num_ctx=16384, keep_alive=0 if last else "10m", retries=2,
                                label=f"occhio AI clip {item['id']}")
    if res is None:
        grid = make_grid(frames, frames[0].with_name(f"cand{item['id']:02d}_grid.jpg"))
        if grid is not None:
            res = ctx.llm.chat_json(model, VISION_SYSTEM,
                                    f"L'immagine e' un mosaico di {len(frames)} fotogrammi in ordine di lettura.\n" + user,
                                    VISION_SCHEMA, images=[image_to_b64(grid)], temperature=0.0, num_predict=700,
                                    num_ctx=8192, keep_alive=0 if last else "10m", retries=2,
                                    label=f"occhio AI clip {item['id']} (mosaico)")
            if res is not None and not single:
                setattr(ctx, "_vision_single", True)    # questo modello accetta una sola immagine: non riprovare
    return res


CLIP_PHASES = ["GARA", "QUALIFICA", "PARTENZA", "GIRO_FORMAZIONE_O_OUTLAP", "PUBBLICITA", "INTERVISTA",
               "CLASSIFICA_O_PAUSA", "POST_GARA"]
CLIP_NON_RACE = {"GIRO_FORMAZIONE_O_OUTLAP", "PUBBLICITA", "INTERVISTA", "CLASSIFICA_O_PAUSA", "POST_GARA"}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "string", "minLength": 10},
        "phase": {"type": "string", "enum": CLIP_PHASES},
        "is_live_action": {"type": "boolean"},
        "action_type": {"type": "string", "enum": ACTION_TYPES + ["NESSUNA"]},
        "drivers": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "default": []},
        "reason": {"type": "string", "minLength": 5},
        "story_score": {"type": "integer", "minimum": 0, "maximum": 35},
    },
    "required": ["facts", "phase", "is_live_action", "action_type", "reason", "story_score"],
}


def describe_metrics(m: Dict[str, Any]) -> str:
    parts = [
        f"auto visibili per il {int(m['car_presence'] * 100)}% della clip",
        f"due o piu' auto insieme per il {int(m['multi_car_ratio'] * 100)}%",
        f"in lotta ravvicinata per {m.get('close_battle_seconds', 0):.0f} s su {m.get('clip_seconds', 0):.0f} "
        f"(ruota a ruota {m.get('wheel_to_wheel_seconds', 0):.0f} s)",
        f"sorpassi misurati: {m['overtakes']}",
        f"contatti rilevati: {m['contacts']}",
        f"gruppo di 3 o piu' auto per il {int(m['pack_3plus_ratio'] * 100)}%",
    ]
    return "; ".join(parts)


def editorial_judge(ctx: Ctx, item: Dict[str, Any], last: bool) -> Optional[Dict[str, Any]]:
    race, m, trim = ctx.race, item["metrics"], item["trim"]
    vis = item.get("vision") or {}
    transcript = text_in_range(ctx.segments, trim["t_in"] - 1.0, trim["t_out"] + 1.0, max_chars=3600)
    before = text_in_range(ctx.segments, trim["t_in"] - 25.0, trim["t_in"] - 1.0, max_chars=500)
    system = (
        "Sei il direttore editoriale di un canale di motorsport e sim racing. Valuti una clip incrociando cio' che "
        "si sente e cio' che si vede. Sei severo: i voti alti vanno solo all'azione vera. Usa solo nomi presenti "
        "nei dati; se non sai chi e' coinvolto, non inventare."
    )
    user = (
        f"CONTESTO: pista '{race.get('circuit') or 'non nota'}', campionato '{race.get('championship') or 'non noto'}', "
        f"auto: {', '.join(race.get('cars', [])) or 'non note'}.\n"
        f"PILOTI NOTI: {', '.join(race.get('drivers', [])[:30]) or 'non disponibili'}\n"
        f"TELECRONACA SUBITO PRIMA: \"{before}\"\n"
        f"TELECRONACA DELLA CLIP ({trim['dur']:.0f} s): \"{transcript}\"\n"
        f"SENSORI VIDEO: {describe_metrics(m)}.\n"
        f"OCCHIO AI (fotogrammi): {vis.get('description', 'non disponibile')}\n"
        f"VOCE DEL TELECRONISTA: picco {int(m['voice_peak'] * 100)}/100, primi 3 secondi {int(m['hook_voice'] * 100)}/100.\n\n"
        "Compila in quest'ordine:\n"
        "facts: che cosa succede nella clip, in una o due frasi concrete (chi, dove, come), senza aggettivi vuoti.\n"
        f"phase: {', '.join(CLIP_PHASES)}.\n"
        "is_live_action: false se e' pubblicita', intervista, lettura di classifica, giro di formazione o outlap.\n"
        f"action_type: {', '.join(ACTION_TYPES)} oppure NESSUNA.\n"
        "drivers: cognomi dei piloti coinvolti, solo se citati nella telecronaca della clip.\n"
        "reason: perche' merita il voto.\n"
        "story_score da 0 a 35: 31-35 azione epica confermata da video e voce (tre auto affiancate, contatto, "
        "sorpasso sull'erba, partenza caotica, volata); 23-30 bel sorpasso o difesa dura ruota a ruota ben visibile; "
        "12-22 inseguimento senza sorpasso concluso o giro veloce; 0-11 poca azione, giro solitario, momento morto. "
        "La clip contiene l'azione per intero (preparazione, manovra, esito): la durata non e' un difetto, "
        "valuta il momento migliore e la storia che racconta.\n"
        f"Rispondi SOLO con JSON conforme a: {schema_hint(JUDGE_SCHEMA)}"
    )
    return ctx.llm.chat_json(ctx.models["text"], system, user, JUDGE_SCHEMA, temperature=0.1, num_predict=600,
                             keep_alive=0 if last else "20m", label=f"giudice clip {item['id']}")


def precheck_candidate(item: Dict[str, Any]) -> str:
    """VERIFICA 1, parte deterministica (prima di spendere il modello visivo). Restituisce il motivo dello scarto."""
    m = item["metrics"]
    if m["static_ratio"] > 0.60:
        return "schermata fissa (classifica, grafica o pausa)"
    if m["car_presence"] < 0.08 and m["static_ratio"] > 0.25:
        return "nessuna auto in pista nei fotogrammi"
    return ""


def verify_and_score(item: Dict[str, Any]) -> None:
    """VERIFICA 1 + VERIFICA 2 e punteggio 0-100. Scrive verdict, reject_reason, flags, score e breakdown."""
    m = item["metrics"]
    vis: Optional[Dict[str, Any]] = item.get("vision")
    judge: Optional[Dict[str, Any]] = item.get("judge")
    types = set(item["cand"].get("types", []))
    flags: List[str] = []
    reject = ""

    # --- Verifica 1: spot, classifiche, giri di formazione, interviste ---
    if vis and vis["scene_type"] in ("GRAFICA_O_CLASSIFICA", "PUBBLICITA", "INTERVISTA_O_WEBCAM", "MENU_O_BOX") \
            and m["car_presence"] < 0.5:
        reject = f"il giudice visivo vede {vis['scene_type'].lower().replace('_', ' ')}"
    elif judge and judge["phase"] in CLIP_NON_RACE and not judge["is_live_action"]:
        reject = f"fase non di gara: {judge['phase'].lower().replace('_', ' ')}"
    elif m["car_presence"] < 0.15 and not (vis and vis["scene_type"] == "AZIONE_IN_PISTA" and vis["action_visible"]):
        reject = "auto non visibili per quasi tutta la clip"

    # --- Verifica 2: azione visibile davvero (fisica + occhio AI) ---
    solo_phys = (m["multi_car_ratio"] < 0.20 and m.get("close_battle_seconds", 0.0) < 3.0
                 and m["overtakes"] == 0 and m["contacts"] == 0)
    error_ok = m["contacts"] > 0 or m["error_words"] > 0 or bool(vis and vis["action"] in (
        "TESTACODA_O_FUORIPISTA", "CONTATTO_O_INCIDENTE") and vis["action_visible"])
    pole_ok = ("POLE_LAP" in types or bool(judge and judge["action_type"] == "POLE_LAP")) and \
        bool(judge and judge["phase"] == "QUALIFICA") and m["voice_peak"] >= 0.6
    if not reject:
        if vis is None:
            flags.append("giudizio visivo non disponibile")
            if solo_phys and not (error_ok or pole_ok):
                reject = "auto da sola secondo il tracciamento e nessuna verifica visiva"
        else:
            solo_vis = vis["action"] in ("AUTO_SOLITARIA", "NESSUNA_AZIONE") or not vis["action_visible"]
            if solo_phys and solo_vis and not (error_ok or pole_ok):
                reject = "auto da sola, nessuna azione visibile (tracciamento e occhio AI concordi)"
            elif solo_phys != solo_vis and not (error_ok or pole_ok):
                flags.append("tracciamento e occhio AI discordanti")
    if judge is None:
        flags.append("giudizio editoriale non disponibile")
    if vis and vis["scene_type"] == "REPLAY":
        flags.append("replay della stessa azione (la diretta vale di piu')")

    physics = float(m["physics_score_30"])
    vision_pts = float(vis["visual_score"]) if vis else 8.0
    story = float(judge["story_score"]) if judge else 14.0
    audio = 7.0 * m["voice_rank"] + 4.0 * min(1.0, m["hook_voice"] / 0.75) + 4.0 * min(1.0, m["speech_rate"] / 3.6)
    total = physics + vision_pts + story + audio
    if "tracciamento e occhio AI discordanti" in flags:
        total -= 8.0
    if any(f.startswith("replay") for f in flags):
        total -= 10.0
    if vis is None:
        total = min(total, 70.0)
    if judge is None:
        total = min(total, 72.0)
    item["flags"] = flags
    item["reject_reason"] = reject
    item["verdict"] = "SCARTATA" if reject else "OK"
    item["score"] = int(clamp(round(total), 0, 100))
    item["breakdown"] = {"fisica_30": round(physics, 1), "occhio_ai_20": round(vision_pts, 1),
                         "storia_35": round(story, 1), "audio_15": round(audio, 1)}


def select_finalists(items: List[Dict[str, Any]], top_k: int, min_score: int = 0) -> List[Dict[str, Any]]:
    """Le migliori top_k fra le promosse sopra il punteggio minimo, senza due clip sullo stesso momento."""
    chosen: List[Dict[str, Any]] = []
    for it in items:
        if it["verdict"] == "OK" and it["score"] < min_score:
            it["verdict"], it["reject_reason"] = "SCARTATA", f"punteggio {it['score']} sotto il minimo {min_score}"
    for it in sorted([i for i in items if i["verdict"] == "OK"], key=lambda x: x["score"], reverse=True):
        a0, a1 = it["trim"]["t_in"], it["trim"]["t_out"]
        clash = False
        for c in chosen:
            b0, b1 = c["trim"]["t_in"], c["trim"]["t_out"]
            inter = min(a1, b1) - max(a0, b0)
            if inter > 0.3 * min(a1 - a0, b1 - b0):
                clash = True
                break
        if not clash:
            chosen.append(it)
        if len(chosen) >= top_k:
            break
    return chosen


# ---------------------------------------------------------------------------------------------------------
# AGENTE SEO COPYWRITER (zero hashtag, parole chiave integrate) e controlli deterministici
# ---------------------------------------------------------------------------------------------------------
COPY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 12},
        "hook_3s": {"type": "string", "minLength": 4},
        "caption": {"type": "string", "minLength": 80},
        "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 8, "maxItems": 16},
    },
    "required": ["title", "hook_3s", "caption", "keywords"],
}
CLICHE_PATTERNS = [
    r"chi vince", r"chi la spunta", r"non crederai", r"guarda cosa", r"da non credere", r"\bincredibil\w+",
    r"\bpazzesc\w+", r"\bassurd\w+", r"\bspettacolar\w+", r"\bepic[oa]\b", r"\bclamoros\w+", r"\bmozzafiato\b",
    r"al limite\s*!", r"\bche gara\b", r"\bda paura\b", r"\bda urlo\b", r"\bfolle\b", r"\bfollia\b",
]
ACTION_KEYWORDS = {
    "CONTATTO": ["contatto in gara", "incidente"], "PARTENZA": ["partenza", "prima curva"],
    "SORPASSO": ["sorpasso", "staccata"], "DUELLO": ["duello ruota a ruota", "bagarre"],
    "ERRORE_SALVATAGGIO": ["testacoda", "salvataggio"], "POLE_LAP": ["giro di qualifica", "pole position"],
    "VOLATA": ["volata al traguardo", "arrivo in volata"],
}
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9']*")


def strip_hashtags(text: str) -> str:
    """Regola assoluta della v11: nessun '#' in titoli, caption, hook, parole chiave e file esportati."""
    text = re.sub(r"#\s*(\w+)", r"\1", str(text))
    return re.sub(r"\s{2,}", " ", text.replace("#", "")).strip()


def _tokens(text: str) -> set:
    stop = {"alla", "allo", "della", "dello", "delle", "degli", "nella", "nello", "sulla", "sullo", "dopo", "prima",
            "con", "per", "che", "non", "una", "uno", "gli", "dei", "del", "the", "all", "sul", "nel", "tra", "fra"}
    return {strip_accents(t.lower()) for t in _WORD_RE.findall(text) if len(t) >= 3} - stop


def titles_too_similar(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    jac = len(ta & tb) / len(ta | tb)
    first_a = " ".join(strip_accents(a.lower()).split()[:3])
    first_b = " ".join(strip_accents(b.lower()).split()[:3])
    return jac >= 0.6 or (first_a == first_b and len(first_a) > 8)


def unknown_names(text: str, ctx: Ctx, allowed: set) -> List[str]:
    """Nomi propri (maiuscola a meta' frase) che non compaiono ne' nella telecronaca ne' nei dati di gara."""
    out: List[str] = []
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        if not tok[0].isupper() or len(tok) < 3:
            continue
        if tok.isupper() and len(tok) <= 5:
            continue                                   # sigle: GT3, BTCC, DRS...
        before = text[:m.start()].rstrip()
        if not before or before[-1] in ".!?:\n\"«(":
            continue                                   # inizio frase: la maiuscola e' grammaticale
        if not name_supported(tok, ctx.vocab, allowed):
            out.append(tok)
    return list(dict.fromkeys(out))


def allowed_names(ctx: Ctx) -> set:
    race = ctx.race
    blob = " ".join([race.get("circuit", ""), race.get("championship", ""), race.get("category", ""),
                     " ".join(race.get("cars", [])), " ".join(race.get("drivers", [])),
                     ctx.args.piloti or "", ctx.args.pista or "", ctx.args.campionato or ""])
    return {re.sub(r"[^a-z0-9]", "", strip_accents(t.lower())) for t in _WORD_RE.findall(blob)} - {""}


def copy_issues(ctx: Ctx, copy: Dict[str, Any], other_titles: List[str], allowed: set) -> List[str]:
    issues: List[str] = []
    title, caption = copy["title"], copy["caption"]
    if "#" in title or "#" in caption or "#" in copy["hook_3s"] or any("#" in k for k in copy["keywords"]):
        issues.append("contiene il simbolo # (hashtag vietati: riscrivi senza)")
    if len(title) < 25:
        issues.append("titolo troppo corto e generico: descrivi il fatto concreto")
    if len(title) > 85:
        issues.append("titolo troppo lungo (massimo 80 caratteri)")
    low = strip_accents(title.lower())
    for pat in CLICHE_PATTERNS:
        if re.search(pat, low):
            issues.append("titolo con formula da clickbait generico: racconta il fatto specifico")
            break
    for other in other_titles:
        if titles_too_similar(title, other):
            issues.append(f"titolo troppo simile a un altro gia' usato (\"{other}\"): cambia struttura e parole")
            break
    bad = unknown_names(title + ". " + caption, ctx, allowed)
    if bad:
        issues.append("nomi che non compaiono nei dati (toglili o sostituiscili con una descrizione): " + ", ".join(bad[:5]))
    if len(caption) < 160:
        issues.append("caption troppo corta: servono 3-5 frasi con le parole chiave integrate")
    return issues


def clean_keywords(ctx: Ctx, kws: List[str], action_type: str) -> List[str]:
    race = ctx.race
    base = [race.get("circuit", ""), race.get("championship", ""), race.get("category", "")] + \
        race.get("cars", [])[:2] + ACTION_KEYWORDS.get(action_type, []) + ["highlights gara", "motorsport"]
    out: List[str] = []
    seen: set = set()
    for k in list(kws) + base:
        k = strip_hashtags(k).strip(" .,;:-").lower()
        k = re.sub(r"\s+", " ", k)
        key = strip_accents(k)
        if 2 <= len(k) <= 60 and key not in seen:
            seen.add(key)
            out.append(k)
        if len(out) >= 15:
            break
    return out


def write_copy(ctx: Ctx, item: Dict[str, Any], other_titles: List[str], notes: str = "") -> Dict[str, Any]:
    """Titolo, hook, caption e parole chiave per una finalista. Rigenera finche' i controlli non passano (max 3)."""
    race, m, trim = ctx.race, item["metrics"], item["trim"]
    judge = item.get("judge") or {}
    vis = item.get("vision") or {}
    allowed = allowed_names(ctx)
    facts = judge.get("facts") or "; ".join(item["cand"].get("summaries", [])) or "azione di gara"
    drivers = [d for d in judge.get("drivers", []) if d.strip() and name_supported(d.split()[-1], ctx.vocab, allowed)]
    action = judge.get("action_type") if judge.get("action_type") not in (None, "NESSUNA") else item["cand"]["action_type"]
    transcript = text_in_range(ctx.segments, trim["t_in"], trim["t_out"], max_chars=1600)
    system = (
        "Sei un copywriter SEO senior per canali di motorsport e sim racing su Instagram Reels, TikTok e YouTube Shorts. "
        "Scrivi in italiano naturale, da giornalista sportivo. REGOLE FISSE: 1) ZERO hashtag: il simbolo # non deve "
        "comparire mai. 2) Il titolo nasce dal fatto concreto della clip (chi, dove, come), in forma di frase normale "
        "con la maiuscola solo all'inizio e nei nomi propri, senza emoji. 3) Vietate le formule generiche da clickbait "
        "(\"Sorpasso al limite! Chi vince?\", \"Incredibile\", \"Non crederai\"). 4) Usa solo nomi presenti nei DATI: "
        "se un pilota o una curva non sono indicati, descrivi senza nominare. 5) Le parole chiave di ricerca vanno "
        "integrate nel testo in modo naturale, non elencate."
    )
    base_user = (
        "DATI DELLA CLIP\n"
        f"- Pista: {race.get('circuit') or 'non nota (non citarla)'}\n"
        f"- Campionato / evento: {race.get('championship') or 'non noto (non citarlo)'}\n"
        f"- Categoria o simulatore: {race.get('category') or 'non noto'}\n"
        f"- Auto: {', '.join(race.get('cars', [])) or 'non note'}\n"
        f"- Tipo di azione: {action}\n"
        f"- Che cosa succede: {facts}\n"
        f"- Piloti coinvolti: {', '.join(drivers) or 'non identificati (non inventarli)'}\n"
        f"- Sensori video: {describe_metrics(m)}\n"
        f"- Cosa si vede: {vis.get('description', 'non disponibile')}\n"
        f"- Parole del telecronista: \"{transcript}\"\n"
        f"- Durata: {trim['dur']:.0f} secondi\n"
        f"- TITOLI GIA' USATI PER ALTRE CLIP (non ripeterne struttura ne' parole): {json.dumps(other_titles, ensure_ascii=False)}\n\n"
        "SCRIVI\n"
        "title: 40-75 caratteri, specifico, alto CTR. Esempi di stile (non copiarli): \"Tre auto affiancate sull'erba "
        "all'ultimo giro a Oulton Park\", \"Contatto in staccata: perde il controllo ma la salva di traverso\".\n"
        "hook_3s: testo breve (massimo 38 caratteri) da sovrapporre nei primi 3 secondi.\n"
        "caption: 3-5 frasi (350-650 caratteri) che raccontano il fatto e integrano le parole chiave cercate "
        "(pista, categoria, tipo di vettura, tecnica di guida, staccata, sorpasso, duello ruota a ruota...), "
        "chiusa da una domanda tecnica concreta ai follower.\n"
        "keywords: 10-14 parole chiave e long-tail, senza #, pronte per i tag di YouTube Studio e il testo SEO di "
        "TikTok/Instagram (es. \"sorpasso all'esterno\", \"duello ruota a ruota\", nome pista + categoria).\n"
        f"Rispondi SOLO con JSON conforme a: {schema_hint(COPY_SCHEMA)}"
    )
    feedback = f"\n\nNOTE DEL REVISORE DA RISOLVERE: {notes}" if notes else ""
    best: Optional[Dict[str, Any]] = None
    best_issues: List[str] = ["nessun testo generato"]
    for attempt in range(3):
        res = ctx.llm.chat_json(ctx.models["copy"], system, base_user + feedback, COPY_SCHEMA,
                                temperature=0.6 if attempt == 0 else 0.75, num_predict=1100,
                                label=f"copy clip {item['id']}")
        if not res:
            continue
        issues = copy_issues(ctx, res, other_titles, allowed)
        if best is None or len(issues) < len(best_issues):
            best, best_issues = res, issues
        if not issues:
            break
        feedback = ("\n\nLA VERSIONE PRECEDENTE E' STATA RESPINTA. Problemi da correggere:\n- " + "\n- ".join(issues) +
                    f"\nTitolo respinto: \"{res['title']}\". Riscrivi tutto rispettando le regole." +
                    (f"\nNOTE DEL REVISORE: {notes}" if notes else ""))
    if best is None:
        # ripiego deterministico: testo sobrio costruito dai fatti, mai un cliche'
        where = f" a {race['circuit']}" if race.get("circuit") else ""
        title = strip_hashtags(facts.split(".")[0])[:70] or f"Azione di gara{where}"
        best = {"title": title, "hook_3s": "Guarda la manovra", "caption": strip_hashtags(facts), "keywords": []}
        best_issues = ["testo generato in automatico: il copywriter non ha risposto"]
    copy = {
        "title": strip_hashtags(best["title"]).strip(" \"'"),
        "hook_3s": strip_hashtags(best["hook_3s"]).strip(" \"'")[:48],
        "caption": strip_hashtags(best["caption"]),
        "keywords": clean_keywords(ctx, best.get("keywords", []), action),
        "warnings": [i for i in best_issues if not i.startswith("contiene il simbolo #")],
    }
    return copy


# ---------------------------------------------------------------------------------------------------------
# VERIFICA 3 - Comitato editoriale sulle finaliste
# ---------------------------------------------------------------------------------------------------------
AUDIT_SCHEMA = {
    "type": "object",
    "properties": {"reviews": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "score_comment": {"type": "string", "default": ""},
            "final_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "title_verdict": {"type": "string", "enum": ["OK", "DEBOLE", "GENERICO", "ERRORE_FATTUALE", "DUPLICATO"]},
            "title_note": {"type": "string", "default": ""},
            "caption_verdict": {"type": "string", "enum": ["OK", "DEBOLE", "ERRORE_FATTUALE"]},
            "caption_note": {"type": "string", "default": ""},
        },
        "required": ["id", "final_score", "title_verdict", "caption_verdict"]}}},
    "required": ["reviews"],
}


def editorial_committee(ctx: Ctx, finalists: List[Dict[str, Any]]) -> None:
    """
    Revisore di famiglia diversa dal copywriter: confronta le finaliste fra loro, ricalibra i punteggi entro un
    margine, segnala titoli e caption deboli, generici, duplicati o con errori di fatto. I testi segnalati
    vengono rigenerati e ricontrollati; l'unicita' dei titoli e l'assenza di '#' sono poi garantite dal codice.
    """
    if not finalists:
        return
    rows = []
    for it in finalists:
        rows.append({
            "id": it["id"], "punteggio": it["score"], "dettaglio": it["breakdown"],
            "fatti": (it.get("judge") or {}).get("facts", ""), "sensori": describe_metrics(it["metrics"]),
            "titolo": it["copy"]["title"], "caption": it["copy"]["caption"][:420],
        })
    system = (
        "Sei il caporedattore: revisione finale e comparativa delle clip scelte. Controlli quattro cose. "
        "1) Punteggi: devono essere coerenti fra loro e con i fatti (un contatto o un sorpasso confermato dai sensori "
        "vale piu' di un inseguimento); correggi final_score dove serve, usando tutta la scala e senza pari merito. "
        "2) Titoli: tutti diversi per struttura e parole, specifici, senza formule generiche. "
        "3) Errori di fatto: titolo e caption non devono affermare cio' che i fatti e i sensori non dicono, ne' "
        "contenere nomi assenti dai fatti. 4) Nessun hashtag. Sii severo ma non riscrivere tu: indica il problema."
    )
    user = ("CLIP FINALISTE:\n" + json.dumps(rows, ensure_ascii=False, indent=1) +
            "\n\nPer ogni clip: id, score_comment, final_score (0-100), title_verdict (OK, DEBOLE, GENERICO, "
            "ERRORE_FATTUALE, DUPLICATO), title_note, caption_verdict (OK, DEBOLE, ERRORE_FATTUALE), caption_note.\n"
            f"Rispondi SOLO con JSON conforme a: {schema_hint(AUDIT_SCHEMA)}")
    res = ctx.llm.chat_json(ctx.models["text"], system, user, AUDIT_SCHEMA, temperature=0.1,
                            num_predict=260 * len(finalists) + 300, keep_alive=0, label="comitato editoriale")
    reviews = {int(r["id"]): r for r in (res or {}).get("reviews", [])}
    if res is None:
        warn("comitato editoriale non disponibile: restano validi i controlli deterministici")
    flagged: List[Tuple[Dict[str, Any], str]] = []
    for it in finalists:
        r = reviews.get(it["id"])
        it["audit"] = {"reviewed": bool(r), "original_score": it["score"]}
        if not r:
            continue
        adjusted = int(clamp(r["final_score"], it["score"] - 12, it["score"] + 8))
        it["audit"].update({"score_comment": r.get("score_comment", ""), "title_verdict": r["title_verdict"],
                            "caption_verdict": r["caption_verdict"]})
        it["score"] = adjusted
        notes = []
        if r["title_verdict"] != "OK":
            notes.append(f"titolo {r['title_verdict'].lower().replace('_', ' ')}: {r.get('title_note', '')}")
        if r["caption_verdict"] != "OK":
            notes.append(f"caption {r['caption_verdict'].lower().replace('_', ' ')}: {r.get('caption_note', '')}")
        if notes:
            flagged.append((it, " | ".join(notes)))

    if flagged:
        log(f" -> Il comitato chiede di riscrivere {len(flagged)} testi: rigenero e ricontrollo.")
        if ctx.models["copy"] != ctx.models["text"]:
            ctx.llm.unload_all()
        for it, notes in flagged:
            others = [x["copy"]["title"] for x in finalists if x is not it]
            new = write_copy(ctx, it, others, notes=notes)
            if not new["warnings"] or len(new["warnings"]) <= len(it["copy"].get("warnings", [])):
                it["copy"] = new
                it["audit"]["rewritten"] = True
                log(f"    clip {it['id']:02d}: nuovo titolo -> {new['title']}")
            else:
                it["audit"]["rewritten"] = False
                log(f"    clip {it['id']:02d}: la riscrittura non supera i controlli ({new['warnings'][0]}), resta il testo precedente")

    # Garanzie finali fatte dal codice, non dal modello.
    finalists.sort(key=lambda x: x["score"], reverse=True)
    prev = 101
    for it in finalists:                        # nessun pari merito, ordine coerente con la classifica
        it["score"] = int(clamp(min(it["score"], prev - 1), 0, 100))
        prev = it["score"]
    seen: List[str] = []
    for it in finalists:
        title = strip_hashtags(it["copy"]["title"])
        if any(titles_too_similar(title, s) for s in seen):
            title = f"{title} (minuto {int(it['trim']['t_in'] // 60)} della diretta)"
            it["copy"].setdefault("warnings", []).append("titolo reso unico in automatico: rivederlo a mano")
        it["copy"]["title"] = title
        seen.append(title)


# ---------------------------------------------------------------------------------------------------------
# AGENTE COMPOSITORE - Regia 9:16: layout per inquadratura, camere virtuali, effetti
# ---------------------------------------------------------------------------------------------------------
MODE_FULL, MODE_REACTION, MODE_WIDE, MODE_DUAL = 0, 1, 2, 3
MODE_NAMES = {MODE_FULL: "9:16 pieno", MODE_REACTION: "Reaction split (webcam + azione)",
              MODE_WIDE: "Vista larga + zoom", MODE_DUAL: "Dual (auto A sopra, auto B sotto)"}
TOP_NONE, TOP_WEBCAM, TOP_WIDE, TOP_CAR = 0, 1, 2, 3
MODE_TOP_KIND = {MODE_FULL: TOP_NONE, MODE_REACTION: TOP_WEBCAM, MODE_WIDE: TOP_WIDE, MODE_DUAL: TOP_CAR}


@dataclass
class SegPlan:
    n_in: int
    nf: int
    mode: np.ndarray
    top_h: np.ndarray
    kind_a: np.ndarray
    kind_b: np.ndarray
    alpha: np.ndarray
    bcx: np.ndarray
    bcy: np.ndarray
    bcw: np.ndarray
    tcx: np.ndarray
    tcy: np.ndarray
    tcw: np.ndarray
    shx: np.ndarray
    shy: np.ndarray


def _box_array(lst: List[List[float]]) -> np.ndarray:
    return np.array([b if b else [np.nan] * 4 for b in lst], dtype=np.float64).reshape(-1, 4)


def _interp_by_shot(fa: np.ndarray, shot_f: np.ndarray, s_f: np.ndarray, s_shot: np.ndarray,
                    values: np.ndarray) -> np.ndarray:
    """Interpola una serie campionata sui frame di uscita, senza mai attraversare uno stacco di camera."""
    out = np.full(len(fa), np.nan)
    for sh in np.unique(shot_f):
        fm = shot_f == sh
        sm = (s_shot == sh) & ~np.isnan(values)
        if sm.any():
            out[fm] = np.interp(fa[fm], s_f[sm], values[sm])
    return out


def _smooth_by_shot(values: np.ndarray, shot_f: np.ndarray, sigma: float, default: float) -> np.ndarray:
    out = np.empty(len(values))
    for sh in np.unique(shot_f):
        fm = shot_f == sh
        out[fm] = gaussian_smooth(fill_nan(values[fm], default), sigma)
    return out


# Quanto "largo" inquadra ogni layout: un tratto troppo corto confluisce nel vicino piu' largo, mai in uno
# piu' stretto, cosi' un'auto non esce dal quadro per far durare di piu' un layout.
MODE_WIDENESS = {MODE_FULL: 0, MODE_REACTION: 1, MODE_WIDE: 2, MODE_DUAL: 3}


def _settle_modes(desired: np.ndarray, shot_f: np.ndarray, min_run: int) -> np.ndarray:
    """Dentro ogni inquadratura un layout dura almeno min_run frame: i tratti piu' corti confluiscono nel vicino."""
    out = desired.copy()
    for sh in np.unique(shot_f):
        idx = np.where(shot_f == sh)[0]
        seg = out[idx[0]:idx[-1] + 1]
        while True:
            change = np.flatnonzero(np.diff(seg)) + 1
            starts = np.concatenate([[0], change])
            ends = np.concatenate([change, [len(seg)]])
            lengths = ends - starts
            if len(lengths) <= 1:
                break
            i = int(np.argmin(lengths))
            if lengths[i] >= min_run:
                break
            left = int(seg[starts[i] - 1]) if i > 0 else None
            right = int(seg[ends[i]]) if i + 1 < len(lengths) else None
            if left is None or right is None:
                target = right if left is None else left
            elif MODE_WIDENESS[left] != MODE_WIDENESS[right]:
                target = left if MODE_WIDENESS[left] > MODE_WIDENESS[right] else right
            else:
                target = left if lengths[i - 1] >= lengths[i + 1] else right
            seg[starts[i]:ends[i]] = target
        out[idx[0]:idx[-1] + 1] = seg
    return out


def top_height(mode: int, wide_h: int) -> int:
    return {MODE_FULL: 0, MODE_REACTION: REACTION_TOP_H, MODE_WIDE: wide_h, MODE_DUAL: DUAL_TOP_H}[int(mode)]


def plan_segment(ctx: Ctx, an: Dict[str, Any], n_in: int, n_out: int, events: Dict[str, Any],
                 webcam_ok: bool) -> SegPlan:
    info = ctx.info
    W, H, fps = float(info.width), float(info.height), info.fps
    nf = n_out - n_in
    wide_h = even(OUT_W * H / W)
    fit = TUNING["full_fit"]
    layout = ctx.args.layout
    max_zoom = float(ctx.args.max_zoom)

    fa = (np.arange(n_in, n_out) - an["n0"]).astype(np.float64)
    cuts = np.array(an["cuts"], dtype=np.float64)
    shot_f = np.searchsorted(cuts, fa, side="right")
    s_f = np.array(an["s_f"], dtype=np.float64)
    s_shot = np.searchsorted(cuts, s_f, side="right")
    F, A, B = _box_array(an["s_focus"]), _box_array(an["s_a"]), _box_array(an["s_b"])

    def series(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(box) == 0:
            nanv = np.full(nf, np.nan)
            return nanv, nanv.copy(), nanv.copy(), nanv.copy()
        x1 = _interp_by_shot(fa, shot_f, s_f, s_shot, box[:, 0]) * W
        y1 = _interp_by_shot(fa, shot_f, s_f, s_shot, box[:, 1]) * H
        x2 = _interp_by_shot(fa, shot_f, s_f, s_shot, box[:, 2]) * W
        y2 = _interp_by_shot(fa, shot_f, s_f, s_shot, box[:, 3]) * H
        return x1, y1, x2, y2

    fx1, fy1, fx2, fy2 = series(F)
    ax1, ay1, ax2, ay2 = series(A)
    bx1, by1, bx2, by2 = series(B)

    # la coppia vale solo vicino a un campione reale (mezzo secondo), non per estrapolazione
    pair_valid = np.zeros(nf, bool)
    pidx = np.where(~np.isnan(A[:, 0]))[0] if len(A) else np.zeros(0, int)
    if len(pidx):
        pf, ps = s_f[pidx], s_shot[pidx]
        pos = np.searchsorted(pf, fa)
        left, right = np.clip(pos - 1, 0, len(pf) - 1), np.clip(pos, 0, len(pf) - 1)
        dl, dr = np.abs(fa - pf[left]), np.abs(pf[right] - fa)
        near = np.where(dl <= dr, left, right)
        pair_valid = (np.minimum(dl, dr) <= 0.5 * fps) & (ps[near] == shot_f) & ~np.isnan(ax1) & ~np.isnan(bx1)

    ux1, ux2 = np.fmin(ax1, bx1), np.fmax(ax2, bx2)
    uy1, uy2 = np.fmin(ay1, by1), np.fmax(ay2, by2)
    span = (ux2 - ux1) * 1.08
    wa, wb = ax2 - ax1, bx2 - bx1
    ha, hb = ay2 - ay1, by2 - by1
    with np.errstate(invalid="ignore", divide="ignore"):
        area_ratio = np.fmax(wa * ha, wb * hb) / np.fmax(1.0, np.fmin(wa * ha, wb * hb))
        same_row = np.abs((ay1 + ay2) / 2 - (by1 + by2) / 2) < 1.5 * (ha + hb) / 2
        dual_ok = pair_valid & (np.fmin(wa, wb) >= 0.035 * W) & (area_ratio < 3.5) & same_row

    onboard_f = np.zeros(nf, bool)
    for sh in an.get("shots", []):
        if sh.get("onboard"):
            onboard_f |= (fa >= sh["start"]) & (fa < sh["end"])

    full_w = min(W, H * OUT_W / OUT_H)
    panel_w = min(W, H * OUT_W / (OUT_H - REACTION_TOP_H))
    base_mode = MODE_REACTION if webcam_ok else MODE_FULL
    desired = np.full(nf, base_mode, dtype=np.int64)
    if layout == "full":
        desired[:] = MODE_FULL
    elif layout == "wide":
        desired[:] = MODE_WIDE
    elif layout == "reaction":
        desired[:] = MODE_REACTION if ctx.webcam else MODE_WIDE
    else:
        with np.errstate(invalid="ignore"):
            fits_full = span <= full_w * fit
            fits_panel = span <= panel_w * fit
        if layout == "dual":
            desired[dual_ok] = MODE_DUAL
        elif webcam_ok:
            desired[pair_valid & ~fits_panel & dual_ok] = MODE_DUAL
        else:
            desired[pair_valid & ~fits_full & fits_panel] = MODE_WIDE
            desired[pair_valid & ~fits_panel & dual_ok] = MODE_DUAL
            desired[pair_valid & ~fits_panel & ~dual_ok] = MODE_WIDE
        # camera onboard: vista larga (o webcam) sopra e inquadratura centrale sotto, senza inseguire l'auto davanti
        desired[onboard_f] = MODE_REACTION if webcam_ok else MODE_WIDE
    mode = _settle_modes(desired, shot_f, max(2, int(round(TUNING["mode_hold"] * fps))))

    # Nessun cambio di layout a cavallo del momento clou: se un confine cade nell'intorno di un evento
    # (zoom punch o impatto) viene anticipato all'inizio dell'intorno, cosi' l'animazione e' gia' finita.
    t_axis0 = np.arange(n_in, n_out) / fps
    for t_evt, _rise in events.get("punch", []):
        lo_i = int(np.searchsorted(t_axis0, t_evt - 1.2))
        hi_i = int(np.searchsorted(t_axis0, t_evt + 1.6))
        for j0 in np.flatnonzero(np.diff(mode)) + 1:
            if lo_i < j0 < hi_i and shot_f[j0] == shot_f[j0 - 1] and shot_f[max(0, lo_i)] == shot_f[j0]:
                mode[max(0, lo_i):j0] = mode[j0]

    top_h = np.array([top_height(m, wide_h) for m in mode], dtype=np.float64)
    w_dual = (mode == MODE_DUAL).astype(np.float64)
    kind_a = np.array([MODE_TOP_KIND[int(m)] for m in mode], dtype=np.int64)
    kind_b = kind_a.copy()
    alpha = np.ones(nf)
    morph = max(2, int(round(TUNING["morph_s"] * fps)))
    for j0 in np.flatnonzero(np.diff(mode)) + 1:
        if shot_f[j0] != shot_f[j0 - 1]:
            continue                       # il layout cambia sullo stacco: taglio netto, nessuna animazione
        m1, m2 = int(mode[j0 - 1]), int(mode[j0])
        h1, h2 = top_height(m1, wide_h), top_height(m2, wide_h)
        a0 = j0 - morph // 2
        for j in range(max(0, a0), min(nf, a0 + morph)):
            if shot_f[j] != shot_f[j0]:
                continue
            e = float(smoothstep(np.array((j - a0 + 0.5) / morph)))
            top_h[j] = h1 + (h2 - h1) * e
            w_dual[j] = (1.0 if m1 == MODE_DUAL else 0.0) * (1 - e) + (1.0 if m2 == MODE_DUAL else 0.0) * e
            kind_a[j], kind_b[j], alpha[j] = MODE_TOP_KIND[m1], MODE_TOP_KIND[m2], e

    # --- camera dell'azione (9:16 pieno o pannello inferiore) ---
    nominal_h = np.array([top_height(m if m != MODE_DUAL else base_mode, wide_h) for m in mode], dtype=np.float64)
    aspect = OUT_W / (OUT_H - nominal_h)
    crop_h0 = np.minimum(H, W / aspect)
    crop_w0 = crop_h0 * aspect
    with np.errstate(invalid="ignore"):
        use_union = pair_valid & (span <= crop_w0 * fit)
    tx1, tx2 = np.where(use_union, ux1, fx1), np.where(use_union, ux2, fx2)
    ty1, ty2 = np.where(use_union, uy1, fy1), np.where(use_union, uy2, fy2)
    if layout == "auto":
        tx1[onboard_f] = tx2[onboard_f] = ty1[onboard_f] = ty2[onboard_f] = np.nan   # onboard: camera ferma al centro
    tcx_raw, tcy_raw = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    tw, th = tx2 - tx1, ty2 - ty1
    with np.errstate(invalid="ignore", divide="ignore"):
        # np.maximum/np.minimum propagano i NaN: dove non c'e' un soggetto lo zoom resta 1.0
        z_fit = np.minimum(crop_w0 * 0.60 / np.maximum(tw, 1.0), crop_h0 * 0.50 / np.maximum(th, 1.0))
    z_auto = np.clip(np.where(np.isnan(z_fit), 1.0, z_fit), 1.0, min(max_zoom, 1.25))
    z_auto = _smooth_by_shot(z_auto, shot_f, 0.8 * fps, 1.0)

    # Zoom punch: salita ease-in-out fino al momento clou, breve tenuta, rilascio morbido
    t_axis = (np.arange(n_in, n_out)) / fps
    punch = np.zeros(nf)
    for t_evt, rise in events.get("punch", []):
        up = smoothstep((t_axis - (t_evt - rise)) / max(rise, 1e-3))
        down = 1.0 - smoothstep((t_axis - (t_evt + 0.45)) / 1.0)
        punch = np.maximum(punch, np.minimum(up, down))
    punch *= TUNING["punch_amp"] if not ctx.args.no_effects else 0.0
    zoom = np.minimum(z_auto * (1.0 + punch), max(max_zoom, 1.0) * (1.0 + TUNING["punch_amp"]))

    acw = crop_w0 / zoom
    ach = crop_h0 / zoom
    acx = _smooth_by_shot(tcx_raw, shot_f, 0.45 * fps, W / 2)
    acy = _smooth_by_shot(tcy_raw, shot_f, 0.60 * fps, H * 0.5)
    # contenimento: il soggetto resta dentro il crop con un margine, poi una lisciata leggera toglie gli spigoli
    margin = 0.04 * acw
    with np.errstate(invalid="ignore"):
        lo, hi = tx2 - acw / 2 + margin, tx1 + acw / 2 - margin
        ok = ~np.isnan(lo) & (lo <= hi)
        acx = np.where(ok, np.clip(acx, np.where(ok, lo, acx), np.where(ok, hi, acx)), acx)
        lo_y, hi_y = ty2 - ach / 2 + 0.04 * ach, ty1 + ach / 2 - 0.04 * ach
        ok_y = ~np.isnan(lo_y) & (lo_y <= hi_y)
        acy = np.where(ok_y, np.clip(acy, np.where(ok_y, lo_y, acy), np.where(ok_y, hi_y, acy)), acy)
    acx = _smooth_by_shot(acx, shot_f, 0.12 * fps, W / 2)
    acy = _smooth_by_shot(acy, shot_f, 0.12 * fps, H * 0.5)

    # --- camere del Dual: un pannello per auto, stessa scala ---
    dual_aspect = OUT_W / (OUT_H - DUAL_TOP_H)
    with np.errstate(invalid="ignore"):
        dcw_raw = np.clip(2.8 * np.fmax(wa, wb), 0.20 * W, min(0.55 * W, H * dual_aspect))
    dcw = _smooth_by_shot(dcw_raw, shot_f, 0.8 * fps, 0.32 * W) / (1.0 + 0.6 * punch)
    dax = _smooth_by_shot((ax1 + ax2) / 2, shot_f, 0.35 * fps, W / 2)
    day = _smooth_by_shot((ay1 + ay2) / 2, shot_f, 0.50 * fps, H * 0.5)
    dbx = _smooth_by_shot((bx1 + bx2) / 2, shot_f, 0.35 * fps, W / 2)
    dby = _smooth_by_shot((by1 + by2) / 2, shot_f, 0.50 * fps, H * 0.5)

    bcx = acx * (1 - w_dual) + dbx * w_dual
    bcy = acy * (1 - w_dual) + dby * w_dual
    bcw = acw * (1 - w_dual) + dcw * w_dual

    # Micro-shake sull'impatto: 0,25 s, oscillazione smorzata
    shx, shy = np.zeros(nf), np.zeros(nf)
    if not ctx.args.no_effects:
        dur = TUNING["shake_s"]
        for k, t_evt in enumerate(events.get("shake", [])):
            tau = t_axis - t_evt
            on = (tau >= 0) & (tau <= dur)
            decay = np.where(on, (1.0 - tau / dur) ** 2, 0.0)
            amp = 0.012 * H * decay
            shx += amp * np.sin(2 * np.pi * 24.0 * tau + 1.3 * (k + 1))
            shy += amp * np.cos(2 * np.pi * 19.0 * tau + 0.7 * (k + 1))
    return SegPlan(n_in=n_in, nf=nf, mode=mode, top_h=top_h, kind_a=kind_a, kind_b=kind_b, alpha=alpha,
                   bcx=bcx, bcy=bcy, bcw=bcw, tcx=dax, tcy=day, tcw=dcw, shx=shx, shy=shy)


class Compositor:
    """Compone il frame 1080x1920 a partire dal frame sorgente e dal piano di regia."""

    def __init__(self, ctx: Ctx):
        info = ctx.info
        self.W, self.H = info.width, info.height
        self.effects = not ctx.args.no_effects
        # vignettatura solo sui bordi verticali (sinistra/destra): il centro pista resta pieno
        m = int(OUT_W * 0.15)
        x = np.arange(m, dtype=np.float32)
        gain = 1.0 - TUNING["vignette"] * 0.5 * (1.0 + np.cos(np.pi * x / m))
        self.vig_m = m
        self.vig_left = np.repeat(np.tile(gain[None, :, None], (OUT_H, 1, 1)), 3, axis=2).astype(np.float32)
        self.vig_right = np.ascontiguousarray(self.vig_left[:, ::-1])
        # linea di separazione: anima piena + alone
        color = np.array(ctx.args.divider_rgb[::-1], dtype=np.float32)        # BGR
        half = 16
        dy = np.arange(-half, half + 1, dtype=np.float32)
        glow = 0.55 * np.exp(-(dy / 7.0) ** 2)
        self.div_half = half
        self.div_glow = np.clip(glow[:, None, None] * color[None, None, :], 0, 255).astype(np.uint8)
        self.div_glow = np.ascontiguousarray(np.repeat(self.div_glow, OUT_W, axis=1))
        self.div_color = tuple(int(c) for c in color)
        self.webcam = None
        if ctx.webcam:
            x1, y1, x2, y2 = ctx.webcam["rect"]
            px1, py1, px2, py2 = ctx.webcam.get("person") or ctx.webcam["rect"]
            self.webcam = (x1 * self.W, y1 * self.H, x2 * self.W, y2 * self.H,
                           (px1 + px2) / 2 * self.W, (py1 + py2) / 2 * self.H)

    def _crop(self, src: np.ndarray, x0: float, y0: float, cw: float, ch: float, ow: int, oh: int) -> np.ndarray:
        sx, sy = cw / ow, ch / oh
        if max(sx, sy) > 1.15:
            # riduzione: crop intero + media d'area (niente aliasing)
            xi, yi = int(round(x0)), int(round(y0))
            wi, hi = max(2, int(round(cw))), max(2, int(round(ch)))
            xi, yi = max(0, min(self.W - wi, xi)), max(0, min(self.H - hi, yi))
            return cv2.resize(src[yi:yi + hi, xi:xi + wi], (ow, oh), interpolation=cv2.INTER_AREA)
        # ingrandimento: trasformazione affine a precisione sub-pixel (panoramiche fluide) con Lanczos4
        mat = np.array([[sx, 0.0, x0 + 0.5 * sx - 0.5], [0.0, sy, y0 + 0.5 * sy - 0.5]], dtype=np.float32)
        return cv2.warpAffine(src, mat, (ow, oh), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def _camera(self, src: np.ndarray, cx: float, cy: float, cw: float, ow: int, oh: int,
                dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        aspect = ow / oh
        cw = float(min(cw, self.W, self.H * aspect))
        ch = cw / aspect
        x0 = clamp(cx - cw / 2 + dx, 0.0, self.W - cw)
        y0 = clamp(cy - ch / 2 + dy, 0.0, self.H - ch)
        return self._crop(src, x0, y0, cw, ch, ow, oh)

    def _top(self, src: np.ndarray, kind: int, th: int, plan: SegPlan, j: int) -> np.ndarray:
        if kind == TOP_WEBCAM and self.webcam is not None:
            x1, y1, x2, y2, pcx, pcy = self.webcam
            rw, rh = x2 - x1, y2 - y1
            aspect = OUT_W / th
            if rw / rh > aspect:
                ch, cw = rh, rh * aspect
            else:
                cw, ch = rw, rw / aspect
            x0 = clamp(pcx - cw / 2, x1, x2 - cw)
            y0 = clamp(pcy - ch * 0.45, y1, y2 - ch)
            return self._crop(src, x0, y0, cw, ch, OUT_W, th)
        if kind == TOP_CAR:
            return self._camera(src, plan.tcx[j], plan.tcy[j], plan.tcw[j], OUT_W, th, plan.shx[j], plan.shy[j])
        # vista larga: l'intero frame della diretta (con HUD e classifica) adattato alla larghezza
        aspect = OUT_W / th
        cw = min(self.W, self.H * aspect)
        ch = cw / aspect
        return self._crop(src, (self.W - cw) / 2, (self.H - ch) / 2, cw, ch, OUT_W, th)

    def frame(self, src: np.ndarray, plan: SegPlan, j: int) -> np.ndarray:
        th = even(plan.top_h[j])
        if th < 12:
            th = 0
        th = min(th, OUT_H - 400)
        bh = OUT_H - th
        canvas = np.empty((OUT_H, OUT_W, 3), np.uint8)
        canvas[th:] = self._camera(src, plan.bcx[j], plan.bcy[j], plan.bcw[j], OUT_W, bh, plan.shx[j], plan.shy[j])
        if th > 0:
            ka, kb, al = int(plan.kind_a[j]), int(plan.kind_b[j]), float(plan.alpha[j])
            if ka == kb or ka == TOP_NONE or kb == TOP_NONE:
                canvas[:th] = self._top(src, kb if kb != TOP_NONE else ka, th, plan, j)
            else:
                canvas[:th] = cv2.addWeighted(self._top(src, ka, th, plan, j), 1.0 - al,
                                              self._top(src, kb, th, plan, j), al, 0.0)
        if self.effects:
            m = self.vig_m
            canvas[:, :m] = cv2.multiply(canvas[:, :m], self.vig_left, dtype=cv2.CV_8U)
            canvas[:, OUT_W - m:] = cv2.multiply(canvas[:, OUT_W - m:], self.vig_right, dtype=cv2.CV_8U)
        if th > 0:
            h = self.div_half
            y0, y1 = max(0, th - h), min(OUT_H, th + h + 1)
            g0 = y0 - (th - h)
            canvas[y0:y1] = cv2.add(canvas[y0:y1], self.div_glow[g0:g0 + (y1 - y0)])
            canvas[max(0, th - 2):th + 2] = self.div_color
        return canvas


def apply_transition(frame: np.ndarray, kind: str, phase: float, outgoing: bool) -> np.ndarray:
    """
    Transizione fra due momenti uniti (es. azione in diretta + replay). phase va da 0 (lontano dal taglio)
    a 1 (sul taglio). 'flash': bagliore morbido. 'whip': strisciata orizzontale con sfocatura di movimento.
    """
    if phase <= 0.0:
        return frame
    if kind == "whip":
        k = int(4 + 150 * phase) | 1
        blurred = cv2.blur(frame, (k, 1))
        shift = int(OUT_W * 0.45 * phase * phase) * (-1 if outgoing else 1)
        return np.roll(blurred, shift, axis=1)
    a = 0.85 * phase ** 1.5
    white = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 1.0 - a, white, a, 0.0)


def layout_report(plans: List[SegPlan]) -> Dict[str, Any]:
    modes = np.concatenate([p.mode for p in plans]) if plans else np.zeros(0, int)
    total = max(1, len(modes))
    share = {MODE_NAMES[m]: round(float((modes == m).sum()) / total, 3) for m in MODE_NAMES if (modes == m).any()}
    switches = int(sum(int((np.diff(p.mode) != 0).sum()) for p in plans))
    main = max(share, key=share.get) if share else MODE_NAMES[MODE_FULL]
    return {"principale": main, "quota": share, "cambi_layout": switches}


# ---------------------------------------------------------------------------------------------------------
# FASE 7 - Render: 9:16 composto (pipe rawvideo) + master 16:9, audio a -14 LUFS in due passate
# ---------------------------------------------------------------------------------------------------------
def grade_params(luma: List[float]) -> Tuple[float, float, float]:
    """Grading adattivo e prudente: contrasto, gamma e saturazione dipendono dalla luce misurata nella clip."""
    p5, p50, p95, sat = luma
    spread = max(1.0, p95 - p5) / 255.0
    contrast = clamp(1.03 + 0.10 * (0.72 - spread) / 0.30, 1.02, 1.10)
    mid = p50 / 255.0
    if mid < 0.36:
        gamma = clamp(1.0 + (0.36 - mid) * 0.9, 1.0, 1.15)       # gara notturna o scura: apre i mezzitoni
    elif mid > 0.62:
        gamma = clamp(1.0 - (mid - 0.62) * 0.6, 0.92, 1.0)
    else:
        gamma = 1.0
    s = sat / 255.0
    saturation = 1.10 if s < 0.25 else 1.06 if s < 0.40 else 1.03
    return round(contrast, 3), round(gamma, 3), saturation


def _audio_inputs(info: VideoInfo, segs: List[List[int]]) -> List[str]:
    args: List[str] = []
    for a, b in segs:
        args += ["-ss", f"{a / info.fps:.6f}", "-t", f"{(b - a) / info.fps + 0.30:.6f}", "-i", str(info.path)]
    return args


def _audio_graph(durs: List[float], base_idx: int, tail: str) -> str:
    """Ritaglio esatto di ogni segmento, micro-dissolvenze sulle giunzioni, compressore, poi `tail` (loudnorm)."""
    parts: List[str] = []
    n = len(durs)
    for i, d in enumerate(durs):
        chain = (f"[{base_idx + i}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                 f"atrim=0:{d:.6f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={0.03 if i == 0 else 0.06}")
        if i < n - 1:
            chain += f",afade=t=out:st={max(0.0, d - 0.06):.4f}:d=0.06"
        parts.append(chain + f"[s{i}]")
    src = "[s0]"
    if n > 1:
        parts.append("".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[cat]")
        src = "[cat]"
    parts.append(f"{src}highpass=f=45,acompressor=threshold=-18dB:ratio=3:attack=8:release=160:makeup=1.5[cmp]")
    parts.append(f"[cmp]{tail}[a]")
    return ";".join(parts)


LOUD_TARGET = "I=-14:TP=-1.0:LRA=9"


def measure_loudness(ctx: Ctx, segs: List[List[int]], log_path: Path) -> Optional[Dict[str, str]]:
    """Prima passata di loudnorm: misura il programma gia' compresso. La seconda applica un guadagno lineare."""
    info = ctx.info
    durs = [(b - a) / info.fps for a, b in segs]
    graph = _audio_graph(durs, 0, f"loudnorm={LOUD_TARGET}:print_format=json")
    cmd = (["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info"] + _audio_inputs(info, segs) +
           ["-filter_complex", graph, "-map", "[a]", "-f", "null", "-"])
    try:
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             timeout=900, creationflags=_no_window_flags())
    except Exception as e:
        warn(f"misura loudness non riuscita ({e}): uso la normalizzazione a passata singola")
        return None
    text = res.stderr.decode("utf-8", "replace")
    with open(log_path, "ab") as lf:
        lf.write(("\n$ " + " ".join(cmd) + "\n" + text[-1500:]).encode("utf-8", "replace"))
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", text, flags=re.S)
    if res.returncode != 0 or not blocks:
        return None
    try:
        data = json.loads(blocks[-1])
        vals = {k: float(data[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except Exception:
        return None
    if any(math.isinf(v) or math.isnan(v) for v in vals.values()) or vals["input_i"] < -60:
        return None                    # clip praticamente muta: niente normalizzazione forzata
    return {k: f"{v:.2f}" for k, v in vals.items()}


def render_events(item: Dict[str, Any]) -> Dict[str, Any]:
    """Dove mettere zoom punch e micro-shake: sul momento clou e sugli impatti rilevati."""
    trim, m = item["trim"], item["metrics"]
    t_in = trim["t_in"]
    contacts = [t_in + t for t in m.get("contact_times", [])]
    punch: List[Tuple[float, float]] = []
    if trim["climax_kind"] == "contatto":
        punch.append((trim["climax_t"], 0.30))
    else:
        punch.append((trim["climax_t"], 0.70))
    for t in sorted(contacts + [t_in + x for x in m.get("overtake_times", [])]):
        if all(abs(t - p[0]) >= 4.0 for p in punch) and len(punch) < 3:
            punch.append((t, 0.30 if t in contacts else 0.70))
    return {"punch": punch, "shake": contacts[:2]}


def render_vertical(ctx: Ctx, item: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    info = ctx.info
    fps = info.fps
    segs: List[List[int]] = item["trim"]["segments"]
    an = item["analysis"]
    webcam_ok = bool(ctx.webcam) and item["metrics"]["webcam_ratio"] >= 0.5
    events = render_events(item)
    plans = [plan_segment(ctx, an, a, b, events, webcam_ok) for a, b in segs]
    total_frames = sum(p.nf for p in plans)
    total = total_frames / fps
    durs = [p.nf / fps for p in plans]
    log_path = ctx.log_path("render")

    contrast, gamma, saturation = grade_params(item["metrics"]["luma"])
    fade = 0.15
    # La conversione BGR->YUV avviene per prima e con matrice BT.709 esplicita: senza, FFmpeg usa la BT.601
    # e i colori dell'HD virano (rossi e verdi) rispetto alla sorgente.
    vchain = ("[0:v]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
              f"eq=contrast={contrast}:gamma={gamma}:saturation={saturation},"
              "unsharp=5:5:0.45:5:5:0.0,"
              f"fade=t=out:st={max(0.0, total - fade):.3f}:d={fade}[v]")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-framerate", info.fps_str, "-i", "-"]
    loud_mode = "assente"
    if info.has_audio:
        measured = measure_loudness(ctx, segs, log_path)
        if measured:
            tail = (f"loudnorm={LOUD_TARGET}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
                    f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
                    f"offset={measured['target_offset']}:linear=true:print_format=summary")
            loud_mode = "due passate"
        else:
            tail = f"loudnorm={LOUD_TARGET}"
            loud_mode = "passata singola"
        tail += (f",aresample=48000,apad=whole_dur={total:.6f},atrim=0:{total:.6f},"
                 f"afade=t=out:st={max(0.0, total - fade):.3f}:d={fade}")
        cmd += _audio_inputs(info, segs)
        graph = vchain + ";" + _audio_graph(durs, 1, tail)
    else:
        cmd += ["-f", "lavfi", "-t", f"{total:.6f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph = vchain + ";[1:a]anull[a]"
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]"]
    cmd += video_encoder_args(ctx.encoder, "social", fps, ctx.fast_encode) + COLOR_TAGS
    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
            "-t", f"{total:.6f}", str(out_path)]

    comp = Compositor(ctx)
    kind = "flash"                    # azione + replay: bagliore morbido; "whip" per altri accostamenti
    tn = max(2, int(round(0.16 * fps)))
    writer = FFmpegWriter(cmd, log_path, f"render 9:16 di {out_path.name}")
    padded = 0
    try:
        for si, plan in enumerate(plans):
            last: Optional[np.ndarray] = None
            with FFmpegReader(info, plan.n_in, plan.nf, info.width, info.height, log_path) as reader:
                for j in range(plan.nf):
                    src = reader.read()
                    if src is None:
                        if last is None:
                            raise RaceClipperError("la sorgente non restituisce frame per il segmento richiesto")
                        out, padded = last, padded + 1     # fine sorgente: ripete l'ultimo frame, durata invariata
                    else:
                        out = comp.frame(src, plan, j)
                        if len(plans) > 1 and comp.effects:
                            if si < len(plans) - 1 and j >= plan.nf - tn:
                                out = apply_transition(out, kind, (j - (plan.nf - tn) + 1) / tn, True)
                            if si > 0 and j < tn:
                                out = apply_transition(out, kind, 1.0 - j / tn, False)
                        last = out
                    writer.write(out)
        writer.close()
    except BaseException:
        writer.abort()
        raise

    # Controllo finale del file: stream presenti, formato giusto, durata audio/video coerente
    probe = ffprobe_info(out_path)
    if (probe.width, probe.height) != (OUT_W, OUT_H) or not probe.has_audio:
        raise RaceClipperError(f"{out_path.name}: file prodotto ma non conforme ({probe.width}x{probe.height}, "
                               f"audio={probe.has_audio})")
    if abs(probe.duration - total) > max(0.12, 3.0 / fps):
        raise RaceClipperError(f"{out_path.name}: durata {probe.duration:.2f} s diversa da quella attesa ({total:.2f} s)")
    rep = layout_report(plans)
    rep.update({"frames": total_frames, "frame_ripetuti_fine_sorgente": padded, "loudness": loud_mode,
                "grading": {"contrast": contrast, "gamma": gamma, "saturation": saturation},
                "zoom_punch": len(events["punch"]) if comp.effects else 0,
                "shake_impatto": len(events["shake"]) if comp.effects else 0,
                "transizione": kind if len(plans) > 1 and comp.effects else "nessuna"})
    return rep


def render_master(ctx: Ctx, item: Dict[str, Any], out_path: Path) -> None:
    """Master 16:9 tagliato al fotogramma sugli stessi segmenti del 9:16: nessun effetto, audio originale."""
    info = ctx.info
    fps = info.fps
    segs: List[List[int]] = item["trim"]["segments"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    for a, b in segs:
        cmd += ["-ss", f"{max(0.0, (a - 0.4) / fps):.6f}", "-t", f"{(b - a) / fps + 0.30:.6f}", "-i", str(info.path)]
    parts: List[str] = []
    for i, (a, b) in enumerate(segs):
        d = (b - a) / fps
        off = a / fps - max(0.0, (a - 0.4) / fps)      # il seek parte 0,4 frame prima: l'audio si riallinea qui
        parts.append(f"[{i}:v:0]fps={info.fps_str},trim=end_frame={b - a},setpts=PTS-STARTPTS[v{i}]")
        if info.has_audio:
            parts.append(f"[{i}:a:0]aresample=48000,atrim={off:.6f}:{off + d:.6f},asetpts=PTS-STARTPTS[a{i}]")
    n = len(segs)
    if n > 1:
        if info.has_audio:
            parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]")
        else:
            parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
        vmap, amap = "[v]", "[a]"
    else:
        vmap, amap = "[v0]", "[a0]"
    cmd += ["-filter_complex", ";".join(parts), "-map", vmap]
    if info.has_audio:
        cmd += ["-map", amap, "-c:a", "aac", "-b:a", "320k", "-ar", "48000"]
    cmd += video_encoder_args(ctx.encoder, "master", fps, ctx.fast_encode)
    cmd += ["-movflags", "+faststart", str(out_path)]
    run_ffmpeg(cmd, ctx.log_path("render"), f"master 16:9 di {out_path.name}", timeout=3600)
    probe = ffprobe_info(out_path)
    expected = sum(b - a for a, b in segs) / fps
    if abs(probe.duration - expected) > max(0.15, 3.0 / fps):
        raise RaceClipperError(f"{out_path.name}: durata master {probe.duration:.2f} s, attesa {expected:.2f} s")


# ---------------------------------------------------------------------------------------------------------
# Export: cartelle, schede SEO, report
# ---------------------------------------------------------------------------------------------------------
DIR_VERTICAL = "01_Verticale_9x16"
DIR_MASTER = "02_Master_16x9"
DIR_SOCIAL = "03_Titoli_e_Caption_SEO"
FILE_ALL_SOCIAL = "PRONTI_PER_I_SOCIAL.txt"
FILE_REPORT = "classifica_viralita.json"


def prepare_output_dirs(out_dir: Path) -> Dict[str, Path]:
    """Le clip di una run precedente non vengono cancellate (il v10 lo faceva): finiscono in _archivio/<data>."""
    dirs = {k: out_dir / k for k in (DIR_VERTICAL, DIR_MASTER, DIR_SOCIAL)}
    old = [d for d in dirs.values() if d.exists() and any(d.iterdir())]
    old_files = [out_dir / f for f in (FILE_ALL_SOCIAL, FILE_REPORT) if (out_dir / f).exists()]
    if old or old_files:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = out_dir / "_archivio" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        for d in old:
            shutil.move(str(d), str(archive / d.name))
        for f in old_files:
            shutil.move(str(f), str(archive / f.name))
        log(f" -> Risultati precedenti spostati in {archive}")
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def social_card(rank: int, item: Dict[str, Any], names: Dict[str, str]) -> str:
    c = item["copy"]
    lines = [
        f"RANK {rank:02d} | PUNTEGGIO {item['score']}/100 | {viral_tier_label(item['score'])}",
        f"File 9:16        : {names['vertical']}",
        f"File master 16:9 : {names['master']}",
        f"Momento nella live: {format_time(item['trim']['t_in'])} (durata {item['trim']['dur']:.0f} s)",
        "",
        "TITOLO",
        c["title"],
        "",
        "HOOK PRIMI 3 SECONDI (testo a schermo)",
        c["hook_3s"],
        "",
        "CAPTION",
        c["caption"],
        "",
        "PAROLE CHIAVE SEO DA INSERIRE NEL POST / TAGS",
        ", ".join(c["keywords"]),
        "",
    ]
    if c.get("warnings"):
        lines += ["DA RIVEDERE A MANO", *[f"- {w}" for w in c["warnings"]], ""]
    return "\n".join(lines).replace("#", "")       # nessun '#' nei file esportati, in nessun punto


def export_all(ctx: Ctx, finalists: List[Dict[str, Any]], rejected: List[Dict[str, Any]], dirs: Dict[str, Path],
               stats: Dict[str, Any]) -> None:
    race = ctx.race
    title_bits = " @ ".join(x for x in (race.get("championship"), race.get("circuit")) if x) or "gara"
    all_lines = ["=" * 88, f"TOP {len(finalists)} HIGHLIGHTS ({title_bits})", "=" * 88, ""]
    report: List[Dict[str, Any]] = []
    for rank, it in enumerate(finalists, start=1):
        names = it["files"]
        card = social_card(rank, it, names)
        with open(dirs[DIR_SOCIAL] / names["social"], "w", encoding="utf-8") as f:
            f.write(card)
        all_lines += [card, "-" * 88, ""]
        vis = it.get("vision") or {}
        report.append({
            "rank": rank, "viral_score": it["score"], "recommendation": viral_tier_label(it["score"]),
            "score_breakdown_100": it["breakdown"], "audit": it.get("audit", {}), "flags": it.get("flags", []),
            "start_time": format_time(it["trim"]["t_in"]), "start_seconds": it["trim"]["t_in"],
            "duration_seconds": it["trim"]["dur"],
            "trimming": {k: it["trim"].get(k) for k in ("climax_kind", "method", "snap_in", "snap_out", "replay",
                                                        "story_window", "truncated", "window_grown")},
            "frammenti_riuniti": it["cand"].get("merged", 1),
            "innesco_s": round(it["trim"]["trigger_t"] - it["trim"]["t_in"], 2),
            "risoluzione_s": round(it["trim"]["resolution_t"] - it["trim"]["t_in"], 2),
            "visual_telemetry": {k: v for k, v in it["metrics"].items() if k != "luma"},
            "what_vision_ai_saw": vis.get("description", ""), "vision_action": vis.get("action", ""),
            "facts": (it.get("judge") or {}).get("facts", ""),
            "regia": it.get("render", {}),
            "social_kit": {"title": it["copy"]["title"], "hook_first_3s": it["copy"]["hook_3s"],
                           "caption": it["copy"]["caption"], "seo_keywords": it["copy"]["keywords"],
                           "warnings": it["copy"].get("warnings", [])},
            "file_9x16": names["vertical"], "file_16x9_master": names["master"], "file_social_txt": names["social"],
        })
    with open(ctx.out_dir / FILE_ALL_SOCIAL, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    blob = {
        "versione": VERSION, "sorgente": str(ctx.info.path.name), "contesto_gara": race,
        "modelli": ctx.models, "statistiche": stats, "clip": report,
        "candidate_scartate": [{
            "momento": format_time(r["trim"]["t_in"]) if r.get("trim") else format_time(r["cand"]["start"]),
            "motivo": r.get("reject_reason") or "fuori dalla top", "punteggio": r.get("score"),
            "tipo": r["cand"].get("action_type"),
        } for r in rejected],
    }
    with open(ctx.out_dir / FILE_REPORT, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)


def final_checks(ctx: Ctx, dirs: Dict[str, Path], top_k: int) -> List[str]:
    """Ultimo controllo sul disco: mai piu' di top_k clip per cartella, nessun '#' nei testi."""
    problems: List[str] = []
    for key in (DIR_VERTICAL, DIR_MASTER, DIR_SOCIAL):
        n = len([p for p in dirs[key].iterdir() if p.is_file()])
        if n > top_k:
            problems.append(f"{key}: {n} file, attesi al massimo {top_k}")
    for p in list(dirs[DIR_SOCIAL].glob("*.txt")) + [ctx.out_dir / FILE_ALL_SOCIAL]:
        if p.exists() and "#" in p.read_text(encoding="utf-8"):
            problems.append(f"{p.name}: contiene il simbolo #")
    return problems


# ---------------------------------------------------------------------------------------------------------
# COLLAUDO (--selftest): diretta sintetica + rilevatore sintetico + Ollama simulato
# Verifica in un paio di minuti FFmpeg, encoder, pipe, sincronia audio/video, regia, validazione JSON, export.
# Non richiede GPU, Ollama, Whisper ne' un video reale. Non valuta la qualita' dei modelli AI.
# ---------------------------------------------------------------------------------------------------------
ST_W, ST_H, ST_FPS = 1280, 720, 30
ST_SCENES = {"classifica": (0.0, 24.0), "duello": (24.0, 60.0), "solo": (60.0, 86.0),
             "larghi": (86.0, 120.0), "interviste": (120.0, 136.0), "lungo": (136.0, 216.0)}
ST_TOTAL = 216.0
ST_YELLOW, ST_MAGENTA, ST_GREEN = (0, 220, 255), (255, 0, 255), (0, 220, 0)       # BGR
ST_WEBCAM = (960, 30, 1240, 190)
ST_HUD = (20, 60, 200, 300)
ST_CAR_W, ST_CAR_H = 140, 70


def _st_car_positions(t: float) -> List[Tuple[Tuple[int, int, int], float, float]]:
    """Posizione (colore, cx, cy) delle auto sintetiche al tempo t."""
    def lerp(a: float, b: float, u: float) -> float:
        return a + (b - a) * clamp(u, 0.0, 1.0)

    s0, s1 = ST_SCENES["duello"]
    if s0 <= t < s1:
        u = t - s0
        shift = 80.0 if u >= 10.0 else 0.0                       # stacco di camera a +10 s
        y_x = 560.0 + shift
        if u < 16.0:
            m_x = lerp(300.0, 400.0, u / 16.0) + shift           # insegue da dietro
        elif u < 20.0:
            m_x = lerp(400.0, 760.0, (u - 16.0) / 4.0) + shift   # sorpasso
        elif u < 24.0:
            m_x = 760.0 + shift
        else:
            m_x = lerp(760.0, 1500.0, (u - 24.0) / 6.0) + shift  # allunga ed esce
        cars = [(ST_YELLOW, y_x, 430.0)]
        if m_x < ST_W + ST_CAR_W:
            cars.append((ST_MAGENTA, m_x, 520.0))
        return cars
    s0, s1 = ST_SCENES["solo"]
    if s0 <= t < s1:
        return [(ST_YELLOW, 640.0 + 30.0 * math.sin((t - s0) * 0.8), 440.0)]
    s0, s1 = ST_SCENES["lungo"]
    if s0 <= t < s1:
        # Un'unica azione lunga: Bianchi (magenta) si avvicina, passa, sul rettilineo le auto si allungano per
        # 8 s, poi Rossi (gialla) incrocia e ripassa, restano affiancati 10 s e infine la gialla allunga.
        u = t - s0
        m_x = 150.0 if u < 10.0 else lerp(150.0, 470.0, (u - 10.0) / 8.0) if u < 18.0 else \
            lerp(470.0, 820.0, (u - 18.0) / 4.0) if u < 22.0 else lerp(820.0, 900.0, (u - 22.0) / 8.0)
        y_x = 640.0 if u < 22.0 else lerp(640.0, 560.0, (u - 22.0) / 4.0) if u < 30.0 else \
            lerp(560.0, 980.0, (u - 30.0) / 4.0) if u < 44.0 else lerp(980.0, 1500.0, (u - 44.0) / 8.0)
        cars = [(ST_MAGENTA, m_x, 520.0)]
        if y_x < ST_W + ST_CAR_W:
            cars.append((ST_YELLOW, y_x, 430.0))
        return cars
    s0, s1 = ST_SCENES["larghi"]
    if s0 <= t < s1:
        u = t - s0
        if u < 16.0:
            yx, mx = 150.0, 1090.0
        elif u < 24.0:
            yx, mx = lerp(150.0, 600.0, (u - 16.0) / 8.0), lerp(1090.0, 730.0, (u - 16.0) / 8.0)
        elif u < 25.0:
            yx, mx = 600.0, 730.0                                # contatto
        else:
            yx, mx = lerp(600.0, 420.0, (u - 25.0) / 9.0), lerp(730.0, 900.0, (u - 25.0) / 9.0)
        return [(ST_YELLOW, yx, 450.0), (ST_MAGENTA, mx, 450.0)]
    return []


def _st_frame(t: float, textures: Dict[str, np.ndarray]) -> np.ndarray:
    scene = next((k for k, (a, b) in ST_SCENES.items() if a <= t < b), "interviste")
    if scene == "classifica":
        img = np.full((ST_H, ST_W, 3), (60, 30, 20), np.uint8)
        cv2.putText(img, "CLASSIFICA", (380, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 5, cv2.LINE_AA)
        for i, name in enumerate(["1  ROSSI", "2  BIANCHI", "3  VERDI"]):
            cv2.putText(img, name, (420, 280 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (230, 230, 230), 3, cv2.LINE_AA)
        return img
    if scene == "interviste":
        img = np.full((ST_H, ST_W, 3), (40, 40, 70), np.uint8)
        cv2.putText(img, "INTERVISTE", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.ellipse(img, (640, 300), (110, 130), 0, 0, 360, ST_GREEN, -1)
        cv2.rectangle(img, (460, 420), (820, 720), ST_GREEN, -1)
        return img
    key = "b" if (scene == "duello" and t - ST_SCENES["duello"][0] >= 10.0) else {"duello": "a", "solo": "c", "larghi": "d", "lungo": "b"}[scene]
    tex = textures[key]
    off = int(t * 450.0) % (tex.shape[1] - ST_W)
    img = tex[:, off:off + ST_W].copy()
    for color, cx, cy in _st_car_positions(t):
        x1, y1 = int(cx - ST_CAR_W / 2), int(cy - ST_CAR_H / 2)
        cv2.rectangle(img, (x1, y1), (x1 + ST_CAR_W, y1 + ST_CAR_H), color, -1)
        cv2.rectangle(img, (x1 + 78, y1 + 12), (x1 + 112, y1 + ST_CAR_H - 12), (40, 40, 40), -1)
    hx1, hy1, hx2, hy2 = ST_HUD
    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (25, 25, 25), -1)
    for i, name in enumerate(["P1 ROSSI", "P2 BIANCHI", "P3 VERDI", "P4 NERI"]):
        cv2.putText(img, name, (hx1 + 10, hy1 + 40 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    wx1, wy1, wx2, wy2 = ST_WEBCAM
    cv2.rectangle(img, (wx1, wy1), (wx2, wy2), (95, 95, 95), -1)
    jx, jy = int(3 * math.sin(t * 5.0)), int(2 * math.cos(t * 3.7))
    cv2.ellipse(img, (wx1 + 140 + jx, wy1 + 62 + jy), (30, 36), 0, 0, 360, ST_GREEN, -1)
    cv2.rectangle(img, (wx1 + 85 + jx, wy1 + 100 + jy), (wx1 + 195 + jx, wy2 - 4), ST_GREEN, -1)
    cv2.rectangle(img, (wx1, wy1), (wx2, wy2), (255, 255, 255), 3)
    return img


def _st_textures() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    out: Dict[str, np.ndarray] = {}
    for key, (base, tint) in {"a": (95, (0, 0, 0)), "b": (140, (25, 5, -15)), "c": (70, (-10, 10, 20)),
                              "d": (120, (10, -15, 5))}.items():
        w = ST_W * 3
        noise = rng.integers(-38, 38, size=(ST_H // 4, w // 4), dtype=np.int16)
        noise = cv2.resize(noise.astype(np.float32), (w, ST_H), interpolation=cv2.INTER_CUBIC)
        tex = np.clip(base + noise[:, :, None] + np.array(tint, np.float32)[None, None, :], 0, 255).astype(np.uint8)
        for x in range(0, w, 220):
            cv2.rectangle(tex, (x, 352), (x + 110, 362), (235, 235, 235), -1)
            cv2.rectangle(tex, (x + 60, 600), (x + 170, 610), (235, 235, 235), -1)
        tex[:, w - ST_W:] = tex[:, :ST_W]              # texture ciclica
        out[key] = tex
    return out


def _st_audio(path: Path) -> None:
    import wave
    sr = 48000
    n = int(ST_TOTAL * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(5)
    engine = 0.04 * (2.0 * ((t * 110.0) % 1.0) - 1.0)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec[(freqs < 300) | (freqs > 3000)] = 0.0
    voice = np.fft.irfft(spec, n)
    voice /= np.abs(voice).max() + 1e-9
    level = np.full(n, 0.10)
    for a, b, lv in ((37.0, 47.0, 0.75), (62.0, 80.0, 0.70), (106.0, 114.0, 0.80),
                     (153.0, 159.0, 0.75), (164.5, 171.5, 0.80), (180.0, 185.0, 0.55)):
        level[(t >= a) & (t < b)] = lv
    k = int(0.4 * sr)
    level = np.convolve(level, np.ones(k) / k, mode="same")
    voice *= level * (0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 2.2 * t)))
    sig = engine + voice
    c0 = int(110.0 * sr)
    sig[c0:c0 + int(0.04 * sr)] += 0.9 * rng.standard_normal(int(0.04 * sr))      # botto del contatto
    pcm = (np.clip(sig, -1.0, 1.0) * 32000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def selftest_make_video(path: Path, log_path: Path) -> None:
    wav = path.with_suffix(".wav")
    _st_audio(wav)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{ST_W}x{ST_H}", "-framerate", str(ST_FPS), "-i", "-", "-i", str(wav),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-g", str(ST_FPS), "-pix_fmt", "yuv420p"] + COLOR_TAGS + \
          ["-c:a", "aac", "-b:a", "160k", "-shortest", str(path)]
    textures = _st_textures()
    writer = FFmpegWriter(cmd, log_path, "generazione della diretta sintetica")
    try:
        for i in range(int(ST_TOTAL * ST_FPS)):
            writer.write(_st_frame(i / ST_FPS, textures))
        writer.close()
    except BaseException:
        writer.abort()
        raise
    wav.unlink(missing_ok=True)


def selftest_transcript() -> List[Dict[str, Any]]:
    lines = [
        (2.0, 12.0, "ecco la classifica dopo le qualifiche primo rossi secondo bianchi terzo verdi"),
        (26.0, 32.0, "siamo a oulton park per il campionato btcc con rossi davanti a bianchi"),
        (36.0, 39.5, "occhio qui che bianchi ci prova all'interno"),
        (40.0, 46.0, "si butta dentro in staccata sorpasso completato bianchi passa rossi"),
        (48.5, 53.0, "e adesso bianchi allunga su rossi"),
        (62.0, 80.0, "che spettacolo ragazzi una atmosfera bellissima oggi a oulton park grazie a tutti quelli "
                     "che ci seguono da casa siamo tantissimi collegati per questa gara del btcc"),
        (88.0, 100.0, "rossi e bianchi sono appaiati sul rettilineo larghissimi uno a sinistra e uno a destra"),
        (106.0, 109.5, "attenzione si stanno avvicinando"),
        (110.2, 114.0, "contatto contatto si toccano rossi e bianchi"),
        (115.0, 119.0, "ma restano in pista tutti e due"),
        (122.0, 134.0, "passiamo alle interviste grazie mille per l'intervista ci tenevo a ringraziare tutti"),
        (146.0, 152.0, "bianchi adesso si avvicina a rossi e gli mette pressione"),
        (153.5, 158.0, "occhio che bianchi ci prova all'esterno sorpasso riuscito"),
        (165.0, 171.0, "ma rossi non ci sta incrocia la traiettoria e si riprende la posizione"),
        (180.0, 185.0, "che duello fra questi due rossi adesso allunga"),
    ]
    segs = []
    for a, b, text in lines:
        toks = text.split()
        step = (b - a) / len(toks)
        words = [{"w": w, "s": round(a + i * step, 2), "e": round(a + (i + 0.8) * step, 2)} for i, w in enumerate(toks)]
        segs.append({"start": a, "end": b, "text": text, "words": words})
    return segs


def synthetic_dets(frame: np.ndarray, persons: bool) -> List[Det]:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = [((22, 140, 140), (38, 255, 255), 2), ((140, 140, 140), (162, 255, 255), 2)]
    if persons:
        ranges.append(((50, 140, 110), (70, 255, 255), PERSON_CLASS))
    dets: List[Det] = []
    for lo, hi, cls in ranges:
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area >= 0.0008 * w * h:
                dets.append((cls, 0.9, x / w, y / h, (x + bw) / w, (y + bh) / h))
    return dets


class SyntheticDetector:
    name = "rilevatore_sintetico"
    batch = 8

    def detect(self, frames: List[np.ndarray]) -> List[List[Det]]:
        return [synthetic_dets(f, persons=False) for f in frames]

    def close(self) -> None:
        pass


class SyntheticScreener:
    device = "CPU (sintetico)"
    device_name = "collaudo"
    model_name = "screener_sintetico"

    def infer(self, frame: np.ndarray) -> List[Det]:
        return synthetic_dets(frame, persons=True)


class FakeOllama:
    """Server HTTP minimo con le API di Ollama usate dal tool. Risponde in modo deterministico e, apposta,
    sbaglia alcune prime risposte per far lavorare validazione, coercizione dei tipi e retry."""

    MODELS = ["finto-regista:1b", "finto-occhio:1b", "finto-copy:1b"]

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self
        self.calls: Dict[str, int] = {}
        self.bad_sent: set = set()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, obj: Any) -> None:
                body = json.dumps(obj).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.startswith("/api/tags"):
                    self._send({"models": [{"name": m} for m in outer.MODELS]})
                else:
                    self._send({"models": []})

            def do_POST(self) -> None:
                size = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(size) or b"{}")
                if not self.path.startswith("/api/chat"):
                    self._send({"done": True})
                    return
                content = outer.answer(req)
                self._send({"message": {"role": "assistant", "content": content}, "done": True,
                            "done_reason": "stop", "prompt_eval_count": 200, "eval_count": 80})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()

    def _once(self, key: str) -> bool:
        if key in self.bad_sent:
            return False
        self.bad_sent.add(key)
        return True

    def answer(self, req: Dict[str, Any]) -> str:
        fmt = req.get("format") if isinstance(req.get("format"), dict) else {}
        props = set((fmt.get("properties") or {}).keys())
        msgs = req.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), {})
        text = user.get("content", "")
        retry = len(msgs) > 2
        kind = ("continuity" if "action_start" in props else
                "context" if "circuit" in props else "phases" if "blocks" in props else "scout" if "clips" in props
                else "vision" if "scene_type" in props else "judge" if "story_score" in props
                else "copy" if "hook_3s" in props else "audit" if "reviews" in props else "?")
        self.calls[kind] = self.calls.get(kind, 0) + 1
        if kind == "continuity":
            est = re.search(r"FINESTRA STIMATA: da (\d+(?:\.\d+)?) a (\d+(?:\.\d+)?)", text)
            e0, e1 = (float(est.group(1)), float(est.group(2))) if est else (0.0, 0.0)
            rows = [(float(m.group(1)), float(m.group(2)), m.group(3))
                    for m in re.finditer(r"\[(\d+\.\d)-(\d+\.\d)\] (.*)", text)]
            inside = " ".join(body for a, b, body in rows if b >= e0 and a <= e1)

            def edge(word: str, which: int, default: float) -> float:
                hit = [r for r in rows if word in r[2]]
                return hit[0][which] if hit else default

            if "sorpasso" in inside:           # la storia parte dalla preparazione e finisce quando uno dei due allunga
                a0, b0 = edge("si avvicina", 0, edge("ci prova", 0, e0)), edge("allunga", 1, e1)
                story = "Bianchi studia Rossi, lo attacca all'interno in staccata, lo passa e poi allunga."
            elif "contatto" in inside:
                a0, b0 = edge("attenzione", 0, e0), edge("restano in pista", 1, e1)
                story = "Rossi e Bianchi convergono sul rettilineo, si toccano e restano entrambi in pista."
            else:
                a0, b0, story = e0, e1, "Il telecronista commenta, nessuno sviluppo ulteriore dell'azione."
            return json.dumps({"story": story, "action_start": a0, "action_end": f"{b0:.1f}s", "resolved": "si"})
        if kind == "context":
            return json.dumps({"circuit": "Oulton Park", "championship": "BTCC", "category": "", "cars": [],
                               "drivers": ["Rossi", "Bianchi", "Verstappen"]})     # l'ultimo non e' in telecronaca
        if kind == "phases":
            blocks = []
            for m in re.finditer(r"BLOCCO (\d+) \[[^\]]*\]: (.*)", text):
                body = m.group(2)
                phase = "INTERVISTE" if ("intervist" in body and "sorpasso" not in body) else \
                    "PAUSA_CLASSIFICA" if ("classifica" in body and "sorpasso" not in body) else "GARA"
                blocks.append({"id": int(m.group(1)), "phase": phase})
            return json.dumps({"blocks": blocks})
        if kind == "scout":
            clips = []
            for m in re.finditer(r"\[(\d+\.\d)-(\d+\.\d)[^\]]*\] (.*)", text):
                a, b, body = float(m.group(1)), float(m.group(2)), m.group(3)
                if "sorpasso" in body:
                    clips.append({"summary": "Bianchi supera Rossi all'interno in staccata", "start": a - 4, "end": b + 3,
                                  "action_type": "SORPASSO", "drivers": ["Bianchi", "Rossi"], "confidence": 88})
                elif "contatto" in body:
                    clips.append({"summary": "Contatto fra Rossi e Bianchi in rettilineo", "start": a - 4, "end": b + 3,
                                  "action_type": "CONTATTO", "drivers": ["Rossi", "Bianchi"], "confidence": 92})
                elif "spettacolo" in body:
                    clips.append({"summary": "Momento di grande entusiasmo in cabina", "start": a, "end": b,
                                  "action_type": "DUELLO", "drivers": [], "confidence": 70})
            if clips and not retry and self._once("scout"):
                bad = [dict(c) for c in clips]
                bad[0]["confidence"] = "alta"          # stringa non numerica: deve scattare il retry
                bad[0].pop("end")                      # campo obbligatorio mancante
                return "```json\n" + json.dumps({"clips": bad}) + "\n```"
            return json.dumps({"clips": clips[:3]})
        if kind == "vision":
            best_n, close = 0, 0
            for b64 in user.get("images", []):
                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                cars = [d for d in synthetic_dets(img, persons=False)]
                if len(cars) > best_n:
                    best_n = len(cars)
                if len(cars) >= 2:
                    xs = sorted((d[2] + d[4]) / 2 for d in cars)
                    if xs[1] - xs[0] < 0.45:
                        close = 2
            if best_n >= 2 and close:
                res = {"description": "Due auto, una gialla e una magenta, molto vicine sulla stessa porzione di pista.",
                       "scene_type": "AZIONE_IN_PISTA", "cars_visible_max": "2", "cars_close_max": 2,
                       "action": "duello ravvicinato", "action_visible": "si", "visual_score": "17/20"}
            elif best_n >= 2:
                res = {"description": "Due auto distanti fra loro ai lati opposti della pista.",
                       "scene_type": "AZIONE_IN_PISTA", "cars_visible_max": 2, "cars_close_max": 0,
                       "action": "DUELLO_RAVVICINATO", "action_visible": True, "visual_score": 11}
            elif best_n == 1:
                res = {"description": "Una sola auto gialla al centro della pista, nessun avversario vicino.",
                       "scene_type": "AZIONE_IN_PISTA", "cars_visible_max": 1, "cars_close_max": 0,
                       "action": "AUTO_SOLITARIA", "action_visible": False, "visual_score": 4}
            else:
                res = {"description": "Schermata con testo, nessuna auto in pista.",
                       "scene_type": "GRAFICA_O_CLASSIFICA", "cars_visible_max": 0, "cars_close_max": 0,
                       "action": "NESSUNA_AZIONE", "action_visible": False, "visual_score": 0}
            return json.dumps(res)
        if kind == "judge":
            clip_text = (re.search(r"TELECRONACA DELLA CLIP[^\"]*\"(.*?)\"\n", text, flags=re.S) or [None, ""])[1]
            contact = int((re.search(r"contatti rilevati: (\d+)", text) or [0, 0])[1])
            passes = int((re.search(r"sorpassi misurati: (\d+)", text) or [0, 0])[1])
            if "riprende la posizione" in clip_text:
                res = {"facts": "Bianchi passa Rossi all'esterno, Rossi incrocia la traiettoria e si riprende la posizione.",
                       "phase": "GARA", "is_live_action": True, "action_type": "DUELLO", "drivers": ["Bianchi", "Rossi"],
                       "reason": "sorpasso e controsorpasso nella stessa azione", "story_score": 30}
            elif "contatto" in clip_text or contact:
                res = {"facts": "Rossi e Bianchi arrivano larghi sul rettilineo, si avvicinano e si toccano, restando in pista.",
                       "phase": "GARA", "is_live_action": True, "action_type": "CONTATTO", "drivers": ["Rossi", "Bianchi"],
                       "reason": "contatto confermato da sensori e telecronaca", "story_score": "31"}
            elif "sorpasso" in clip_text or passes:
                res = {"facts": "Bianchi si butta all'interno in staccata e completa il sorpasso su Rossi.",
                       "phase": "GARA", "is_live_action": True, "action_type": "SORPASSO", "drivers": ["Bianchi", "Rossi"],
                       "reason": "sorpasso ruota a ruota ben visibile", "story_score": "27/35"}
            else:
                res = {"facts": "Il telecronista commenta l'atmosfera, in pista non succede nulla di rilevante.",
                       "phase": "GARA", "is_live_action": True, "action_type": "NESSUNA", "drivers": [],
                       "reason": "nessuna azione concreta", "story_score": 8}
            return json.dumps(res)
        if kind == "copy":
            contact = "Tipo di azione: CONTATTO" in text
            if "RESPINTA" not in text and "NOTE DEL REVISORE" not in text and self._once("copy"):
                return json.dumps({"title": "Sorpasso al limite! Chi vince? #motorsport",
                                   "hook_3s": "Guarda qui #shorts", "caption": "Che gara! #simracing #btcc " * 8,
                                   "keywords": ["#motorsport", "#shorts", "sorpasso", "btcc", "gara", "auto", "pista", "video"]})
            suffix = " dopo la revisione" if "NOTE DEL REVISORE" in text else ""
            if "Tipo di azione: DUELLO" in text:
                title = "Sorpasso e controsorpasso fra Bianchi e Rossi a Oulton Park" + suffix
                caption = ("Duello completo a Oulton Park nel BTCC: Bianchi si avvicina, mette pressione e passa Rossi "
                           "all'esterno. Sul rettilineo successivo Rossi resta in scia, incrocia la traiettoria e si "
                           "riprende la posizione, poi le due auto restano ruota a ruota per diverse curve prima "
                           "dell'allungo decisivo. Voi avreste chiuso la porta all'interno o difeso l'uscita?")
                kws = ["sorpasso e controsorpasso", "oulton park", "btcc", "duello ruota a ruota", "incrocio di traiettoria",
                       "sim racing", "sorpasso all'esterno", "highlights btcc", "gara turismo", "bagarre"]
            elif contact:
                title = "Contatto in pieno rettilineo fra Rossi e Bianchi a Oulton Park" + suffix
                caption = ("A Oulton Park, nel BTCC, Rossi e Bianchi percorrono il rettilineo ai lati opposti della pista, "
                           "poi convergono e arriva il contatto fianco contro fianco. Le due auto si toccano ma restano "
                           "in pista e il duello ruota a ruota prosegue. Un episodio che racconta bene quanto conti la "
                           "gestione dello spazio in rettilineo. Secondo voi chi dei due doveva lasciare piu' margine?")
                kws = ["contatto in gara", "oulton park", "btcc", "duello ruota a ruota", "rettilineo", "sim racing",
                       "contatto fianco a fianco", "highlights btcc", "gara turismo", "incidente di gara"]
            else:
                title = "Bianchi si infila all'interno in staccata e passa Rossi a Oulton Park" + suffix
                caption = ("Sorpasso in staccata a Oulton Park nel BTCC: Bianchi prende la scia di Rossi, ritarda la "
                           "frenata e si infila all'interno. Le due auto restano ruota a ruota per tutta la curva, poi "
                           "Bianchi completa la manovra e allunga. Una staccata pulita, senza contatto, costruita "
                           "sull'uscita dalla curva precedente. Voi avreste difeso la traiettoria interna?")
                kws = ["sorpasso in staccata", "oulton park", "btcc", "duello ruota a ruota", "sorpasso all'interno",
                       "sim racing", "staccata", "highlights btcc", "gara turismo", "tecnica di guida"]
            return json.dumps({"title": title, "hook_3s": "Guarda la staccata", "caption": caption, "keywords": kws})
        if kind == "audit":
            rows = re.findall(r"\"id\": (\d+),\s*\"punteggio\": (\d+)", text)
            reviews = []
            for i, (cid, score) in enumerate(rows):
                rev = {"id": int(cid), "score_comment": "coerente", "final_score": int(score) - 2,
                       "title_verdict": "OK", "title_note": "", "caption_verdict": "OK", "caption_note": ""}
                if i == len(rows) - 1:
                    rev.update({"title_verdict": "DEBOLE", "title_note": "rendi piu' specifico il punto della pista"})
                reviews.append(rev)
            return json.dumps({"reviews": reviews})
        return "{}"


def selftest_report(ctx: Ctx, finalists: List[Dict[str, Any]], items: List[Dict[str, Any]], fake: FakeOllama) -> bool:
    """Controlli del collaudo: ogni riga e' un comportamento atteso sulla diretta sintetica."""
    checks: List[Tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    types = [(it.get("judge") or {}).get("action_type") for it in finalists]
    add("Esportate esattamente le 3 azioni vere (sorpasso, contatto, duello lungo)",
        sorted(types) == ["CONTATTO", "DUELLO", "SORPASSO"], str(types))
    solo = [it for it in items if 60.0 <= it["cand"]["start"] <= 86.0 or 60.0 <= it["trim"]["t_in"] <= 84.0]
    add("Auto da sola scartata nonostante il telecronista esaltato",
        bool(solo) and all(it["verdict"] == "SCARTATA" for it in solo),
        "; ".join(it.get("reject_reason", "") for it in solo))
    add("Nome non presente in telecronaca escluso dai piloti", "Verstappen" not in ctx.race.get("drivers", []),
        str(ctx.race.get("drivers")))
    add("Schermata fissa iniziale vista dalla scansione", any(a <= 4.0 and b >= 18.0 for a, b in (ctx.scan or {}).get("static_zones", [])))
    wc = ctx.webcam["rect"] if ctx.webcam else None
    exp = [ST_WEBCAM[0] / ST_W, ST_WEBCAM[1] / ST_H, ST_WEBCAM[2] / ST_W, ST_WEBCAM[3] / ST_H]
    add("Webcam del telecronista individuata", bool(wc) and all(abs(a - b) < 0.03 for a, b in zip(wc, exp)), str(wc))
    add("Retry su JSON malformato dello scout", fake.calls.get("scout", 0) >= 2 and ctx.llm.stats["retries"] >= 1,
        f"chiamate scout {fake.calls.get('scout', 0)}, retry {ctx.llm.stats['retries']}")
    add("Copy con hashtag e cliche' respinto e rigenerato", fake.calls.get("copy", 0) >= 3)
    if not ctx.args.no_continuity:
        add("Agente Continuita' interpellato per ogni candidata", fake.calls.get("continuity", 0) >= len(items),
            f"chiamate {fake.calls.get('continuity', 0)}")
    spans = sorted((it["trim"]["t_in"], it["trim"]["t_out"]) for it in items)
    add("Nessuna azione spezzata in clip che si sovrappongono",
        all(b[0] - a[1] > -2.0 for a, b in zip(spans, spans[1:])), str(spans))
    for it in finalists:
        kind = (it.get("judge") or {}).get("action_type", "?")
        tr, m, rep = it["trim"], it["metrics"], it.get("render", {})
        if kind == "SORPASSO":
            add("Sorpasso misurato dal tracciamento", m["overtakes"] >= 1, f"sorpassi {m['overtakes']}")
            add("Sorpasso: azione intera in una sola clip (dal primo attacco all'allungo finale)",
                tr["t_in"] <= 36.5 and tr["t_out"] >= 52.5 and tr["dur"] <= 45.0, f"{tr['t_in']:.1f}-{tr['t_out']:.1f}")
            add("Sorpasso: Reaction split come layout principale", "Reaction" in rep.get("principale", ""), str(rep.get("quota")))
        if kind == "DUELLO":
            add("Duello lungo: sorpasso e contro-sorpasso nella STESSA clip", m["overtakes"] >= 2 and tr["dur"] >= 28.0,
                f"sorpassi {m['overtakes']}, durata {tr['dur']:.1f} s")
            add("Duello lungo: la calma sul rettilineo non spezza l'azione", tr["t_in"] <= 153.0 and tr["t_out"] >= 176.0,
                f"{tr['t_in']:.1f}-{tr['t_out']:.1f}")
            if not ctx.args.no_continuity:
                add("Duello lungo: dalla preparazione all'allungo finale indicati dalla telecronaca",
                    tr["t_in"] <= 147.5 and tr["t_out"] >= 184.5, f"{tr['t_in']:.1f}-{tr['t_out']:.1f}")
        if kind == "CONTATTO":
            add("Contatto rilevato (box a contatto + botto + parola)", m["contacts"] >= 1, f"contatti {m['contacts']}")
            add("Contatto: clip agganciata al momento giusto", tr["t_in"] < 110.0 < tr["t_out"] and tr["dur"] <= 30.0,
                f"{tr['t_in']:.1f}-{tr['t_out']:.1f}")
            add("Contatto: Dual a due auto usato quando sono lontane", any("Dual" in k for k in rep.get("quota", {})), str(rep.get("quota")))
            add("Contatto: micro-shake applicato", rep.get("shake_impatto", 0) >= 1)
        vp = ctx.out_dir / DIR_VERTICAL / it["files"]["vertical"]
        mp = ctx.out_dir / DIR_MASTER / it["files"]["master"]
        try:
            v, mm = ffprobe_info(vp), ffprobe_info(mp)
            add(f"{kind}: 9:16 a 1080x1920 con audio", (v.width, v.height) == (OUT_W, OUT_H) and v.has_audio)
            add(f"{kind}: durata 9:16 = durata master (entro 2 frame)", abs(v.duration - mm.duration) <= 2.5 / ST_FPS,
                f"{v.duration:.3f} vs {mm.duration:.3f}")
            res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                                  "-of", "csv=p=0", str(vp)], capture_output=True)
            add(f"{kind}: audio a 48 kHz", res.stdout.decode().strip().startswith("48000"), res.stdout.decode().strip())
        except RaceClipperError as e:
            add(f"{kind}: file leggibili", False, str(e))
        add(f"{kind}: titolo senza hashtag e senza cliche'", "#" not in it["copy"]["title"] and "Chi vince" not in it["copy"]["title"],
            it["copy"]["title"])
    titles = [it["copy"]["title"] for it in finalists]
    add("Titoli tutti diversi", len(set(titles)) == len(titles))
    add("Nessun '#' nei file esportati e cartelle entro il limite",
        not final_checks(ctx, {k: ctx.out_dir / k for k in (DIR_VERTICAL, DIR_MASTER, DIR_SOCIAL)}, ctx.args.top_k))
    log("\n" + "=" * 88 + "\nESITO DEL COLLAUDO\n" + "=" * 88)
    for name, ok, detail in checks:
        log(f"  [{'OK' if ok else 'ERRORE'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    passed = all(ok for _, ok, _ in checks)
    log("\n" + ("COLLAUDO SUPERATO" if passed else "COLLAUDO NON SUPERATO") +
        f": {sum(ok for _, ok, _ in checks)}/{len(checks)} controlli. Clip di prova in {ctx.out_dir}")
    return passed


# ---------------------------------------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------------------------------------
def preflight(ctx: Ctx) -> None:
    args = ctx.args
    log("\n" + "=" * 88)
    log(f"RACECLIPPER v{VERSION} - regia composta, azione intera, NPU, SEO senza hashtag, top {args.top_k}")
    log("=" * 88)
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RaceClipperError(f"'{tool}' non trovato nel PATH. Installa FFmpeg e riapri il terminale.")
    ctx.encoder = pick_video_encoder(args.encoder)
    log(f" -> [ENCODER]   : {ctx.encoder}" + ("" if ctx.encoder == "h264_nvenc" else " (NVENC non disponibile: ripiego software)"))
    if not args.selftest:
        try:
            torch = _import_torch()
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                log(f" -> [GPU CUDA]  : {torch.cuda.get_device_name(0)} ({props.total_memory / 1024 ** 3:.1f} GB VRAM)")
            else:
                warn("CUDA non rilevata: Whisper e YOLO gireranno su CPU, molto piu' lenti.")
        except ImportError:
            warn("PyTorch non installato: servira' per trascrizione e analisi visiva.")
    ctx.llm = OllamaClient(args.ollama_url, allow_pull=not args.no_pull)
    ctx.llm.ensure_server()
    ctx.models["text"] = ctx.llm.resolve_model("regista (testo)", MODEL_PREFS["text"], args.model)
    ctx.models["vision"] = ctx.llm.resolve_model("occhio critico (visione)", MODEL_PREFS["vision"], args.vision_model)
    if args.copy_model == "same":
        ctx.models["copy"] = ctx.models["text"]
    else:
        ctx.models["copy"] = ctx.llm.resolve_model("SEO copywriter", MODEL_PREFS["copy"] + [ctx.models["text"]],
                                                   args.copy_model)
    log(f" -> [AGENTI]    : regista/revisore = {ctx.models['text']} | occhio = {ctx.models['vision']} | "
        f"copywriter = {ctx.models['copy']}")


def setup_screener(ctx: Ctx) -> None:
    if ctx.screener is not None:
        return
    if ctx.args.no_npu:
        log(" -> [NPU]       : disattivata da opzione (--no-npu)")
        return
    try:
        import openvino  # noqa: F401
    except ImportError:
        warn("OpenVINO non installato (pip install openvino): niente scansione NPU ne' webcam automatica.")
        return
    try:
        prefs = [ctx.args.npu_model] if ctx.args.npu_model else YOLO_NPU_PREFS
        ctx.screener = OpenVINOScreener(default_models_dir(), prefs)
        if ctx.screener.device == "NPU":
            log(f" -> [INTEL NPU] : ATTIVA ({ctx.screener.device_name}) con {ctx.screener.model_name}")
        else:
            log(f" -> [OpenVINO]  : NPU non esposta dai driver, lo screener gira su {ctx.screener.device} "
                f"({ctx.screener.device_name})")
    except Exception as e:
        warn(f"screener OpenVINO non disponibile: {str(e)[:200]}")


def choose_webcam(ctx: Ctx) -> None:
    arg = (ctx.args.webcam_box or "auto").strip().lower()
    if arg == "none":
        ctx.webcam = None
        return
    if arg != "auto":
        rect = parse_box_arg(arg)
        ctx.webcam = {"rect": rect, "person": rect, "source": "manuale"}
        return
    cams = sorted((ctx.scan or {}).get("webcams", []), key=lambda c: c["presence"], reverse=True)
    if not cams:
        return
    best = dict(cams[0])
    if len(cams) > 1:
        a, b = best["rect"], cams[1]["rect"]
        gap = max(a[0], b[0]) - min(a[2], b[2])
        v_overlap = (min(a[3], b[3]) - max(a[1], b[1])) / max(1e-6, min(a[3] - a[1], b[3] - b[1]))
        if gap < 0.05 and v_overlap > 0.6:        # due commentatori affiancati: un unico riquadro largo
            pa, pb = best["person"], cams[1]["person"]
            best["rect"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
            best["person"] = [min(pa[0], pb[0]), min(pa[1], pb[1]), max(pa[2], pb[2]), max(pa[3], pb[3])]
    ctx.webcam = {"rect": best["rect"], "person": best["person"], "source": "scansione NPU",
                  "presence": best.get("presence"), "framed": best.get("framed")}


def stage_media(ctx: Ctx) -> None:
    info, args = ctx.info, ctx.args
    feats_path = ctx.cache.root / "audio_features.npz"
    tr_key = {"model": WHISPER_MODEL, "lang": args.language}
    transcript = None if args.retranscribe else ctx.cache.load("transcript", tr_key)
    ctx.feats = None if args.fresh else load_audio_features(feats_path)
    audio: Optional[np.ndarray] = None
    if ctx.feats is None or transcript is None:
        log("\n[FASE 1] Audio: energia vocale, volume e transienti a passo 0,2 s...")
        audio = load_audio_16k(info, ctx.log_path("audio"))
        if ctx.feats is None:
            ctx.feats = compute_audio_features(audio)
            save_audio_features(feats_path, ctx.feats)
        log(f" -> {len(audio) / 16000 / 60:.1f} minuti di audio analizzati in RAM.")
    else:
        log("\n[FASE 1] Audio e trascrizione gia' in cache.")

    setup_screener(ctx)
    scan_key = {"screener": getattr(ctx.screener, "model_name", ""), "step": 2.0}
    scan = ctx.cache.load("npu_scan", scan_key) if ctx.screener is not None else None
    box: Dict[str, Any] = {}
    thread: Optional[threading.Thread] = None
    if ctx.screener is not None and scan is None:
        log(f"[FASE 2b] Scansione dell'intera live sullo screener ({ctx.screener.device}) in parallelo alla trascrizione...")

        def _scan() -> None:
            try:
                box["res"] = run_global_scan(info, ctx.screener, ctx.log_path("scansione"))
            except Exception as e:      # la scansione e' un aiuto: se fallisce la run prosegue
                box["err"] = e

        thread = threading.Thread(target=_scan, daemon=True)
        thread.start()
    if transcript is None:
        hints = ", ".join(x.strip() for x in (args.piloti, args.pista, args.campionato) if x and x.strip())
        transcript = transcribe_audio(audio, info.duration, args.language, hints)
        ctx.cache.save("transcript", tr_key, transcript)
    del audio
    gc.collect()
    if thread is not None:
        thread.join()
        if "res" in box:
            scan = box["res"]
            ctx.cache.save("npu_scan", scan_key, scan)
            log(f" -> Scansione completata: {scan['samples']} campioni in {scan['seconds']:.0f} s su {scan['device']}.")
        else:
            warn(f"scansione non riuscita: {str(box.get('err'))[:160]}")
    ctx.segments = transcript
    ctx.words = WordIndex(flatten_words(transcript))
    ctx.vocab = build_vocab(ctx.words.words)
    ctx.scan = scan
    ctx.overlay = unpack_overlay(scan)
    choose_webcam(ctx)
    if ctx.webcam:
        r = ctx.webcam["rect"]
        log(f" -> [WEBCAM]    : riquadro del telecronista ({ctx.webcam['source']}) a x {r[0] * 100:.0f}-{r[2] * 100:.0f}%, "
            f"y {r[1] * 100:.0f}-{r[3] * 100:.0f}% del frame.")
        if scan and scan.get("preview_b64"):
            try:
                (ctx.out_dir / "_controllo_webcam.jpg").write_bytes(base64.b64decode(scan["preview_b64"]))
            except Exception:
                pass
    else:
        log(" -> [WEBCAM]    : nessun riquadro webcam: il Reaction split non verra' usato "
            "(per indicarlo a mano: --webcam-box \"x,y,larghezza,altezza\" in percentuale).")
    if ctx.overlay is not None:
        log(f" -> [GRAFICHE]  : maschera delle grafiche fisse pari al {ctx.overlay.mean() * 100:.0f}% del frame.")


def stage_text(ctx: Ctx) -> List[Dict[str, Any]]:
    args = ctx.args
    log(f"\n[FASE 3] Agenti di testo con '{ctx.models['text']}': contesto, fasi della live, Action Detector...")
    ads, interviews = load_keyword_file(args.blacklist_file)
    zones = rule_blacklist(ctx.segments, args.skip_intro, ads, interviews)
    for a, b in (ctx.scan or {}).get("static_zones", []):
        zones.append((float(a), float(b), "schermata fissa"))
    ctx.blacklist = merge_zones(zones)
    base_key = {"text": ctx.models["text"], "piloti": args.piloti, "pista": args.pista, "camp": args.campionato,
                "zones": [[round(a), round(b)] for a, b, _ in ctx.blacklist], "n": len(ctx.segments)}
    race = ctx.cache.load("context", base_key)
    if race is None:
        race = extract_race_context(ctx)
        ctx.cache.save("context", base_key, race)
    ctx.race = race
    log(f" -> [CONTESTO]  : pista '{race.get('circuit') or '?'}' | campionato '{race.get('championship') or '?'}' | "
        f"piloti: {', '.join(race.get('drivers', [])[:18]) or 'non identificati'}")
    phases = ctx.cache.load("phases", base_key)
    if phases is None:
        phases = classify_phases(ctx)
        ctx.cache.save("phases", base_key, phases)
    ctx.phases = phases
    ctx.blacklist = merge_zones(ctx.blacklist + phase_zones(phases, ctx.info.duration))
    excluded = sum(b - a for a, b, _ in ctx.blacklist)
    log(f" -> [VERIFICA 1]: {len(ctx.blacklist)} zone escluse a priori ({excluded / 60:.1f} min fra spot, interviste, "
        f"classifiche e schermate fisse).")

    cand_key = dict(base_key, prompt=args.prompt, limit=args.max_candidates, phases=[p["phase"] for p in phases],
                    v=VERSION, max_dur=args.max_dur, continuity=not args.no_continuity)
    cands = ctx.cache.load("candidates", cand_key)
    if cands is None:
        found = scout_with_llm(ctx)
        peaks = scout_audio_peaks(ctx)
        max_len = max(150.0, float(args.max_dur) + 30.0)
        merged = merge_candidates(found + peaks, max_len=max_len)
        cands = rank_candidates(ctx, merged, args.max_candidates)
        log(f" -> Action Detector: {len(found)} momenti dalla telecronaca + {len(peaks)} picchi di voce = "
            f"{len(merged)} finestre dopo l'unione; {len(cands)} passano all'analisi visiva.")
        if cands and not args.no_continuity:
            n_before = len(cands)
            cands, widened, added = refine_story_windows(ctx, cands)
            cands = merge_candidates(cands, max_len=max_len)
            for i, c in enumerate(cands, start=1):
                c["id"] = i
            log(f" -> Agente Continuita': {widened} azioni allargate fino alla loro conclusione "
                f"(+{added:.0f} s in totale), {n_before - len(cands)} frammenti della stessa azione riuniti.")
        ctx.cache.save("candidates", cand_key, cands)
    else:
        log(f" -> {len(cands)} candidate gia' in cache.")
    ctx.llm.unload_all()
    return cands


def stage_analysis(ctx: Ctx, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    log(f"\n[FASE 4] Analisi visiva e taglio sull'azione intera per {len(cands)} candidate...")
    state: Dict[str, Any] = {"detector": None}
    overlay_sig = hashlib.sha1(np.packbits(ctx.overlay).tobytes()).hexdigest()[:8] if ctx.overlay is not None else ""
    max_dur, join_gap = float(ctx.args.max_dur), float(ctx.args.join_gap)

    def analysis_for(c: Dict[str, Any], extra_pre: float, extra_post: float) -> Dict[str, Any]:
        win = analysis_window(ctx, c, extra_pre, extra_post)
        key = {"win": list(win), "webcam": (ctx.webcam or {}).get("rect"), "overlay": overlay_sig,
               "tuning": [TUNING["det_fps"], TUNING["det_conf"], TUNING["analysis_width"]],
               "yolo": ctx.args.yolo_model, "screener": getattr(ctx.screener, "model_name", "")}
        name = f"analysis/win_{win[0]}_{win[1]}"
        an = ctx.cache.load(name, key)
        if an is None:
            if state["detector"] is None:
                ctx.llm.unload_all()            # nessun LLM in VRAM mentre lavora YOLO
                state["detector"] = ctx.detector_factory()
                log(f" -> Rilevatore: {getattr(state['detector'], 'name', '?')}")
            an = analyze_candidate(ctx, c, state["detector"], extra_pre, extra_post)
            ctx.cache.save(name, key, an)
        return an

    def make_item(c: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza e taglia; se l'azione tocca il bordo della finestra analizzata, analizza altro video e ripete."""
        extra_pre = extra_post = 0.0
        prev_win: Optional[Tuple[int, int]] = None
        an: Dict[str, Any] = {}
        cv: Dict[str, Any] = {}
        trim: Dict[str, Any] = {}
        grown = 0.0
        for _ in range(3):
            win = analysis_window(ctx, c, extra_pre, extra_post)
            if win == prev_win:
                break                           # zona esclusa o fine del video: la finestra non puo' crescere
            if prev_win is not None:
                grown += (prev_win[0] - win[0] + win[1] - prev_win[1]) / ctx.info.fps
            prev_win = win
            an = analysis_for(c, extra_pre, extra_post)
            cv = build_curves(ctx, an)
            trim = smart_trim(ctx, c, an, cv)
            if not (trim["open_start"] or trim["open_end"]):
                break
            if trim["open_end"]:
                extra_post += TUNING["grow_post"]
            if trim["open_start"]:
                extra_pre += TUNING["grow_pre"]
        trim["window_grown"] = round(grown, 1)
        return {"id": c.get("id", 0), "cand": c, "analysis": an, "trim": trim,
                "metrics": clip_metrics(ctx, an, cv, trim), "_cv": cv}

    def describe(it: Dict[str, Any]) -> str:
        tr, m = it["trim"], it["metrics"]
        extra = ""
        if tr.get("window_grown"):
            extra += f" | l'azione proseguiva: analizzati altri {tr['window_grown']:.0f} s"
        if tr.get("truncated"):
            extra += f" | PIU' LUNGA DEL MASSIMO ({max_dur:.0f} s): alza --max-dur per averla intera"
        return (f"{format_time(tr['t_in'])} | {tr['dur']:.1f} s ({tr['climax_kind']}, ingresso su {tr['snap_in']}, "
                f"uscita su {tr['snap_out']}) | lotta {m.get('close_battle_seconds', 0):.0f} s | sorpassi {m['overtakes']} | "
                f"contatti {m['contacts']} | fisica {m['physics_score_30']}/30{extra}")

    items: List[Dict[str, Any]] = []
    for c in cands:
        it = make_item(c)
        items.append(it)
        log(f" -> [{c['id']:02d}/{len(cands)}] {describe(it)}")

    # Clip che dopo il taglio risultano la stessa azione diventano UNA clip sola, rianalizzata sull'intervallo
    # unito: niente collage da fare a mano. Si uniscono solo se c'e' continuita' reale (si sovrappongono, oppure
    # nel passaggio dall'una all'altra le auto restano ingaggiate) e se entrambe portano azione: un commento a
    # pista vuota non viene "agganciato" a un sorpasso solo perche' arriva pochi secondi dopo.
    def interval_stats(pair: Sequence[Dict[str, Any]], t0: float, t1: float) -> Optional[Tuple[float, float, bool]]:
        """(quota di tempo con auto ingaggiate, secondi di lotta ravvicinata, eventi) oppure None se non analizzato."""
        if t1 - t0 < 0.4:
            return 0.0, 0.0, False
        best: Optional[Tuple[float, float, float, bool]] = None
        for it in pair:
            cv = it["_cv"]
            binw, nb = cv["bin"], cv["nb"]
            b0, b1 = int(math.floor((t0 - cv["w0"]) / binw)), int(math.ceil((t1 - cv["w0"]) / binw))
            c0, c1 = max(0, b0), min(nb, b1)
            cover = (c1 - c0) / max(1, b1 - b0)
            if c1 <= c0 or cover < 0.6 or (best is not None and cover <= best[0]):
                continue
            g = np.nan_to_num(cv["gap"][c0:c1], nan=99.0)
            marks = [t for t in list(cv["overtakes"]) + list(cv["contacts"]) + list(cv["spin_words"]) if t0 <= t <= t1]
            best = (cover, float((g < 1.5).mean()), float((g < TUNING["gap_close"]).sum()) * binw, bool(marks))
        return best[1:] if best else None

    def has_action(st: Optional[Tuple[float, float, bool]]) -> bool:
        return st is not None and (st[1] >= 2.0 or st[2])

    def merge_once() -> bool:
        items.sort(key=lambda x: x["trim"]["t_in"])
        for i in range(len(items) - 1):
            x, y = items[i], items[i + 1]
            x_in, x_out = x["trim"]["t_in"], x["trim"]["t_out"]
            y_in, y_out = y["trim"]["t_in"], y["trim"]["t_out"]
            overlap = x_out - y_in                    # > 0: si sovrappongono; < 0: distanza fra le due
            if -overlap > join_gap:
                continue
            # una delle due sta tutta dentro l'altra: e' lo stesso materiale, resta la piu' ampia
            inner = 1 if y_out <= x_out + 1.0 else 0 if (y_in - x_in <= 1.0 and x_out <= y_out + 1.0) else -1
            if inner >= 0:
                keep, drop = (x, y) if inner == 1 else (y, x)
                c = combine_cands(keep["cand"], drop["cand"])
                for key in ("start", "end", "story", "cover", "continues_after"):
                    c[key] = keep["cand"].get(key)
                keep["cand"] = c
                items.remove(drop)
                log(f" -> La candidata di {format_time(drop['trim']['t_in'])} e' gia' tutta dentro la clip di "
                    f"{format_time(keep['trim']['t_in'])}: resta una sola clip.")
                return True
            if y_out - x_in > max_dur:
                continue
            if overlap < 0 and any(za < y_in and zb > x_out for za, zb, _ in ctx.blacklist):
                continue                        # in mezzo c'e' uno spot o una schermata fissa
            lo_j, hi_j = min(x_out, y_in), max(x_out, y_in)
            if overlap >= 2.0:
                joined = True
            else:
                junction = interval_stats((y, x), lo_j - 1.5, hi_j + 1.5)
                joined = junction is not None and junction[0] >= 0.5
            if not (joined and has_action(interval_stats((x, y), x_in, lo_j))
                    and has_action(interval_stats((y, x), hi_j, y_out))):
                continue
            c = combine_cands(x["cand"], y["cand"])
            c["start"] = round(min(x["cand"]["start"], x_in), 1)
            c["end"] = round(max(x["cand"]["end"], y["cand"]["end"], y_out), 1)
            c["cover"] = [x_in, y_out]
            new = make_item(c)
            items[i:i + 2] = [new]
            log(f" -> Riunite in una sola clip le azioni di {format_time(x_in)} e {format_time(y_in)}: {describe(new)}")
            return True
        return False

    rounds = 0
    while rounds < 40 and merge_once():
        rounds += 1
    items.sort(key=lambda x: x["trim"]["t_in"])
    for i, it in enumerate(items, start=1):
        it["id"] = it["cand"]["id"] = i
        it.pop("_cv", None)
    if rounds:
        log(f" -> {len(items)} azioni distinte dopo aver riunito i frammenti.")
    if state["detector"] is not None:
        state["detector"].close()
    flush_vram()
    return items


def stage_verify(ctx: Ctx, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alive: List[Dict[str, Any]] = []
    for it in items:
        reason = precheck_candidate(it)
        if reason:
            it.update({"verdict": "SCARTATA", "reject_reason": reason, "score": 0, "flags": [], "breakdown": {}})
        else:
            alive.append(it)
    log(f"\n[FASE 5] Verifica 2: giudice visivo '{ctx.models['vision']}' su {len(alive)} candidate "
        f"({len(items) - len(alive)} gia' scartate come schermate fisse)...")
    frames_dir = ctx.work_dir / "fotogrammi"
    for i, it in enumerate(alive):
        times = vlm_frame_times(it["trim"], ctx.info)
        key = {"model": ctx.models["vision"], "times": times}
        name = f"vision/clip_{int(it['trim']['t_in'] * 10)}_{int(it['trim']['t_out'] * 10)}"
        vis = ctx.cache.load(name, key)
        if vis is None:
            frames = prepare_vlm_frames(ctx, it, frames_dir)
            vis = vision_judge(ctx, it, frames, last=(i == len(alive) - 1))
            if vis is not None:
                ctx.cache.save(name, key, vis)
        it["vision"] = vis
        if vis:
            log(f" -> [OCCHIO {it['id']:02d}] {vis['action']} | auto {vis['cars_visible_max']} (vicine {vis['cars_close_max']}) | "
                f"voto {vis['visual_score']}/20 | {vis['description'][:70]}")
        else:
            log(f" -> [OCCHIO {it['id']:02d}] nessun giudizio valido: la clip resta 'non verificata'")
    ctx.llm.unload_all()

    log(f"\n[FASE 6] Giudice editoriale '{ctx.models['text']}' e punteggi...")
    for i, it in enumerate(alive):
        key = {"model": ctx.models["text"], "t": [it["trim"]["t_in"], it["trim"]["t_out"]],
               "vis": (it.get("vision") or {}).get("description", ""), "m": it["metrics"]["physics_score_30"]}
        name = f"judge/clip_{int(it['trim']['t_in'] * 10)}_{int(it['trim']['t_out'] * 10)}"
        judge = ctx.cache.load(name, key)
        if judge is None:
            judge = editorial_judge(ctx, it, last=False)
            if judge is not None:
                ctx.cache.save(name, key, judge)
        it["judge"] = judge
        verify_and_score(it)
        status = f"SCARTATA: {it['reject_reason']}" if it["verdict"] != "OK" else f"{it['score']}/100"
        log(f" -> [GIUDICE {it['id']:02d}] {format_time(it['trim']['t_in'])} | {status}" +
            (f" | {'; '.join(it['flags'])}" if it["flags"] else ""))
    finalists = select_finalists(alive, ctx.args.top_k, ctx.args.min_score)
    log(f" -> Promosse {sum(1 for it in alive if it['verdict'] == 'OK')} clip sopra {ctx.args.min_score}/100, "
        f"finaliste {len(finalists)} (limite {ctx.args.top_k}).")
    return finalists


def stage_copy(ctx: Ctx, finalists: List[Dict[str, Any]]) -> None:
    if not finalists:
        return
    log(f"\n[FASE 6b] SEO Copywriter '{ctx.models['copy']}' (zero hashtag) e Verifica 3 del comitato editoriale...")
    if ctx.models["copy"] != ctx.models["text"]:
        ctx.llm.unload_all()
    titles: List[str] = []
    for it in finalists:
        it["copy"] = write_copy(ctx, it, titles)
        titles.append(it["copy"]["title"])
        log(f" -> [COPY {it['id']:02d}] {it['copy']['title']}")
    if ctx.models["copy"] != ctx.models["text"]:
        ctx.llm.unload_all()
    editorial_committee(ctx, finalists)
    ctx.llm.unload_all()


def stage_render(ctx: Ctx, finalists: List[Dict[str, Any]], items: List[Dict[str, Any]], stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    log(f"\n[FASE 7] Regia e render delle {len(finalists)} clip finali (9:16 composto + master 16:9)...")
    dirs = prepare_output_dirs(ctx.out_dir)
    done: List[Dict[str, Any]] = []
    for rank, it in enumerate(finalists, start=1):
        slug = sanitize_filename(it["copy"]["title"])
        base = (f"TOP{rank:02d}_SCORE_{it['score']:03d}_{format_time(it['trim']['t_in'])}_{slug}_"
                f"{int(round(it['trim']['dur']))}s")
        names = {"vertical": base + "_9x16.mp4", "master": base + "_16x9_MASTER.mp4", "social": base + "_SOCIAL_SEO.txt"}
        v_path, m_path = dirs[DIR_VERTICAL] / names["vertical"], dirs[DIR_MASTER] / names["master"]
        t0 = time.time()
        try:
            it["render"] = render_vertical(ctx, it, v_path)
            render_master(ctx, it, m_path)
        except RaceClipperError as e:
            warn(f"clip {rank:02d} non esportata: {e}")
            for p in (v_path, m_path):
                p.unlink(missing_ok=True)
            continue
        it["files"] = names
        done.append(it)
        r = it["render"]
        log(f" -> [TOP {rank:02d}] {it['score']}/100 | {r['principale']} | cambi layout {r['cambi_layout']} | "
            f"zoom punch {r['zoom_punch']} | shake {r['shake_impatto']} | audio {r['loudness']} | {time.time() - t0:.0f} s")
    done_ids = {id(it) for it in done}
    rejected = [it for it in items if id(it) not in done_ids]
    export_all(ctx, done, rejected, dirs, stats)
    problems = final_checks(ctx, dirs, ctx.args.top_k)
    for p in problems:
        warn(f"controllo finale: {p}")
    return done


def run_pipeline(ctx: Ctx) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    t_start = time.time()
    stats: Dict[str, Any] = {}
    stage_media(ctx)
    cands = stage_text(ctx)
    stats["candidate_analizzate"] = len(cands)
    if not cands:
        log("\nNessun momento d'azione individuato: nessuna clip esportata. Controlla la trascrizione o "
            "abbassa --skip-intro se la gara parte subito.")
        return [], []
    items = stage_analysis(ctx, cands)
    finalists = stage_verify(ctx, items)
    stats["scartate_dalle_verifiche"] = sum(1 for it in items if it.get("verdict") == "SCARTATA")
    if not finalists:
        log("\nNessuna candidata ha superato le verifiche: nessuna clip esportata. Il dettaglio degli scarti "
            f"e' in {FILE_REPORT}.")
        export_all(ctx, [], items, prepare_output_dirs(ctx.out_dir), stats)
        return [], items
    stage_copy(ctx, finalists)
    stats["ollama"] = dict(ctx.llm.stats)
    stats["minuti_totali"] = round((time.time() - t_start) / 60.0, 1)
    done = stage_render(ctx, finalists, items, stats)
    log("\n" + "=" * 88)
    log(f"FINITO: {len(done)} clip in {ctx.out_dir} ({(time.time() - t_start) / 60.0:.1f} min).")
    log(f"  {DIR_VERTICAL}  |  {DIR_MASTER}  |  {DIR_SOCIAL}  |  {FILE_ALL_SOCIAL}  |  {FILE_REPORT}")
    log("=" * 88)
    return done, items


def parse_rgb(text: str) -> Tuple[int, int, int]:
    try:
        r, g, b = [int(clamp(int(v), 0, 255)) for v in re.split(r"[,;\s]+", text.strip()) if v][:3]
        return r, g, b
    except Exception:
        raise RaceClipperError('--divider-color: usa "R,G,B", es. "255,255,255"')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"RaceClipper v{VERSION}: dalle dirette di gara 16:9 alle 10 migliori clip verticali 9:16.")
    p.add_argument("-i", "--input", default="", help="URL YouTube oppure file video locale")
    p.add_argument("-o", "--output", default="C:/RaceClipper/Output_Gara", help="cartella di output")
    p.add_argument("-p", "--prompt", default="Estrai solo le migliori azioni reali di gara e qualifica.",
                   help="indicazioni libere per l'Action Detector")
    p.add_argument("--piloti", default="", help="cognomi dei piloti separati da virgola (aiuta Whisper e i titoli)")
    p.add_argument("--pista", default="", help="nome del circuito, se noto")
    p.add_argument("--campionato", default="", help="nome del campionato o dell'evento, se noto")
    p.add_argument("--top-k", type=int, default=10, help="numero massimo di clip esportate (mai piu' di 10)")
    p.add_argument("--max-candidates", type=int, default=36, help="candidate passate all'analisi visiva")
    p.add_argument("--min-score", type=int, default=50, help="punteggio minimo per esportare una clip (0 = esporta comunque)")
    p.add_argument("--min-dur", type=float, default=8.0, help="durata minima di una clip (s)")
    p.add_argument("--max-dur", type=float, default=180.0,
                   help="tetto di sicurezza alla durata di una clip (s): l'azione viene tenuta intera fino a qui")
    p.add_argument("--join-gap", type=float, default=10.0,
                   help="due clip distanti meno di questi secondi vengono riunite in una, se fra l'una e l'altra "
                        "il duello prosegue (0 = solo se si sovrappongono)")
    p.add_argument("--no-continuity", action="store_true",
                   help="salta l'Agente Continuita' (rilettura della telecronaca per trovare inizio e fine dell'azione)")
    p.add_argument("--max-zoom", type=float, default=1.30, help="zoom massimo della camera virtuale")
    p.add_argument("--layout", choices=["auto", "full", "reaction", "wide", "dual"], default="auto",
                   help="regia: auto sceglie per ogni inquadratura; gli altri valori forzano un layout")
    p.add_argument("--webcam-box", default="auto",
                   help='riquadro webcam: auto, none oppure "x,y,larghezza,altezza" in percentuale del frame')
    p.add_argument("--divider-color", default="255,255,255", help="colore R,G,B della linea fra i pannelli")
    p.add_argument("--no-effects", action="store_true", help="niente zoom punch, shake, transizioni, vignettatura")
    p.add_argument("--no-replay-join", action="store_true", help="non unire il replay immediato all'azione")
    p.add_argument("--skip-intro", type=float, default=145.0, help="secondi iniziali da ignorare (sigla, countdown)")
    p.add_argument("--blacklist-file", default="", help='JSON con parole chiave: {"spot": [...], "interviste": [...]}')
    p.add_argument("--language", default="it", help="lingua della telecronaca per Whisper")
    p.add_argument("--model", default="", help="modello Ollama per regista, giudice e revisore")
    p.add_argument("--vision-model", default="", help="modello Ollama per il giudice visivo")
    p.add_argument("--copy-model", default="", help="modello Ollama per il copywriter ('same' = come --model)")
    p.add_argument("--yolo-model", default="", help="pesi YOLO per l'analisi su CUDA (es. yolo11x.pt)")
    p.add_argument("--npu-model", default="", help="pesi YOLO per lo screener OpenVINO (es. yolo11n.pt)")
    p.add_argument("--no-npu", action="store_true", help="non usare lo screener OpenVINO/NPU")
    p.add_argument("--no-pull", action="store_true", help="usa solo modelli Ollama gia' installati")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="indirizzo del server Ollama")
    p.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto", help="encoder video")
    p.add_argument("--max-height", type=int, default=1440, help="risoluzione massima scaricata da YouTube")
    p.add_argument("--retranscribe", action="store_true", help="rifai la trascrizione ignorando la cache")
    p.add_argument("--fresh", action="store_true", help="ignora tutte le cache di fase e riparti da zero")
    p.add_argument("--keep-cache", action="store_true", help="non cancellare i file temporanei a fine lavoro")
    p.add_argument("--selftest", action="store_true", help="collaudo completo senza GPU, Ollama e video reale")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")     # console Windows con code page legacy
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if not args.selftest and not args.input:
        build_parser().error("serve -i/--input (URL o file) oppure --selftest")
    args.top_k = int(clamp(args.top_k, 1, 10))
    args.max_candidates = max(args.top_k, args.max_candidates)
    args.max_dur = max(args.min_dur + 2.0, args.max_dur)
    fake: Optional[FakeOllama] = None
    try:
        args.divider_rgb = parse_rgb(args.divider_color)
        if args.selftest and args.output == "C:/RaceClipper/Output_Gara":
            args.output = str(Path.cwd() / "RaceClipper_Collaudo")
        out_dir = Path(args.output).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = out_dir / "_temp_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = out_dir / "_log"
        logs_dir.mkdir(parents=True, exist_ok=True)

        ctx = Ctx(args=args, out_dir=out_dir, work_dir=work_dir, logs_dir=logs_dir, cache=StageCache(work_dir))
        if args.selftest:
            log(f"\n[COLLAUDO] Genero una diretta sintetica di {ST_TOTAL:.0f} s e avvio l'intera catena con modelli simulati...")
            args.skip_intro, args.fresh = 0.0, True
            args.piloti = args.pista = args.campionato = ""
            args.model, args.vision_model, args.copy_model = FakeOllama.MODELS
            fake = FakeOllama()
            args.ollama_url, args.no_pull = fake.url, True
            video = work_dir / "diretta_sintetica.mp4"
            selftest_make_video(video, logs_dir / "collaudo.log")
            ctx.screener = SyntheticScreener()
            ctx.detector_factory = SyntheticDetector
            ctx.fast_encode = True
        else:
            video = acquire_video(args.input, work_dir, args.max_height)
            prefs = [args.yolo_model] if args.yolo_model else YOLO_CUDA_PREFS
            ctx.detector_factory = lambda: CudaVehicleDetector(default_models_dir(), prefs,
                                                               imgsz=int(TUNING["analysis_width"]),
                                                               conf=TUNING["det_conf"])
        ctx.info = ffprobe_info(video)
        info = ctx.info
        # La cache segue la gara: per un URL l'impronta e' l'URL stesso (resta valida se il file viene riscaricato).
        is_url = bool(re.match(r"^https?://", args.input or ""))
        fingerprint = hashlib.sha1(args.input.encode("utf-8")).hexdigest()[:12] if is_url else video_fingerprint(video)
        ctx.cache = StageCache(out_dir / "_cache" / fingerprint, enabled=not args.fresh)
        if args.selftest:
            ctx.cache.save("transcript", {"model": WHISPER_MODEL, "lang": args.language}, selftest_transcript())
            ctx.cache.enabled = True
        preflight(ctx)
        log(f" -> [SORGENTE]  : {info.path.name} | {info.width}x{info.height} | {info.fps:.3f} fps"
            + (f" (sorgente {info.src_fps:.1f} fps)" if abs(info.src_fps - info.fps) > 0.5 else "")
            + f" | {info.duration / 60:.1f} min | audio {'si' if info.has_audio else 'NO'}")
        done, items = run_pipeline(ctx)
        ok = True
        if args.selftest:
            ok = selftest_report(ctx, done, items, fake)
        if not args.keep_cache and not args.selftest:
            shutil.rmtree(work_dir, ignore_errors=True)
        return 0 if ok else 2
    except RaceClipperError as e:
        log(f"\nERRORE: {e}")
        return 1
    except KeyboardInterrupt:
        log("\nInterrotto dall'utente. Le fasi completate restano in cache: rilanciando il comando si riparte da li'.")
        return 130
    finally:
        if fake is not None:
            fake.stop()


if __name__ == "__main__":
    sys.exit(main())
