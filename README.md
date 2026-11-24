# ðŸ RaceClipper AI

[!\[License: AGPL v3](https://img.shields.io/badge/License-AGPL\_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[!\[Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB.svg?logo=python\&logoColor=white)](https://www.python.org/)
[!\[NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA\_12.4\_\&\_NVENC-76B900.svg?logo=nvidia\&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[!\[Intel OpenVINO](https://img.shields.io/badge/Intel-NPU\_OpenVINO-0071C5.svg?logo=intel\&logoColor=white)](https://docs.openvino.ai/)
[!\[Ollama Local AI](https://img.shields.io/badge/Ollama-100%25\_Local\_AI-black.svg)](https://ollama.com/)

**RaceClipper AI** Ã¨ una pipeline **100% locale e multi-agente** progettata per trasformare dirette YouTube di gare Motorsport e Sim Racing (16:9, 1â€“3 ore) in clip verticali **9:16 (1080x1920)** pronte per **Instagram Reels, TikTok e YouTube Shorts**, accompagnate dai rispettivi **Master 16:9** sincronizzati al fotogramma per il post-montaggio.

Il tool combina **analisi acustica, trascrizione ad alta precisione, fisica visiva delle vetture e Vision AI** per selezionare solo le azioni reali in pista (sorpassi, duelli ruota a ruota, partenze, contatti, giri da pole position), eliminando automaticamente spot pubblicitari, pause e interviste.

\---

## âœ¨ FunzionalitÃ  Principali

* **100% Offline \& Privacy-First:** Nessun abbonamento cloud, nessun limite di minuti caricati. Tutta l'intelligenza artificiale gira sul tuo hardware locale tramite **CUDA**, **OpenVINO (Intel NPU)** e **Ollama**.
* **Architettura Multi-Agente ("Occhi + Orecchie"):**

  1. **Trascrizione Broadcast (`Faster-Whisper large-v3`):** Riconoscimento vocale completo a 32 layer con glossario motorsport e mappatura dell'energia vocale del telecronista.
  2. **Filtro Anti-Spazzatura \& Roster (`Qwen 2.5 32B`):** Individua ed esclude spot ripetuti, outlap e interviste post-gara; estrae il roster ufficiale di piloti, auto e circuito.
  3. **Radar Fisico (`YOLOv8x` 1280px + Intel NPU):** Misura la distanza reale tra i paraurti (*wheel-to-wheel*), gli incroci di traiettoria (sorpassi visivi), le bagarre di gruppo (*3-wide*) e i picchi di contatto.
  4. **Occhio Critico (`MiniCPM-V 8B`):** Ispeziona uno storyboard a 6 fotogrammi di ogni azione candidata per scartare i giri solitari anche in presenza di picchi audio.
  5. **Regia 9:16 Pro (`FFmpeg NVENC P7`):** Camera virtuale stabile con maschera cieca anti-minimappa/HUD, *Climax Punch-In Zoom*, scaling **Lanczos4**, ritocco colore e audio mastering **-14 LUFS**.
* **Doppio Export Sincronizzato:** Per ogni highlight salva sia la versione verticale `9:16` sia lo spezzone `16:9 MASTER` identico al fotogramma per eventuali rifiniture su DaVinci Resolve / Premiere Pro.
* **Viral Score Oggettivo (0â€“100) \& Kit SEO:** Classifica le clip migliori combinando telemetria visiva, audio e storytelling, generando titoli, hook e caption pronte per la pubblicazione.

\---

## ðŸ’» Requisiti Hardware Consigliati (Workstation Edition)

* **Sistema Operativo:** Windows 11 (64-bit)
* **GPU:** NVIDIA RTX con almeno 16 GB VRAM (testato su **RTX 4080 Super 16GB** con driver CUDA 12.4+)
* **CPU / NPU:** Intel Core Ultra (testato su **Intel Core Ultra 7 265K** con supporto OpenVINO NPU)
* **RAM:** 32 GB â€“ 48 GB DDR5 (consigliati 48 GB per eseguire `qwen2.5:32b` senza swap)
* **Dipendenze di sistema:** `FFmpeg` (con supporto `h264\_nvenc`), `yt-dlp`, `Ollama`, `Python 3.12`

\---

## ðŸš€ Installazione Rapida (Windows PowerShell)

### 1\. Installa gli strumenti di sistema e i modelli AI

```powershell
winget install -e --id Gyan.FFmpeg
winget install -e --id yt-dlp.yt-dlp
winget install -e --id Ollama.Ollama
winget install -e --id Python.Python.3.12
```

Riavvia PowerShell, quindi scarica i modelli locali su Ollama:

```powershell
ollama pull qwen2.5:32b
ollama pull minicpm-v:8b
```

### 2\. Clona il repository e configura l'ambiente CUDA

```powershell
git clone https://github.com/Ramacciotti95/RaceClipper-AI.git
cd RaceClipper-AI
py -3.12 -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

\---

## ðŸŽ¬ Utilizzo

### Da link di una Live YouTube

```powershell
python race\_clipper\_4080.py -i "https://www.youtube.com/watch?v=ID\_LIVE" --top-k 10
```

### Da file video giÃ  scaricato sul PC con prompt personalizzato

```powershell
python race\_clipper\_4080.py `
  -i "C:\\Videos\\Gara\_Live.mp4" `
  -p "Estrai partenze, contatti, duelli ruota a ruota, sorpassi decisivi e giri da pole position" `
  --top-k 10 `
  -o "C:\\RaceClipper\\Output\_Gara"
```

### Struttura dell'Output

```text
Output\_Gara/
â”œâ”€â”€ 01\_Verticale\_9x16/       # Clip 1080x1920 pronte e ordinate per Viral Score
â”œâ”€â”€ 02\_Master\_16x9/          # Tagli master 16:9 sincronizzati per post-montaggio
â”œâ”€â”€ 03\_Titoli\_e\_Caption/     # Schede testuali singole abbinate a ogni clip
â”œâ”€â”€ PRONTI\_PER\_I\_SOCIAL.txt  # Documento unico copia-incolla ordinato per classifica
â””â”€â”€ classifica\_viralita.json # Telemetria completa e scomposizione punteggi
```

\---

## âš–ï¸ Licenza e PaternitÃ 

Copyright Â© 2026 **Nicholas Ramacciotti**. Tutti i diritti riservati secondo i termini della licenza **GNU Affero General Public License v3.0 (AGPL-3.0)**.

Questo progetto Ã¨ distribuito sotto licenza **AGPL-3.0** (vedi il file [`LICENSE`](./LICENSE) e [`NOTICE.txt`](./NOTICE.txt)):

* Chiunque utilizzi, modifichi o ridistribuisca questo codice (anche come servizio web/SaaS) Ã¨ tenuto a **mantenere l'attribuzione all'autore originale** e a **rilasciare le modifiche sotto la stessa licenza open source**.
* Se utilizzi questo software in un tuo flusso di lavoro o progetto derivato, consulta il file [`CITATION.cff`](./CITATION.cff) per la citazione corretta.


