# RTSP Face Crop com YuNet (multi-câmera)

Projeto em Python que conecta a uma ou mais câmeras RTSP, detecta rostos com o modelo **YuNet** (via OpenCV) e salva automaticamente os recortes (crops) em disco, organizados por fonte.

O YuNet é distribuído sob licença **MIT**, sendo livre para uso comercial.

**Principais recursos:**

- 🎥 **Multi-câmera**: 1 thread por fonte RTSP, isolamento total entre câmeras
- 🔄 **Reconexão automática** se o stream cair
- 🧠 **Download automático** do modelo YuNet (se ausente)
- ⚡ **Detecção opcional em GPU** (CUDA) com fallback para CPU
- 🖥️ **Headless-safe**: roda em servidores/Docker sem precisar comentar código
- 📝 **Logging estruturado** com rotação de arquivo em `./logs/`
- 📸 **Snapshots de debug** opcionais por fonte

---

## 📋 Requisitos

- Python **3.8+**
- Acesso às câmeras RTSP (URL, usuário e senha)
- Dependências Python: `opencv-python`, `numpy`

---

## 🚀 Instalação

### 1. Estrutura de arquivos

Coloque todos os arquivos do projeto na estrutura abaixo:

```
projeto/
├── main.py
├── utils.py
└── src/
    └── face_crop/
        ├── callbacks.py
        ├── detector.py
        ├── feed.py
        ├── headless.py
        ├── logging_setup.py
        └── worker.py
```

> Não é preciso baixar o `.onnx` manualmente: `utils.setup()` faz o download automaticamente na primeira execução.

### 2. Ambiente virtual (recomendado)

```bash
python -m venv venv
```

Ative:

- **Linux/macOS**: `source venv/bin/activate`
- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Windows (CMD)**: `venv\Scripts\activate.bat`

### 3. Instale as dependências

```bash
pip install opencv-python numpy
```

> Para suporte a GPU, veja a seção [🖥️ GPU / CUDA](#-gpu--cuda).

---

## ⚙️ Configuração

Toda a configuração de câmeras fica no topo de `main.py`, na lista `SOURCES`.

### Configurando fontes

```python
SOURCES: list[dict] = [
    {
        "url": "rtsp://admin:senha@192.168.0.10:554/Streaming/Channels/101",
        "name": "portaria",
        # opcionais (sobrescrevem os defaults globais):
        "score_threshold": 0.85,
        "min_face_size": 80,
        "debug_snapshot_seconds": 30.0,
    },
    {
        "url": "rtsp://admin:senha@192.168.0.11:554/Streaming/Channels/101",
        "name": "estacionamento",
    },
]
```

| Campo                     | Descrição                                                                 |
|---------------------------|---------------------------------------------------------------------------|
| `url` **(obrigatório)**   | URL completa do stream RTSP, incluindo credenciais.                       |
| `name` **(obrigatório)**  | Nome da fonte; usado como subpasta de saída e nos logs.                   |
| `score_threshold`         | Confiança mínima (0–1). Default global: **0.9**.                          |
| `min_face_size`           | Ignora rostos menores que esse valor (px). Default global: **100**.       |
| `debug_snapshot_seconds`  | Intervalo (s) entre snapshots anotados. `0` desliga. Default: **0.0**.    |

### Defaults globais (topo de `main.py`)

```python
OUTPUT_ROOT = "./results"           # raiz de saída
DEFAULT_SCORE_THRESHOLD = 0.9       # confiança mínima
DEFAULT_MIN_FACE_SIZE = 100         # tamanho mínimo do rosto (px)
DEFAULT_DEBUG_SNAPSHOT = 0.0        # 0 = desligado
SAVE_INTERVAL = 1.0                 # intervalo entre crops (s)
SHOW_FRAMES = True                  # abrir janela de preview
TRY_GPU = True                      # tentar usar CUDA
```

> **Nota**: `score_threshold` e `min_face_size` são aplicados por-fonte. Um valor mais alto no threshold reduz falsos positivos; `min_face_size` maior filtra rostos distantes.

### Variáveis de ambiente (`utils.py`)

Definidas com `setdefault` — só valem se ainda não existirem no ambiente:

| Variável                     | Default | Descrição                                                 |
|------------------------------|---------|-----------------------------------------------------------|
| `CUDA_VISIBLE_DEVICES`       | `"1"`   | Índice da GPU usada. **Ajuste** para `"0"` se tiver uma só. |
| `OPENCV_FFMPEG_LOGLEVEL`     | `"-8"`  | Silencia mensagens do FFmpeg.                             |

> ⚠️ Essas variáveis **precisam** estar definidas antes de qualquer `import cv2`. Por isso `main.py` importa `utils` primeiro. Se for importar o projeto de outro lugar, mantenha essa ordem.

---

## ▶️ Como executar

Com o ambiente virtual ativado:

```bash
python main.py
```

### O que acontece ao rodar

1. `utils.setup()` cria `./results/`, garante o modelo `.onnx` (baixa se preciso) e configura logs em `./logs/`.
2. Cada fonte em `SOURCES` vira uma thread (`Worker`) que:
   - conecta ao RTSP via FFmpeg;
   - instancia um `Detector` YuNet (tenta CUDA, cai pra CPU se indisponível);
   - roda detecção frame a frame;
   - salva crops (1 por `SAVE_INTERVAL` segundos por fonte);
   - opcionalmente grava snapshots anotados em `debug/`.
3. Se `SHOW_FRAMES=True`, abre uma janela por câmera (`RTSP - {name}`). Em ambiente headless, `try_imshow` detecta a falha e desabilita a janela automaticamente — o worker continua rodando.
4. Log de progresso (`frames lidos`) a cada **30s** por worker.

### Encerrar

Pressione **`Ctrl+C`** no terminal. Sinais `SIGINT` e `SIGTERM` são capturados, o `stop_event` é setado, os workers terminam e o processo sai.

Se algum worker travar (ex.: `cap.read()` bloqueado em RTSP), há um timeout de **3s** por join; se ainda vivo, o processo é encerrado com `os._exit(1)`.

### Saída esperada

```
2025-01-15 14:30:22 | INFO    | MainThread         | __main__ | Modelo: /caminho/face_detection_yunet_2023mar.onnx
2025-01-15 14:30:22 | INFO    | MainThread         | __main__ | Iniciando 2 worker(s)
2025-01-15 14:30:23 | INFO    | worker-portaria    | src.face_crop.feed | [portaria] Conectado: 1920x1080
2025-01-15 14:30:23 | INFO    | worker-portaria    | src.face_crop.detector | YuNet: backend CUDA FP16 (devices=1)
2025-01-15 14:30:25 | INFO    | worker-portaria    | src.face_crop.callbacks | [portaria] Crop salvo: 20250115_143025_123_score0.94.jpg (score=0.94, 210x210)
...
```

---

## 📁 Estrutura de saída

```
projeto/
├── main.py
├── utils.py
├── src/face_crop/...
├── logs/
│   └── face_crop.log                # rotação: 10 MB × 5 backups
├── results/
│   └── {name}/                      # ex.: portaria
│       ├── crops/
│       │   ├── 20250115_143025_123_score0.94.jpg
│       │   └── ...
│       └── debug/                   # só se debug_snapshot_seconds > 0
│           └── snapshot_20250115_143030.jpg
└── face_detection_yunet_2023mar.onnx
```

- O nome da fonte é sanitizado (`[^a-zA-Z0-9_-]` → `_`) na pasta de `crops/`.
- Formato dos crops: `{YYYYmmdd_HHMMSS_mmm}_score{score:.2f}.jpg`.

---

## 📝 Logs

Logs vão para `./logs/face_crop.log` (rotação de 10 MB, 5 backups) e para o stdout, no formato:

```
YYYY-MM-DD HH:MM:SS | LEVEL   | threadName         | logger             | mensagem
```

Silencia `urllib3` (nível WARNING). Para ajustar nível, modifique `setup_logging("./logs", logging.INFO)` em `main.py` (use `logging.DEBUG` para ver `input_size` do YuNet etc.).

---

## 🖥️ GPU / CUDA

`TRY_GPU=True` (default) faz o `Detector`:

1. Checar `cv2.cuda.getCudaEnabledDeviceCount()`.
2. Se > 0, setar `DNN_BACKEND_CUDA` + `DNN_TARGET_CUDA_FP16`.
3. Qualquer falha → **fallback silencioso para CPU** (com log de warning).

**Importante**: `pip install opencv-python` **não** vem com CUDA. Para GPU:

- Compile o OpenCV com `-DWITH_CUDA=ON` **ou**
- Use uma build pronta com CUDA (ex.: [opencv-python-cuda](https://pypi.org/project/opencv-python-cuda/) ou imagem Docker com CUDA).

Para detecção em tempo real com YuNet, **CPU costuma ser suficiente** na maioria dos casos (modelo leve). Use GPU só se estiver com muitas fontes simultâneas em 4K.

Se você só tem **uma** GPU, ajuste `CUDA_VISIBLE_DEVICES`:

```bash
# Linux/macOS
CUDA_VISIBLE_DEVICES=0 python main.py

# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=0; python main.py
```

---

## 🐳 Execução headless (servidor / Docker)

**Não é necessário comentar código.** O `try_imshow` detecta a ausência de display e desabilita a janela automaticamente, mantendo o worker rodando normalmente.

Para pular totalmente o `imshow` (economiza CPU), basta:

```python
SHOW_FRAMES = False
```

Exemplo de `Dockerfile` mínimo:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir opencv-python-headless numpy
# use opencv-python-headless em headless — mais leve
CMD ["python", "main.py"]
```

> Troque `opencv-python` por `opencv-python-headless` em ambientes sem display — a API é idêntica, mas sem dependências de GUI.

---

## 🔧 Solução de problemas

### ❌ `Falha ao abrir RTSP`

- Teste a URL no **VLC** (`Mídia → Abrir Fluxo de Rede`) antes de rodar o código.
- Confirme formato: `rtsp://usuario:senha@IP:554/path`.
- Verifique firewall/rede (ping no IP da câmera).
- O `Feed` já reconecta a cada **2s**; se a rede for instável, aumente `reconnect_delay` no `Worker` (`Feed(self.url, self.source_name)` → `Feed(self.url, self.source_name, reconnect_delay=5.0)`).

### ❌ `Modelo não encontrado; baixando...` falha no download

- Verifique acesso à internet a partir do host.
- Baixe manualmente e coloque `face_detection_yunet_2023mar.onnx` na **raiz do projeto**:
  ```bash
  wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  ```

### ❌ Erros de `avcodec` / `ffmpeg` no Linux

```bash
sudo apt install ffmpeg libavcodec-extra
```

### ❌ Detecção lenta

- Reduza a resolução na câmera (ex.: 1280×720 em vez de 4K).
- Aumente `min_face_size` para reduzir frames com muitos rostos distantes.
- Verifique o log: `backend=cuda_fp16` indica GPU ativa; `backend=cpu` significa fallback.

### ❌ `cv2.cuda` presente mas `getCudaEnabledDeviceCount()` retorna 0

Sua build do OpenCV tem o módulo `cuda` compilado mas sem device acessível. Causas comuns:

- Driver NVIDIA desatualizado.
- Container sem `--gpus all`.
- `CUDA_VISIBLE_DEVICES` aponta pra GPU inexistente (ex.: default `"1"` com só 1 GPU → ajuste pra `"0"`).

O código já loga o motivo e cai para CPU automaticamente.

### ❌ Workers ainda vivos após 3s (log `forçando saída`)

Normal quando `cap.read()` está bloqueado em uma conexão RTSP morta (FFmpeg não respeita sempre `READ_TIMEOUT_MSEC`). O processo é encerrado via `os._exit(1)` — aceitável, pois a saída é intencional. Se for frequente, reduza `OPEN_TIMEOUT_MSEC`/`READ_TIMEOUT_MSEC` em `Feed` ou aumente `WORKER_JOIN_TIMEOUT` em `main.py`.

---

## 📜 Licença

- **Código deste projeto**: livre para uso comercial e modificação.
- **Modelo YuNet**: **MIT** (via [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).
- **OpenCV**: **Apache 2.0**.

Ambas permitem uso comercial sem royalties.

Para mais informações: [audit_license.txt](./audit_license.txt)