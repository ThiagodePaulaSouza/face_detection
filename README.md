# RTSP Face Crop com YuNet

Projeto em Python que conecta a uma câmera RTSP, detecta rostos usando o modelo **YuNet** (via OpenCV) e salva os recortes (crops) automaticamente em disco.

O YuNet é distribuído sob licença **MIT**, sendo totalmente livre para uso comercial.

---

## 📋 Requisitos

- Python **3.8+**
- Acesso à câmera RTSP (com URL, usuário e senha)
- Modelo YuNet `.onnx` (instruções abaixo)

---

## 🚀 Instalação

### 1. Clone ou baixe os arquivos do projeto

Coloque em uma pasta os seguintes arquivos:

```
projeto/
├── rtsp_yunet_crop.py
├── requirements.txt
└── README.md
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
```

Ative o ambiente:

- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixe o modelo YuNet

Baixe o arquivo `face_detection_yunet_2023mar.onnx` diretamente do repositório oficial do OpenCV Zoo:

**Link direto:**
```
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Ou via `wget` / `curl`:

```bash
# Linux/macOS
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "face_detection_yunet_2023mar.onnx"
```

Coloque o arquivo `.onnx` na **mesma pasta** do script `rtsp_yunet_crop.py`.

---

## ⚙️ Configuração

Abra o arquivo `rtsp_yunet_crop.py` e ajuste o bloco de configurações no topo:

```python
RTSP_URL = "rtsp://usuario:senha@192.168.1.100:554/stream"
MODEL_PATH = "face_detection_yunet_2023mar.onnx"
OUTPUT_DIR = "crops_faciais"
SCORE_THRESHOLD = 0.6      # Confiança mínima
NMS_THRESHOLD = 0.3
SAVE_INTERVAL = 1.0        # Segundos entre salvamentos
MIN_FACE_SIZE = 40         # Tamanho mínimo do rosto (px)
```

### Parâmetros importantes

| Parâmetro         | Descrição                                                                 |
|-------------------|---------------------------------------------------------------------------|
| `RTSP_URL`        | URL completa do stream RTSP, incluindo credenciais se necessário.         |
| `SCORE_THRESHOLD` | Confiança mínima (0–1). Valores mais altos = menos falsos positivos.      |
| `SAVE_INTERVAL`   | Intervalo em segundos entre crops salvos. Use `0` para salvar todos.      |
| `MIN_FACE_SIZE`   | Ignora rostos menores que esse valor (em pixels).                         |

---

## ▶️ Como executar

Com o ambiente virtual ativado e o modelo baixado:

```bash
python rtsp_yunet_crop.py
```

### O que acontece ao rodar

1. O script conecta ao stream RTSP informado.
2. Uma janela chamada **"RTSP - YuNet Detection"** abre mostrando o vídeo em tempo real com retângulos verdes ao redor dos rostos detectados.
3. Cada rosto detectado é salvo na pasta `crops_faciais/` com nome no formato:
   ```
   face_20241121_143022_345_score0.87.jpg
   ```
4. Pressione **`q`** na janela de vídeo (ou `Ctrl+C` no terminal) para encerrar.

### Saída esperada

```
[INFO] Conectando a rtsp://...
[INFO] Stream aberto: 1920x1080
[SALVO] crops_faciais/face_20241121_143022_345_score0.87.jpg (score=0.87, size=210x210)
[SALVO] crops_faciais/face_20241121_143023_812_score0.91.jpg (score=0.91, size=198x198)
...
[FINALIZADO] Total de crops salvos: 42
[FINALIZADO] Tempo de execução: 128.4s
```

---

## 🐳 Execução sem interface gráfica (servidor / Docker)

Em ambientes **headless** (servidores sem monitor, containers Docker, Raspberry Pi sem X11), o `cv2.imshow` falhará. Para rodar nesses casos, remova ou comente as linhas:

```python
cv2.imshow("RTSP - YuNet Detection", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

E também remova o `cv2.destroyAllWindows()` no `finally`. O restante do script funciona normalmente, salvando os crops e podendo ser encerrado com `Ctrl+C`.

---

## 🔧 Solução de problemas

### ❌ `[ERRO] Não foi possível abrir o stream RTSP`

- Verifique se a URL está correta (teste com **VLC** primeiro: `Mídia → Abrir Fluxo de Rede`).
- Confirme usuário e senha na URL: `rtsp://admin:senha@IP:554/path`.
- Verifique se o computador tem acesso à rede da câmera (ping, firewall).

### ❌ `cv2.error: ... model file not found`

- O arquivo `.onnx` não está na pasta correta ou tem nome diferente. Confirme o caminho em `MODEL_PATH`.

### ❌ Erros de `avcodec` / `ffmpeg` no Linux

Instale as bibliotecas de codec:

```bash
sudo apt install ffmpeg libavcodec-extra
```

### ❌ Detecção lenta

- Reduza a resolução do stream na câmera (ex: 1280x720 em vez de 4K).
- Ou redimensione o frame no código antes de passar ao detector (a bounding box precisará ser escalada proporcionalmente).

### ❌ Conexão cai com frequência

O script já tenta reconectar automaticamente. Se ainda assim falhar, aumente o `time.sleep(2)` na rotina de reconexão para `5`.

---

## 📁 Estrutura de saída

```
projeto/
├── rtsp_yunet_crop.py
├── requirements.txt
├── README.md
├── face_detection_yunet_2023mar.onnx
└── crops_faciais/
    ├── face_20241121_143022_345_score0.87.jpg
    ├── face_20241121_143023_812_score0.91.jpg
    └── ...
```

---

## 📜 Licença

- **Código deste projeto**: livre para uso comercial e modificação.
- **Modelo YuNet**: licença **MIT** (via [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).
- **OpenCV**: licença **Apache 2.0**.

Ambas as licenças permitem uso comercial sem royalties.