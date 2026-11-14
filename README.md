# Analisador Musical

App web local que analisa sua biblioteca de músicas (MP3, FLAC, WAV, M4A, OGG…) e gera
prompts em texto para o Claude entender cada faixa.

## Como usar

Dê dois cliques em `iniciar.bat`. Na primeira vez ele cria o ambiente e instala o PyTorch para
RTX 50xx (CUDA 12.8), o que leva alguns minutos. Depois abre http://127.0.0.1:8765.

- **Adicionar pasta**: cole o caminho ou clique em 📁. Subpastas entram também. Reescanear a mesma
  pasta só enfileira arquivos novos ou alterados.
- **Enviar arquivos** ou arrastar para a janela: os arquivos são copiados para `data/uploads`.
- Clique numa faixa para ver forma de onda, seções, energia, espectro, o **prompt para o Claude**
  (4 objetivos: entender, mix/master, DJ, composição) e **músicas parecidas**.
- **Compatíveis**: filtra a biblioteca por tons vizinhos na roda de Camelot e BPM ±6%.

## Fazendo um vídeo com o Claude

O Claude não ouve áudio, então o app mede tudo e entrega os tempos prontos:

1. Abra a faixa, deixe o objetivo em **Vídeo sincronizado** e descreva o vídeo em "Contexto"
   (ex.: "clipe 9:16, estética neon, 30 fps, em Remotion").
2. Clique em **Copiar** e cole no Claude, junto com o MP3, que entra como trilha.
3. Para legendas palavra a palavra, anexe também o arquivo de **Baixar timing.json**.

O prompt traz: início de cada compasso em segundos, seções com tempo exato, subidas, drops,
quebras, build-ups, clímax, acentos percussivos fortes, letra sincronizada, curva de energia e tudo
isso de novo em JSON, para o Claude converter em frames (`frame = round(segundos × fps)`).

### Gerar o clipe sem sair do site

O botão **🎬 Gerar clipe com Claude** exporta o pacote e roda o Claude Code em modo automático dentro
da pasta. O log aparece ao vivo e, no fim, o MP4 toca na própria página. Usa a sua assinatura do
Claude, sem chave de API.

- Precisa do Claude Code instalado (`%USERPROFILE%\.local\bin\claude.exe`) e logado uma vez:
  `"%USERPROFILE%\.local\bin\claude.exe" auth login`.
- O agente só tem ferramentas de arquivo e Bash, e só pode rodar comandos de projeto de vídeo
  (`npx`, `npm`, `node`, `python`, `ffmpeg`/`ffprobe` e utilitários básicos). Qualquer outro comando é negado.
- Um clipe por vez. Leva de 10 a 40 minutos, conforme o modelo e o effort.
- Para continuar editando depois: abra a pasta do pacote no Claude Code e use `--resume` com o
  `session_id` que aparece no log (`claude_log.jsonl`).

## O que é analisado

| Item | Como |
|---|---|
| Batidas, downbeats, BPM, compasso | [Beat This!](https://github.com/CPJKU/beat_this) (rede neural, GPU) |
| Tonalidade + Camelot | chroma CQT + perfis de Krumhansl |
| Loudness LUFS, LRA, pico, crista, estéreo | pyloudnorm (ITU-R BS.1770) |
| Estrutura | autossimilaridade + curva de novidade, agrupamento das seções |
| Timbre, bandas, percussividade, densidade | librosa |
| Voz / bateria / baixo / outros por seção | Demucs htdemucs (GPU) |
| Instrumentos, gênero, clima | AST treinado no AudioSet (GPU) |
| Acordes e progressão principal | chroma dos stems harmônicos + Viterbi (tríades maiores/menores) |
| Letra com tempos | Whisper large-v3-turbo sobre a voz isolada (GPU) |

Na primeira análise os modelos são baixados (~3 GB no total, em `~/.cache`). Depois, uma faixa de
3 a 4 minutos leva uns 10–20 s. O modelo de gênero não conhece estilos brasileiros (sertanejo,
pagode, funk carioca…); o prompt avisa o Claude disso.

Para desligar partes pesadas: `MUSICA_STEMS=0`, `MUSICA_TAGS=0`, `MUSICA_LYRICS=0`. Para outro
modelo do Whisper: `MUSICA_WHISPER=openai/whisper-large-v3`. Depois de atualizar o analisador, use
**Atualizar análises antigas** no rodapé da biblioteca.

Os nomes das seções (Intro, Verso, Refrão, Ponte, Outro) são palpites heurísticos. As letras
(A, B, C…) indicam seções com material parecido e são mais confiáveis.

## Configuração

- `MUSICA_WORKERS`: número de análises em paralelo (padrão 2; a GPU é usada por uma de cada vez).
- `MUSICA_VRAM_FRACTION`: fração máxima da VRAM para o PyTorch (padrão 0.85; evita que o driver
  do Windows "transborde" para a RAM).
- O banco fica em `data/library.db` (SQLite). Apagar o arquivo zera a biblioteca.

## Estrutura

```
app/analyzer.py   motor de análise
app/prompt.py     geração dos prompts
app/library.py    SQLite + fila de análise
app/main.py       API FastAPI
app/static/       interface (HTML/CSS/JS, wavesurfer.js)
```
