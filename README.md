# venv-llama-cpp-python-win

Ambiente virtual Python 3.10.2 para Windows 11 com `llama-cpp-python` pré-instalado, exposto como uma **API REST compatível com OpenAI** via FastAPI + uvicorn.  
Integra com extensões de IA no VSCode (como [Continue.dev](https://marketplace.visualstudio.com/items?itemName=Continue.continue)) como alternativa local ao GitHub Copilot.

---

## Estrutura do projeto

```
.
├── src/
│   ├── app.py          # FastAPI — endpoints /v1/chat/completions, /v1/completions, /v1/models
│   ├── chat.py         # Chat interativo via terminal (alternativa à API)
│   ├── config.py       # Configurações carregadas do .env
│   ├── downloader.py   # Download do modelo GGUF do Hugging Face
│   ├── inference.py    # Wrapper LLMInference (generate, chat, stream_chat)
│   └── schemas.py      # Modelos Pydantic no formato OpenAI
├── tests/              # Pytest — 36 testes, 99% de cobertura
├── model/              # Pasta onde o modelo .gguf é salvo (não versionado)
├── .env.example        # Template de variáveis de ambiente
├── requirements.txt    # Dependências Python
└── .github/workflows/
    └── build-release.yml  # Gera e publica o .venv no GitHub Releases
```

---

## Configuração

Copie `.env.example` para `.env` e ajuste os valores conforme necessário:

```bash
cp .env.example .env
```

### Variáveis disponíveis

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_FILENAME` | `qwen2.5-7b-instruct-q4_k_m.gguf` | Nome do arquivo GGUF |
| `MODEL_DIR` | `model` | Pasta onde o modelo é armazenado |
| `MODEL_ID` | `qwen2.5-7b-instruct` | ID reportado pelo endpoint `/v1/models` |
| `REPO_ID` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | Repositório do Hugging Face |
| `N_CTX` | `4096` | Tamanho da janela de contexto em tokens |
| `N_THREADS` | `(núcleos CPU)` | Threads de CPU para inferência |
| `N_GPU_LAYERS` | `0` | Camadas na GPU (0 = somente CPU) |
| `TEMPERATURE` | `0.7` | Temperatura de amostragem padrão |
| `MAX_TOKENS` | `512` | Máximo de tokens por resposta |
| `HOST` | `0.0.0.0` | Endereço de escuta da API |
| `PORT` | `8000` | Porta da API |
| `API_KEY` | *(vazio)* | Chave de autenticação (deixe vazio para desativar) |
| `AUTO_DOWNLOAD` | `true` | Baixa o modelo automaticamente na primeira execução |

> **Por que o modelo não está no repositório?**  
> O arquivo `.gguf` tem ~4,4 GB. O GitHub limita arquivos a 100 MB e o Git LFS gratuito a 1 GB.  
> O modelo é baixado automaticamente do Hugging Face na primeira execução da API.

---

## Instalação e uso

### Pré-requisitos

- Python 3.10.2 (recomendado: use o `.venv` do release)
- Windows 11 (para o release pré-compilado) ou Linux/macOS (compilação local)

### 1. Instalar dependências

```bat
.venv\Scripts\pip install -r requirements.txt
```

### 2. Configurar o ambiente

```bat
copy .env.example .env
:: Edite .env conforme necessário
```

### 3. Iniciar a API

```bat
.venv\Scripts\uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Na primeira execução, o modelo (~4,4 GB) é baixado automaticamente se `AUTO_DOWNLOAD=true`.  
Acesse a documentação interativa em `http://localhost:8000/docs`.

### 4. Chat via terminal (opcional)

```bat
.venv\Scripts\python -m src.chat
```

### 5. Baixar o modelo manualmente

```bat
.venv\Scripts\python -m src.downloader
```

---

## Endpoints da API

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e se o modelo está carregado |
| `GET` | `/v1/models` | Lista os modelos disponíveis |
| `POST` | `/v1/chat/completions` | Chat (suporta streaming via SSE) |
| `POST` | `/v1/completions` | Completion de texto |

### Exemplo de requisição

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "Olá, como vai?"}],
    "max_tokens": 256
  }'
```

Com streaming:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b-instruct", "messages": [{"role": "user", "content": "Olá"}], "stream": true}'
```

Com autenticação (quando `API_KEY` está definida):

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sua_chave_aqui"
```

---

## Integração com VSCode (Continue.dev)

O **GitHub Copilot** oficial não permite redirecionar para endpoints customizados.  
Use a extensão **[Continue.dev](https://marketplace.visualstudio.com/items?itemName=Continue.continue)** para obter a mesma experiência (autocompletar, chat inline, explicação de código) com o modelo local.

### Configuração do Continue.dev

Edite `~/.continue/config.json` (ou `%USERPROFILE%\.continue\config.json` no Windows):

```json
{
  "models": [
    {
      "title": "Qwen 2.5 7B (local)",
      "provider": "openai",
      "model": "qwen2.5-7b-instruct",
      "apiBase": "http://localhost:8000",
      "apiKey": "local"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Qwen 2.5 7B Autocomplete",
    "provider": "openai",
    "model": "qwen2.5-7b-instruct",
    "apiBase": "http://localhost:8000",
    "apiKey": "local"
  }
}
```

---

## Testes

```bat
:: Executar todos os testes
.venv\Scripts\pytest tests/

:: Com relatório de cobertura
.venv\Scripts\pytest tests/ --cov=src --cov-report=term-missing
```

- **36 testes** cobrindo API, inferência, download e chat
- **99% de cobertura** de código
- Não requer o modelo instalado — `llama_cpp` e `huggingface_hub` são mockados

---

## Release do .venv

O workflow `build-release.yml` gera automaticamente um `.venv` com `llama-cpp-python` pré-compilado para Windows 11.

### Via tag (recomendado)

```bash
git tag v1.0.0
git push origin v1.0.0
```

### Via workflow manual

GitHub → Actions → "Build venv llama-cpp-python (Windows)" → **Run workflow**

O arquivo `venv-llama-cpp-python-win11-py3.10.2.zip` é publicado na página de [Releases](../../releases).

### Como usar o release

1. Baixe e extraia o `.zip` na raiz do projeto
2. Ative o ambiente:
   ```bat
   .venv\Scripts\activate
   ```
3. Se necessário após extração (caminhos absolutos mudam):
   ```bat
   python -m venv --upgrade .venv
   ```

---

## Hardware recomendado (CPU only)

| Quantização | Tamanho | RAM necessária | Velocidade |
|---|---|---|---|
| `Q4_K_M` *(padrão)* | ~4,4 GB | ~6 GB | Melhor custo-benefício |
| `Q5_K_M` | ~5,2 GB | ~7 GB | Melhor qualidade |
| `Q8_0` | ~7,7 GB | ~10 GB | Quase sem perda |

Para uma máquina com **16 GB de RAM e CPU**, `Q4_K_M` ou `Q5_K_M` são as escolhas ideais.  
Ajuste `N_THREADS` no `.env` para o número de núcleos físicos do seu processador.
