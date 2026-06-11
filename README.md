# venv-llama-cpp-python-win

Ambiente virtual Python 3.10.2 para Windows 11 com `llama-cpp-python` pré-instalado, exposto como uma **API REST compatível com OpenAI** via FastAPI + uvicorn.  
Integra com o [Continue.dev](https://marketplace.visualstudio.com/items?itemName=Continue.continue) no VSCode como alternativa local ao GitHub Copilot.  
Usa o **Qwen3-4B** por padrão — geração de código de alta qualidade com suporte a contexto longo (128k tokens).

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
├── tests/              # Pytest — 41 testes, 99% de cobertura
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
| `MODEL_FILENAME` | `Qwen3-4B-Q4_K_M.gguf` | Nome do arquivo GGUF |
| `MODEL_DIR` | `model` | Pasta onde o modelo é armazenado |
| `MODEL_ID` | `qwen3-4b` | ID reportado pelo endpoint `/v1/models` |
| `REPO_ID` | `Qwen/Qwen3-4B-GGUF` | Repositório do Hugging Face |
| `N_CTX` | `4096` | Tamanho da janela de contexto em tokens |
| `N_THREADS` | `(núcleos CPU)` | Threads de CPU para inferência |
| `N_GPU_LAYERS` | `0` | Camadas na GPU (0 = somente CPU) |
| `N_BATCH` | `512` | Tokens processados em paralelo no prompt (maior = menor latência, mais RAM) |
| `TEMPERATURE` | `0.1` | Temperatura de amostragem padrão |
| `MAX_TOKENS` | `512` | Máximo de tokens por resposta |
| `HOST` | `0.0.0.0` | Endereço de escuta da API |
| `PORT` | `8000` | Porta da API |
| `API_KEY` | *(vazio)* | Chave de autenticação (deixe vazio para desativar) |
| `AUTO_DOWNLOAD` | `true` | Baixa o modelo automaticamente na primeira execução |

> **Por que o modelo não está no repositório?**  
> O arquivo `.gguf` tem ~2 GB. O GitHub limita arquivos a 100 MB e o Git LFS gratuito a 1 GB.  
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

Na primeira execução, o modelo (~2 GB) é baixado automaticamente se `AUTO_DOWNLOAD=true`.  
Acesse a documentação interativa em `http://localhost:8000/docs`.

### 4. Chat via terminal (opcional)

```bat
.venv\Scripts\python -m src.chat
```

### 5. Baixar o modelo manualmente

```bat
.venv\Scripts\python -m src.downloader
```

Se ocorrer erro 404, o downloader lista automaticamente os arquivos disponíveis no repositório.  
Para explorar um repo antes de baixar:

```python
from src.downloader import list_gguf_files
print(list_gguf_files("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"))
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
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Explique este código"}],
    "max_tokens": 512
  }'
```

Com streaming:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-4b", "messages": [{"role": "user", "content": "Olá"}], "stream": true}'
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

O arquivo `continue.config.yaml` na raiz do projeto contém a configuração completa pronta para uso.  
Copie-o para `~/.continue/config.yaml` (macOS/Linux) ou `%USERPROFILE%\.continue\config.yaml` (Windows):

```bash
# macOS / Linux
cp continue.config.yaml ~/.continue/config.yaml

# Windows
copy continue.config.yaml %USERPROFILE%\.continue\config.yaml
```

Conteúdo completo do arquivo:

```yaml
name: Qwen3 GGUF Local
version: 1.0.0
schema: v1

models:
  - name: Qwen3 4B - Chat
    provider: openai
    model: qwen3-4b
    apiBase: "http://localhost:8000/v1"
    apiKey: "local"
    roles:
      - chat
      - edit
      - apply
    requestOptions:
      timeout: 120
    defaultCompletionOptions:
      temperature: 0.1
      maxTokens: 512

  - name: Qwen3 4B - Autocomplete
    provider: openai
    model: qwen3-4b
    apiBase: "http://localhost:8000/v1"
    apiKey: "local"
    roles:
      - autocomplete
    requestOptions:
      timeout: 30
    defaultCompletionOptions:
      temperature: 0.0
      maxTokens: 256

context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```

> No schema v1 do Continue.dev 1.2+, o `tabAutocompleteModel` foi removido.  
> O autocomplete é configurado adicionando `roles: [autocomplete]` em uma entrada de `models`.

#### Por que esses valores?

| Parâmetro | Chat | Autocomplete | Motivo |
|---|---|---|---|
| `maxTokens` | 512 | 256 | Menos tokens = resposta mais rápida |
| `temperature` | 0.1 | 0.0 | Código precisa de determinismo, não criatividade |
| `timeout` | 120s | 30s | Autocomplete deve ser rápido ou desistir |

#### Provedores de contexto (`context`)

| Provider | Atalho no chat | O que fornece |
|---|---|---|
| `code` | `@code` | Funções e classes do projeto |
| `docs` | `@docs` | Documentação de bibliotecas |
| `diff` | `@diff` | Alterações não commitadas |
| `terminal` | `@terminal` | Saída do terminal integrado |
| `problems` | `@problems` | Erros e avisos do editor |
| `folder` | `@folder` | Arquivos de uma pasta |
| `codebase` | `@codebase` | Busca semântica no projeto inteiro |

---

## Testes

```bat
:: Executar todos os testes
.venv\Scripts\pytest tests/

:: Com relatório de cobertura
.venv\Scripts\pytest tests/ --cov=src --cov-report=term-missing
```

- **41 testes** cobrindo API, inferência, download e chat
- **99% de cobertura** de código
- Não requer o modelo instalado — `llama_cpp` e `huggingface_hub` são mockados

---

## Release do .venv

O workflow `build-release.yml` gera automaticamente **dois builds** de `.venv` com `llama-cpp-python` pré-compilado para Windows 11:

| Arquivo | Backend | Indicado para |
|---|---|---|
| `*-cpu.zip` | CPU only | Qualquer máquina |
| `*-vulkan.zip` | Vulkan (GPU) | Intel Iris Xe / AMD Radeon integrada |

> O build Vulkan já inclui todas as dependências — não é necessário rodar `pip install -r requirements.txt` após extrair.

### Via tag (recomendado)

```bash
git tag v1.0.0
git push origin v1.0.0
```

### Via workflow manual

GitHub → Actions → "Build venv llama-cpp-python (Windows)" → **Run workflow**

Os arquivos são publicados na página de [Releases](../../releases).

### Como usar o release

1. Baixe e extraia o `.zip` adequado ao seu hardware na raiz do projeto
2. Ative o ambiente:
   ```bat
   .venv\Scripts\activate
   ```
3. Se necessário após extração (caminhos absolutos mudam):
   ```bat
   python -m venv --upgrade .venv
   ```

### Configuração para GPU integrada (build Vulkan)

No `.env`, configure:

```env
N_GPU_LAYERS=20   # Offload ~20 camadas para Intel Iris Xe / Radeon
N_BATCH=1024      # Maior batch aproveita melhor a GPU
N_THREADS=4       # Libera núcleos para o sistema
```

Ao iniciar a API, confirme o backend nos logs:

```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: Intel(R) Iris(R) Xe Graphics
```

Ganho esperado vs CPU: **~10–30% mais tokens/s** para modelos Q4_K_M.

---

## Hardware recomendado (CPU only)

Comparativo para 16 GB de RAM + CPU (série Qwen3):

| Modelo | Quantização | Tamanho | RAM | Velocidade CPU | Indicado para |
|---|---|---|---|---|---|
| Qwen3-1.7B | `Q4_K_M` | ~1,1 GB | ~2 GB | ~30–40 tok/s | Autocompletar rápido |
| **Qwen3-4B** ⭐ | `Q4_K_M` | ~2,6 GB | ~4 GB | ~12–18 tok/s | **Padrão — melhor equilíbrio** |
| Qwen3-8B | `Q4_K_M` | ~5,2 GB | ~7 GB | ~5–8 tok/s | Máxima qualidade |

> O Qwen3 suporta contexto de até 128k tokens e modo de raciocínio (`/think`). Para uso como assistente de código, o modo padrão (sem `<think>`) já é o adequado — nenhuma configuração extra é necessária.

Para trocar de modelo, edite três variáveis no `.env`:

```env
REPO_ID=Qwen/Qwen3-1.7B-GGUF
MODEL_FILENAME=Qwen3-1.7B-Q4_K_M.gguf
MODEL_ID=qwen3-1.7b
```

Ajuste `N_THREADS` para o número de núcleos físicos do seu processador.  
Para reduzir a latência do prompt, aumente `N_BATCH` para `1024` se tiver RAM disponível.
