# venv-llama-cpp-python

Ambiente virtual Python 3.10.2 com `llama-cpp-python` pré-instalado, gerado automaticamente via GitHub Actions para Windows 11.

## Como gerar um release

### Via tag (recomendado)
```bash
git tag v1.0.0
git push origin v1.0.0
```

### Via workflow manual
GitHub → Actions → "Build venv llama-cpp-python (Windows)" → Run workflow

## Como usar o release

1. Baixe `venv-llama-cpp-python-win11-py3.10.2.zip` na página de Releases
2. Extraia na raiz do projeto
3. Ative:
   ```bat
   .venv\Scripts\activate
   ```
4. Se necessário, regere os scripts do venv (caminhos absolutos mudam após extração):
   ```bat
   python -m venv --upgrade .venv
   ```

## Conteúdo do ambiente

| Item | Versão |
|------|--------|
| Python | 3.10.2 |
| llama-cpp-python | latest stable |
| Target OS | Windows 11 |
