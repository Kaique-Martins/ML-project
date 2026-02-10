# Projeto ML com TensorFlow e Docker

Este repositório fornece um scaffold para um projeto de Machine Learning usando TensorFlow, com código para:

- carregar e pré-processar dados (`src/data.py`)
- construir e treinar um modelo Keras (`src/model.py`, `src/train.py`)
- servir o modelo via API REST com Flask (`src/serve.py`)
- exemplos e notebooks em `notebooks/` e scripts úteis em `examples/`

**Objetivo**: ter um fluxo end-to-end mínimo (dados → treino → inferência) que você possa extender para projetos reais.

**Estrutura principal**
- `src/` — código fonte (data, model, train, serve)
- `models/` — local padrão para salvar modelos (ignorado pelo Git)
- `data/` — datasets gerados/baixados (ex.: `data/iris.csv`)
- `notebooks/` — notebooks de exploração
- `examples/prepare_iris.py` — script para gerar um CSV de exemplo (Iris)

**Requisitos recomendados**
- Python 3.11 (recomendado para compatibilidade com TensorFlow 2.12)
- Docker (opcional, para empacotar e rodar a aplicação isolada)

**1) Instalação rápida (venv, PowerShell)**
```powershell
cd C:\Users\kaique.santos\Downloads\projeto
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Se `tensorflow` não instalar via `pip` na sua plataforma, veja a seção "Instalando TensorFlow" abaixo.

**2) Gerar dados de exemplo (Iris)**
```powershell
python examples\prepare_iris.py
```
Isto cria `data/iris.csv`.

**3) Treinar o modelo**
```powershell
python -m src.train --data data/iris.csv --target target --model-dir models/iris --epochs 10
```
O script salva o modelo Keras em `models/iris` e o `scaler.joblib` para inferência.

**4) Servir e inferir**
```powershell
python -m src.serve --model models/iris
```
Endpoint de inferência: POST `/predict` com JSON {"instances": [[feat1,...], ...]} retorna `{ "predictions": [...] }`.

Exemplo PowerShell para testar (substitua as features conforme o dataset):
```powershell
#$payload = @{ instances = @(@(5.1,3.5,1.4,0.2)) } | ConvertTo-Json -Compress
#Invoke-RestMethod -Uri http://localhost:8080/predict -Method Post -Body $payload -ContentType 'application/json'
```

**5) Docker (build & run)**
```powershell
docker build -t meu-ml-app:latest .
docker run -p 8080:8080 meu-ml-app:latest
```

O `Dockerfile` copia `src/` e instala `requirements.txt` — o container isola dependências e geralmente evita problemas locais de instalação.

**Instalando TensorFlow (notas)**
- Em Windows é comum preferir Conda/Miniconda para criar um ambiente com Python 3.11:
```powershell
conda create -n tf311 python=3.11 -y
conda activate tf311
pip install --upgrade pip
pip install tensorflow==2.12.0 pandas scikit-learn joblib numpy matplotlib seaborn requests
```
- Alternativamente, instale `tensorflow` dentro do venv se sua versão do Python for compatível:
```powershell
pip install tensorflow==2.12.0
# ou, se disponível para sua plataforma: pip install tensorflow-cpu
```

Se `pip install tensorflow` falhar, use Conda ou rode via Docker (o container pode usar uma imagem base compatível).

**Testes e desenvolvimento**
- Crie testes para `src/data.py` e funções de pré-processamento com `pytest` em `tests/`.
- Notebooks em `notebooks/` demonstram EDA e pipeline de experimento.

**Boas práticas**
- Não comite os modelos (a pasta `models/` está em `.gitignore`).
- Versione os parâmetros dos experimentos (um arquivo YAML/JSON ou nomes de pastas por run).
- Para produção, substitua Flask por um servidor ASGI/ML-serving (ex.: TorchServe, TF-Serving, FastAPI + Uvicorn + Gunicorn).

Se quiser, eu posso:
- adicionar `tests/` com exemplos de pytest;
- criar um `docker-compose.yml` para desenvolvimento;
- ajustar o `Dockerfile` para suportar builds de GPU.

---
Arquivo atual: [README.md](README.md)