# ENEM-LLM-RAG

Classificação de questões do ENEM (2009–2024) por **habilidade** da Matriz de Referência oficial, usando um LLM local (Ollama) com saída estruturada.

## Motivação

As bases públicas de questões do ENEM trazem disciplina e ano, mas não a habilidade cognitiva específica exigida por cada questão (ex: H05, H23...) — informação que só existe na Matriz de Referência oficial do INEP, associada a cada questão manualmente. Esse projeto automatiza essa associação, usando um LLM para comparar o enunciado de cada questão às 120 descrições de habilidade da matriz e escolher a mais adequada.

## Pipeline

1. **Consolidação** — questões de múltiplas fontes públicas (2009–2023 e 2024) unificadas em um único JSON, com harmonização de schema entre as fontes.
2. **Rotulagem de disciplina** — inferida pela posição da questão na prova, com regra condicional ao ano (a ordem das áreas mudou na reforma de 2017).
3. **Limpeza** — extração do texto completo da alternativa correta, marcação de questões com figura (`[FIGURA]`), remoção de questões anuladas/sem resposta.
4. **Baseline clássico** (TF-IDF + Regressão Logística) — teste de viabilidade num problema mais simples (classificar disciplina, 4 classes) antes de atacar o problema real (120 habilidades). O resultado (overfitting mesmo após tuning) confirmou que uma abordagem supervisionada clássica não escalaria para o problema de 120 classes com poucos exemplos cada.
5. **Classificação via LLM local** — Ollama (`gemma3:4b`) com saída estruturada via Pydantic, restrita aos códigos da Matriz de Referência, com checkpointing e retry.

## Dataset

- ~2.937 questões (2009–2024), consolidadas a partir de fontes públicas (incluindo o repositório `yunger7/enem-api` e o dataset da Maritaca AI).
- Matriz de Referência: 120 habilidades, distribuídas em 4 áreas de conhecimento (Linguagens, Matemática, Ciências Humanas, Ciências da Natureza).

## Classificação via LLM — detalhes

Cada questão é enviada ao modelo junto da lista de habilidades da sua área, e a resposta é validada contra o schema:

```python
class HabilidadeClassificacao(BaseModel):
    raciocinio: str
    habilidade_principal: str   # ex: "H05"
    habilidade_secundaria: str  # "" na maioria dos casos
    confianca: float            # 0.0–1.0
    justificativa: str
```

- Códigos fora da matriz oficial da área geram um `alerta` para revisão manual, em vez de serem aceitos silenciosamente.
- Progresso salvo em checkpoint (CSV) a cada 25 questões, com retry em caso de falha de geração/validação.
- Processamento sequencial (a maioria dos servidores Ollama locais atende uma geração por vez).

## Resultados

Validação feita em amostra estratificada por área (8% de cada disciplina, 223 questões — ~7,6% do dataset), já que a execução completa levaria ~16-17h de processamento local sequencial:

| Métrica | Valor |
|---|---|
| Questões processadas | 223 |
| Erros técnicos | 0 |
| Códigos fora da matriz (`alerta`) | 4 (~1,8%) |
| Tempo total | ~1h16min (≈20,5s/questão) |

## Stack

Python · pandas · scikit-learn · Ollama (`gemma3:4b`) · Pydantic · matplotlib / seaborn · wordcloud · tqdm

## Como rodar

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
ollama pull gemma3:4b
```

O Ollama precisa estar rodando localmente (`ollama serve`) antes de executar as células de classificação.

## Próximos passos

- Validar uma amostra dos rótulos gerados contra julgamento humano/especialista.
- Priorizar revisão manual pelas questões com `confianca` baixa ou `alerta` ativo.
- Tratar separadamente as questões marcadas com `[FIGURA]`.
- Escalar a execução para o dataset completo.
