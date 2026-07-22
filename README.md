# ENEM-LLM-RAG

Classificação de questões do ENEM (2009–2024) por **habilidade** da Matriz de Referência oficial, usando um LLM com saída estruturada — com uma extensão que usa busca vetorial (RAG) para reduzir o número de habilidades candidatas apresentadas ao modelo.

## Motivação

As bases públicas de questões do ENEM trazem disciplina e ano, mas não a habilidade cognitiva específica exigida por cada questão (ex: H05, H23...) — informação que só existe na Matriz de Referência oficial do INEP, associada a cada questão manualmente. Esse projeto automatiza essa associação, usando um LLM para comparar o enunciado de cada questão às descrições de habilidade da matriz e escolher a mais adequada.

## Pipeline

1. **Consolidação** — questões de múltiplas fontes públicas (2009–2023 e 2024) unificadas em um único JSON, com harmonização de schema entre as fontes.
2. **Rotulagem de disciplina** — inferida pela posição da questão na prova, com regra condicional ao ano (a ordem das áreas mudou na reforma de 2017).
3. **Limpeza** — extração do texto completo da alternativa correta, marcação de questões com figura (`[FIGURA]`), remoção de questões anuladas/sem resposta.
4. **Baseline clássico** (TF-IDF + Regressão Logística) — teste de viabilidade num problema mais simples (classificar disciplina, 4 classes) antes de atacar o problema real (120 habilidades). O resultado (overfitting mesmo após tuning) confirmou que uma abordagem supervisionada clássica não escalaria para o problema de 120 classes com poucos exemplos cada.
5. **Classificação via LLM** — saída estruturada via Pydantic, restrita aos códigos da Matriz de Referência, com checkpointing e retry.
6. **Extensão com RAG** — em vez de enviar as 30 habilidades da área inteira no prompt, uma busca vetorial (ChromaDB) recupera só as mais semanticamente relevantes para cada questão.

## Dataset

- ~2.937 questões (2009–2024), consolidadas a partir de fontes públicas (incluindo o repositório `yunger7/enem-api` e o dataset da Maritaca AI).
- Matriz de Referência: 120 habilidades, distribuídas em 4 áreas de conhecimento (Linguagens, Matemática, Ciências Humanas, Ciências da Natureza).

## Classificação via LLM — detalhes

Cada questão é enviada ao modelo junto da lista de habilidades candidatas, e a resposta é validada contra o schema:

```python
class HabilidadeClassificacao(BaseModel):
    raciocinio: str
    habilidade_principal: str   # ex: "H05"
    habilidade_secundaria: str  # "" na maioria dos casos
    confianca: float            # 0.0–1.0
    justificativa: str
```

- Códigos fora da matriz oficial da área geram um `alerta` para revisão manual, em vez de serem aceitos silenciosamente.
- Progresso salvo em checkpoint (CSV), com retry em caso de falha de geração/validação.

## Extensão com RAG

**Problema que motivou a extensão:** a versão original envia as 30 habilidades da área inteira no prompt para cada questão. Isso funciona, mas (a) aumenta o tempo de inferência por questão e (b) expõe o modelo a candidatos irrelevantes, que — como a avaliação manual abaixo mostrou — às vezes levam a erros categóricos grosseiros (ex: escolher uma habilidade de Língua Estrangeira Moderna para um texto em português, ou uma habilidade de Biologia para uma questão de Física ondulatória).

**Implementação:**
- Embeddings das 120 habilidades gerados com `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, multilíngue, roda em CPU).
- Indexados no **ChromaDB** (distância de cosseno), com `area` como metadado para filtrar a busca.
- Para cada questão, os **8 candidatos mais similares** dentro da área correta substituem a lista completa de 30 no prompt.

## Resultados

Validação feita em amostra estratificada por área (8% de cada disciplina, 223 questões), antes de escalar para o dataset completo:

| Métrica | Valor |
|---|---|
| Questões processadas | 223 |
| Erros técnicos | 0 |
| Códigos fora da matriz (`alerta`) | 4 (~1,8%) |

Execução completa (2.783 questões, versão sem RAG) e comparação direta contra a versão com RAG (amostra de 200 questões, 50 por área, mesmas questões nas duas versões):

| Métrica | Sem RAG | Com RAG |
|---|---|---|
| Tempo médio por questão | 62,37 s | 46,94 s (**~25% mais rápido**) |
| Confiança média auto-relatada | 0,95 | 0,95 |
| Casos de confiança baixa (<0,9) | espalhados em várias faixas | praticamente inexistentes |

A concordância exata entre as duas versões nas 200 questões comparadas foi de apenas **33%** — um número baixo demais para ignorar. Para investigar se essa divergência refletia queda ou ganho de qualidade, uma amostra de **15 casos divergentes foi avaliada manualmente**, lendo o enunciado completo de cada questão:

| Avaliação manual (15 casos divergentes) | Contagem |
|---|---|
| RAG mais correto | 9 |
| Sem RAG mais correto | 5 |
| Nenhuma das duas / inconclusivo | 1 |

A leitura qualitativa mostrou um padrão: a versão sem RAG cometeu **erros categóricos** — escolher habilidades de Língua Estrangeira Moderna para textos em português, ou uma habilidade de Biologia para uma questão de ondas mecânicas — algo que a pré-seleção semântica do RAG evitou na maioria dos casos. **A extensão com RAG, além de mais rápida, mostrou-se qualitativamente superior na amostra avaliada.**

## Stack

Python · pandas · scikit-learn · Ollama (`qwen2.5:7b`) · Pydantic · ChromaDB · sentence-transformers · UMAP · matplotlib / seaborn · wordcloud · tqdm

## Como rodar

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
ollama pull qwen2.5:7b   # modelo usado na execução completa e na extensão com RAG
# ollama pull gemma3:4b  # modelo usado na validação inicial de 223 questões
```

O Ollama precisa estar rodando localmente (`ollama serve`) antes de executar as células de classificação. A extensão com RAG requer `chromadb` e `sentence-transformers` (ver `requirements.txt`).

## Próximos passos

- Ampliar a avaliação manual (15 → amostra maior) para consolidar estatisticamente a vantagem observada do RAG.
- Testar `n_results` diferente de 8 (candidatos recuperados por questão) para verificar sensibilidade do resultado a esse parâmetro.
- Explorar few-shot dinâmico (recuperar questões já classificadas como exemplo no prompt), como extensão futura do RAG.
- Tratar separadamente as questões marcadas com `[FIGURA]`.
- Escalar a execução com RAG para o dataset completo.
