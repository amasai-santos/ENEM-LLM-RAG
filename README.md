# ENEM Classifier

Pipeline de extração, limpeza e consolidação de questões do ENEM (2009–2024) para treinamento de modelos de classificação por **disciplina**, **tópico** e **competência/habilidade**.

## Objetivo

Construir um dataset unificado de questões do ENEM a partir de múltiplas fontes públicas, para servir de base a um classificador (com potencial de virar API ou dashboard de consulta).

## Fontes de dados

| Período | Fonte | Observações |
|---|---|---|
| 2009–2023 | [`yunger7/enem-api`](https://github.com/yunger7/enem-api) | JSONs oficiais por questão (pasta `public/{ano}/questions/`), disciplina já rotulada pela fonte |
| 2024 | [`piresramon/gpt-4-enem`](https://github.com/piresramon/gpt-4-enem) (dataset Maritaca AI / [`maritaca-ai/enem`](https://huggingface.co/datasets/maritaca-ai/enem) no Hugging Face) | Sem campo `discipline` original — inferido pela posição da questão (ver seção *Limitações*) |
| 2025 | Provas e gabaritos oficiais do [INEP](https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem/provas-e-gabaritos/2025) | Extraído via pipeline próprio em PDF (`extracao_pdf_labeling.ipynb`) |

## Notebooks

- **`extracao_pdf_labeling.ipynb`** — Extrai questões diretamente dos cadernos de prova em PDF (classe `EnemExamCorrigido`, baseada em PyMuPDF), trata layout de duas colunas, identifica e sinaliza questões com encoding corrompido, e separa enunciado/alternativas.
- **`consolidacao_dataset.ipynb`** — Junta as questões de todas as fontes/anos em um único JSON (`enem_completo.json`), padroniza o schema e preenche disciplina ausente.

## Schema

Cada questão segue esta estrutura:

```json
{
  "title": "Questão 1 - ENEM 2009",
  "index": 1,
  "year": 2009,
  "language": null,
  "discipline": "ciencias-humanas",
  "context": "...",
  "files": [],
  "correctAlternative": "C",
  "alternativesIntroduction": null,
  "alternatives": [
    {"letter": "A", "text": "...", "file": null, "isCorrect": false}
  ]
}
```

`discipline` assume os valores: `linguagens`, `ciencias-humanas`, `ciencias-natureza`, `matematica`.

## Limitações conhecidas

- **Disciplina de 2024 é inferida por posição**, usando o padrão observado em 2023 como referência. A ordem das disciplinas no caderno não é um bloco perfeitamente fixo de 45 questões (há pequenos desvios perto das fronteiras de bloco em praticamente todos os anos), então essa inferência carrega incerteza residual — não deve ser tratada como rótulo oficial.
- ~4,5% das questões extraídas via PDF (2025) foram sinalizadas com possível corrupção de texto/encoding durante a extração.
- `topic` (tópico) e `competency`/`skill` (competência/habilidade) ainda não estão rotulados — próxima etapa do projeto.

## Próximos passos

- [ ] Rotular tópico e competência (avaliando: inferência por faixa de posição vs. dataset da USP com ~25k questões rotuladas até 2017)
- [ ] Treinar modelo de classificação
- [ ] Empacotar como API ou dashboard

## Licença dos dados

As questões do ENEM são de domínio público (INEP/MEC). Este repositório reorganiza e consolida dados já públicos de terceiros — consulte as fontes originais para os termos específicos de cada uma.
