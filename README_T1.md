# Simulador de Rede de Filas — T1

## Requisitos

- Python 3.11 ou superior
- PyYAML (`python -m pip install -r requirements.txt`)

## Executar o modelo de validação

No terminal, dentro desta pasta:

```bash
python simulador_rede.py modelo_t1.yml
```

Para salvar a saída em arquivo:

```bash
python simulador_rede.py modelo_t1.yml > resultados_t1.txt
```

## Executar os testes

```bash
python -m unittest tests.test_simulador_rede -v
```

Essa suíte possui 12 testes. Para executar todos os testes existentes na pasta:

```bash
python -m unittest discover -s tests -v
```

## Formato do YAML

O arquivo contém:

- parâmetros do gerador congruente linear;
- filas, servidores, capacidades e intervalos de atendimento;
- probabilidades e destinos de roteamento;
- chegadas externas e o instante da primeira chegada.

Use `null` em `capacidade` para representar uma fila com capacidade infinita. O destino `null` no roteamento representa a saída do cliente da rede.

## Critério de encerramento

A simulação encerra quando o 100.000º número pseudoaleatório é utilizado. Os tempos acumulados de cada fila são atualizados sempre que o relógio global avança.
