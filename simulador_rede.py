"""Simulador genérico de redes de filas por eventos discretos — T1.

O modelo é carregado de um arquivo YAML e pode representar qualquer topologia
composta por filas G/G/c/K, chegadas externas e roteamento probabilístico.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import math
from pathlib import Path
from typing import Any

import yaml


CHEGADA = "CHEGADA"
SAIDA = "SAIDA"


class GeradorCongruenteLinear:
    """Gerador congruente linear normalizado no intervalo [0, 1)."""

    def __init__(self, a: int, c: int, m: int, semente: int, limite: int):
        if m <= 0:
            raise ValueError("O módulo do gerador deve ser positivo.")
        if limite <= 0:
            raise ValueError("O limite de aleatórios deve ser positivo.")

        self.a = a
        self.c = c
        self.m = m
        self.ultimo_x = semente
        self.limite = limite
        self.usados = 0

    @property
    def restantes(self) -> int:
        return self.limite - self.usados

    def proximo(self) -> float:
        if self.restantes <= 0:
            raise RuntimeError("O limite de números pseudoaleatórios foi atingido.")

        self.ultimo_x = (self.a * self.ultimo_x + self.c) % self.m
        self.usados += 1

        return self.ultimo_x / self.m


def converter_intervalo(aleatorio: float, minimo: float, maximo: float) -> float:
    """Transforma U em [0,1) numa uniforme contínua entre mínimo e máximo."""
    return minimo + (maximo - minimo) * aleatorio


def carregar_modelo(caminho: str | Path) -> dict[str, Any]:
    """Carrega e valida um modelo de rede descrito em YAML."""
    with Path(caminho).open("r", encoding="utf-8") as arquivo:
        modelo = yaml.safe_load(arquivo)

    validar_modelo(modelo)
    return modelo


def _validar_intervalo(intervalo: Any, descricao: str) -> None:
    if not isinstance(intervalo, list) or len(intervalo) != 2:
        raise ValueError(f"{descricao} deve possuir exatamente dois valores.")

    minimo, maximo = intervalo
    if not isinstance(minimo, (int, float)) or not isinstance(maximo, (int, float)):
        raise ValueError(f"{descricao} deve conter valores numéricos.")
    if minimo < 0 or maximo < minimo:
        raise ValueError(f"{descricao} possui limites inválidos.")


def validar_modelo(modelo: dict[str, Any]) -> None:
    """Valida estrutura, capacidades, destinos e probabilidades do modelo."""
    if not isinstance(modelo, dict):
        raise ValueError("O modelo YAML deve ser um objeto.")

    for chave in ("gerador", "filas", "chegadas_externas"):
        if chave not in modelo:
            raise ValueError(f"Campo obrigatório ausente: {chave}.")

    gerador = modelo["gerador"]
    for chave in ("a", "c", "m", "semente", "limite_aleatorios"):
        if chave not in gerador or not isinstance(gerador[chave], int):
            raise ValueError(f"Parâmetro inteiro obrigatório do gerador: {chave}.")

    if gerador["m"] <= 0 or gerador["limite_aleatorios"] <= 0:
        raise ValueError("Módulo e limite de aleatórios devem ser positivos.")

    filas = modelo["filas"]
    if not isinstance(filas, dict) or not filas:
        raise ValueError("O modelo deve possuir ao menos uma fila.")

    for identificador, fila in filas.items():
        servidores = fila.get("servidores")
        capacidade = fila.get("capacidade")

        if not isinstance(servidores, int) or servidores <= 0:
            raise ValueError(f"{identificador}: servidores deve ser inteiro positivo.")
        if capacidade is not None:
            if not isinstance(capacidade, int) or capacidade < servidores:
                raise ValueError(
                    f"{identificador}: capacidade deve ser nula ou inteira "
                    "não inferior ao número de servidores."
                )

        _validar_intervalo(fila.get("atendimento"), f"{identificador}.atendimento")

        roteamento = fila.get("roteamento")
        if not isinstance(roteamento, list) or not roteamento:
            raise ValueError(f"{identificador}: roteamento não informado.")

        total = 0.0
        for rota in roteamento:
            destino = rota.get("destino")
            probabilidade = rota.get("probabilidade")

            if destino is not None and destino not in filas:
                raise ValueError(f"{identificador}: destino inexistente: {destino}.")
            if not isinstance(probabilidade, (int, float)) or not 0 <= probabilidade <= 1:
                raise ValueError(f"{identificador}: probabilidade inválida.")

            total += float(probabilidade)

        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{identificador}: probabilidades de roteamento devem totalizar 1."
            )

    chegadas = modelo["chegadas_externas"]
    if not isinstance(chegadas, dict) or not chegadas:
        raise ValueError("O modelo deve possuir ao menos uma chegada externa.")

    for identificador, chegada in chegadas.items():
        if identificador not in filas:
            raise ValueError(f"Chegada externa aponta para fila inexistente: {identificador}.")

        _validar_intervalo(chegada.get("intervalo"), f"{identificador}.intervalo")
        primeira = chegada.get("primeira_chegada")
        if not isinstance(primeira, (int, float)) or primeira < 0:
            raise ValueError(f"{identificador}: primeira_chegada inválida.")


def _sortear_destino(
    rotas: list[dict[str, Any]],
    gerador: GeradorCongruenteLinear,
) -> str | None:
    if len(rotas) == 1 and math.isclose(
        float(rotas[0]["probabilidade"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return rotas[0]["destino"]

    sorteio = gerador.proximo()
    acumulado = 0.0

    for rota in rotas:
        acumulado += float(rota["probabilidade"])
        if sorteio < acumulado:
            return rota["destino"]

    return rotas[-1]["destino"]


def executar_modelo(modelo_original: dict[str, Any]) -> dict[str, Any]:
    """Executa a simulação orientada a eventos do modelo informado."""
    modelo = copy.deepcopy(modelo_original)
    validar_modelo(modelo)

    configuracao_gerador = modelo["gerador"]
    gerador = GeradorCongruenteLinear(
        a=configuracao_gerador["a"],
        c=configuracao_gerador["c"],
        m=configuracao_gerador["m"],
        semente=configuracao_gerador["semente"],
        limite=configuracao_gerador["limite_aleatorios"],
    )

    filas = modelo["filas"]
    chegadas_externas = modelo["chegadas_externas"]

    clientes = {identificador: 0 for identificador in filas}
    perdas = {identificador: 0 for identificador in filas}
    tempos = {}

    for identificador, fila in filas.items():
        capacidade = fila["capacidade"]
        tempos[identificador] = [0.0] if capacidade is None else [0.0] * (capacidade + 1)

    eventos: list[tuple[float, int, int, str, str]] = []
    sequencia = 0
    tempo_global = 0.0

    for identificador, chegada in chegadas_externas.items():
        heapq.heappush(
            eventos,
            (
                float(chegada["primeira_chegada"]),
                1,
                sequencia,
                CHEGADA,
                identificador,
            ),
        )
        sequencia += 1

    def garantir_estado(identificador: str, estado: int) -> None:
        while len(tempos[identificador]) <= estado:
            tempos[identificador].append(0.0)

    def ha_vaga(identificador: str) -> bool:
        capacidade = filas[identificador]["capacidade"]
        return capacidade is None or clientes[identificador] < capacidade

    def agendar_atendimento(identificador: str) -> None:
        nonlocal sequencia

        if gerador.restantes <= 0:
            return

        minimo, maximo = filas[identificador]["atendimento"]
        duracao = converter_intervalo(gerador.proximo(), minimo, maximo)
        heapq.heappush(
            eventos,
            (tempo_global + duracao, 0, sequencia, SAIDA, identificador),
        )
        sequencia += 1

    while eventos and gerador.restantes > 0:
        tempo_evento, _, _, tipo, identificador = heapq.heappop(eventos)
        intervalo = tempo_evento - tempo_global

        for nome in filas:
            garantir_estado(nome, clientes[nome])
            tempos[nome][clientes[nome]] += intervalo

        tempo_global = tempo_evento

        if tipo == CHEGADA:
            chegada = chegadas_externas.get(identificador)

            if chegada is not None and gerador.restantes > 0:
                minimo, maximo = chegada["intervalo"]
                intervalo_chegada = converter_intervalo(
                    gerador.proximo(),
                    minimo,
                    maximo,
                )
                heapq.heappush(
                    eventos,
                    (
                        tempo_global + intervalo_chegada,
                        1,
                        sequencia,
                        CHEGADA,
                        identificador,
                    ),
                )
                sequencia += 1

            if not ha_vaga(identificador):
                perdas[identificador] += 1
                continue

            clientes[identificador] += 1
            garantir_estado(identificador, clientes[identificador])

            if clientes[identificador] <= filas[identificador]["servidores"]:
                agendar_atendimento(identificador)

            continue

        destino = _sortear_destino(filas[identificador]["roteamento"], gerador)
        clientes[identificador] -= 1

        if clientes[identificador] >= filas[identificador]["servidores"]:
            agendar_atendimento(identificador)

        if destino is None:
            continue

        if not ha_vaga(destino):
            perdas[destino] += 1
            continue

        clientes[destino] += 1
        garantir_estado(destino, clientes[destino])

        if clientes[destino] <= filas[destino]["servidores"]:
            agendar_atendimento(destino)

    resultados_filas = {}

    for identificador, fila in filas.items():
        probabilidades = [
            tempo / tempo_global if tempo_global > 0 else 0.0
            for tempo in tempos[identificador]
        ]
        resultados_filas[identificador] = {
            "nome": fila.get("nome", identificador),
            "servidores": fila["servidores"],
            "capacidade": fila["capacidade"],
            "perdas": perdas[identificador],
            "tempos_acumulados": tempos[identificador],
            "probabilidades": probabilidades,
        }

    return {
        "semente": configuracao_gerador["semente"],
        "aleatorios_usados": gerador.usados,
        "tempo_global": tempo_global,
        "resultados_filas": resultados_filas,
    }


def formatar_resultado(resultado: dict[str, Any]) -> str:
    """Formata os dados exigidos no relatório da atividade T1."""
    linhas = [
        f"Semente: {resultado['semente']}",
        f"Aleatórios utilizados: {resultado['aleatorios_usados']}",
        "",
    ]

    for dados in resultado["resultados_filas"].values():
        capacidade = dados["capacidade"]
        notacao = f"G/G/{dados['servidores']}"
        if capacidade is not None:
            notacao += f"/{capacidade}"

        linhas.append(f"{dados['nome']} - {notacao}")
        linhas.append("Distribuição dos estados:")

        for estado, (tempo, probabilidade) in enumerate(
            zip(dados["tempos_acumulados"], dados["probabilidades"])
        ):
            linhas.append(
                f"   Estado {estado}: tempo acumulado = {tempo:.6f}; "
                f"probabilidade = {probabilidade:.8f} "
                f"({probabilidade * 100:.6f}%)"
            )

        linhas.append(f"Clientes perdidos: {dados['perdas']}")
        linhas.append("")

    linhas.append(f"Tempo global da simulação: {resultado['tempo_global']:.6f}")

    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simula uma rede de filas descrita em arquivo YAML."
    )
    parser.add_argument("modelo", help="Caminho para o arquivo YAML do modelo.")
    args = parser.parse_args()

    modelo = carregar_modelo(args.modelo)
    resultado = executar_modelo(modelo)
    print(formatar_resultado(resultado))


if __name__ == "__main__":
    main()
