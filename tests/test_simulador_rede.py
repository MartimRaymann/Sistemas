import math
import tempfile
import unittest
from pathlib import Path

from simulador_rede import (
    GeradorCongruenteLinear,
    carregar_modelo,
    executar_modelo,
    formatar_resultado,
    validar_modelo,
)


MODELO_T1 = {
    "gerador": {
        "a": 1_664_525,
        "c": 1_013_904_223,
        "m": 2**32,
        "semente": 42,
        "limite_aleatorios": 100_000,
    },
    "filas": {
        "fila_1": {
            "nome": "Fila 1",
            "servidores": 1,
            "capacidade": None,
            "atendimento": [1.0, 2.0],
            "roteamento": [
                {"destino": "fila_2", "probabilidade": 0.2},
                {"destino": "fila_3", "probabilidade": 0.8},
            ],
        },
        "fila_2": {
            "nome": "Fila 2",
            "servidores": 2,
            "capacidade": 5,
            "atendimento": [4.0, 6.0],
            "roteamento": [
                {"destino": "fila_1", "probabilidade": 0.3},
                {"destino": "fila_3", "probabilidade": 0.5},
                {"destino": None, "probabilidade": 0.2},
            ],
        },
        "fila_3": {
            "nome": "Fila 3",
            "servidores": 2,
            "capacidade": 10,
            "atendimento": [5.0, 15.0],
            "roteamento": [
                {"destino": "fila_2", "probabilidade": 0.7},
                {"destino": None, "probabilidade": 0.3},
            ],
        },
    },
    "chegadas_externas": {
        "fila_1": {
            "intervalo": [2.0, 4.0],
            "primeira_chegada": 2.0,
        }
    },
}


class TestGeradorCongruenteLinear(unittest.TestCase):
    def test_gera_sequencia_reprodutivel_e_respeita_limite(self):
        gerador = GeradorCongruenteLinear(
            a=1_664_525,
            c=1_013_904_223,
            m=2**32,
            semente=42,
            limite=3,
        )

        valores = [gerador.proximo() for _ in range(3)]

        self.assertEqual(gerador.usados, 3)
        self.assertTrue(all(0.0 <= valor < 1.0 for valor in valores))
        self.assertEqual(
            valores,
            [
                0.2523451747838408,
                0.08812504541128874,
                0.5772811982315034,
            ],
        )

        with self.assertRaises(RuntimeError):
            gerador.proximo()


class TestModelo(unittest.TestCase):
    def test_valida_topologia_do_t1(self):
        validar_modelo(MODELO_T1)

    def test_rejeita_probabilidades_que_nao_totalizam_um(self):
        modelo = {
            **MODELO_T1,
            "filas": {
                **MODELO_T1["filas"],
                "fila_1": {
                    **MODELO_T1["filas"]["fila_1"],
                    "roteamento": [
                        {"destino": "fila_2", "probabilidade": 0.2},
                        {"destino": "fila_3", "probabilidade": 0.7},
                    ],
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "totalizar 1"):
            validar_modelo(modelo)

    def test_carrega_yaml_do_modelo(self):
        conteudo = """
gerador:
  a: 1664525
  c: 1013904223
  m: 4294967296
  semente: 42
  limite_aleatorios: 10
filas:
  fila_1:
    nome: Fila 1
    servidores: 1
    capacidade: null
    atendimento: [1.0, 2.0]
    roteamento:
      - destino: null
        probabilidade: 1.0
chegadas_externas:
  fila_1:
    intervalo: [2.0, 4.0]
    primeira_chegada: 2.0
"""
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "modelo.yml"
            caminho.write_text(conteudo, encoding="utf-8")

            modelo = carregar_modelo(caminho)

        self.assertIsNone(modelo["filas"]["fila_1"]["capacidade"])
        self.assertEqual(modelo["gerador"]["limite_aleatorios"], 10)


class TestSimulacaoT1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resultado = executar_modelo(MODELO_T1)

    def test_consumiu_exatamente_cem_mil_aleatorios(self):
        self.assertEqual(self.resultado["aleatorios_usados"], 100_000)

    def test_primeira_chegada_ocorre_em_dois_com_filas_inicialmente_vazias(self):
        modelo = {
            "gerador": {
                "a": 1_664_525,
                "c": 1_013_904_223,
                "m": 2**32,
                "semente": 42,
                "limite_aleatorios": 1,
            },
            "filas": {
                "fila": {
                    "nome": "Fila",
                    "servidores": 1,
                    "capacidade": None,
                    "atendimento": [1.0, 1.0],
                    "roteamento": [
                        {"destino": None, "probabilidade": 1.0},
                    ],
                },
            },
            "chegadas_externas": {
                "fila": {
                    "intervalo": [2.0, 2.0],
                    "primeira_chegada": 2.0,
                },
            },
        }

        resultado = executar_modelo(modelo)
        fila = resultado["resultados_filas"]["fila"]

        self.assertEqual(resultado["tempo_global"], 2.0)
        self.assertEqual(fila["tempos_acumulados"][0], 2.0)
        self.assertEqual(fila["tempos_acumulados"][1], 0.0)

    def test_saida_tem_prioridade_sobre_chegada_simultanea(self):
        modelo = {
            "gerador": {
                "a": 1,
                "c": 0,
                "m": 2,
                "semente": 1,
                "limite_aleatorios": 4,
            },
            "filas": {
                "fila": {
                    "nome": "Fila",
                    "servidores": 1,
                    "capacidade": 1,
                    "atendimento": [1.0, 1.0],
                    "roteamento": [
                        {"destino": None, "probabilidade": 1.0},
                    ],
                },
            },
            "chegadas_externas": {
                "fila": {
                    "intervalo": [1.0, 1.0],
                    "primeira_chegada": 0.0,
                },
            },
        }

        resultado = executar_modelo(modelo)

        self.assertEqual(resultado["tempo_global"], 1.0)
        self.assertEqual(resultado["resultados_filas"]["fila"]["perdas"], 0)

    def test_resultados_reprodutiveis_confirmam_roteamento_e_perdas(self):
        perdas = {
            identificador: dados["perdas"]
            for identificador, dados in self.resultado["resultados_filas"].items()
        }

        self.assertEqual(perdas, {"fila_1": 0, "fila_2": 7, "fila_3": 11_691})
        self.assertAlmostEqual(self.resultado["tempo_global"], 50_936.92058576364)

    def test_tempos_acumulados_fecham_com_tempo_global(self):
        tempo_global = self.resultado["tempo_global"]

        for dados in self.resultado["resultados_filas"].values():
            self.assertTrue(
                math.isclose(
                    sum(dados["tempos_acumulados"]),
                    tempo_global,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            )

    def test_probabilidades_de_cada_fila_totalizam_um(self):
        for dados in self.resultado["resultados_filas"].values():
            self.assertTrue(
                math.isclose(
                    sum(dados["probabilidades"]),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
            )

    def test_capacidades_finitas_limitam_os_estados(self):
        fila_2 = self.resultado["resultados_filas"]["fila_2"]
        fila_3 = self.resultado["resultados_filas"]["fila_3"]

        self.assertEqual(len(fila_2["tempos_acumulados"]), 6)
        self.assertEqual(len(fila_3["tempos_acumulados"]), 11)
        self.assertGreaterEqual(fila_2["perdas"], 0)
        self.assertGreaterEqual(fila_3["perdas"], 0)

    def test_resultado_formatado_contem_todas_as_exigencias(self):
        texto = formatar_resultado(self.resultado)

        self.assertIn("Aleatórios utilizados: 100000", texto)
        self.assertIn("Fila 1 - G/G/1", texto)
        self.assertIn("Fila 2 - G/G/2/5", texto)
        self.assertIn("Fila 3 - G/G/2/10", texto)
        self.assertIn("Clientes perdidos:", texto)
        self.assertIn("Tempo global da simulação:", texto)


if __name__ == "__main__":
    unittest.main()
