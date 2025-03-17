import numpy as np
import pandas as pd
from scipy.optimize import fsolve


def fnc_diseno_rejilla(Q_capt, B_rio, g, L, beta, a, d):
    """
    Añadir documentación.
    """
    # Preprocesamiento
    Q_dis = 1.1 * Q_capt
    Q = Q_dis
    Br = B_rio

    valores_beta = np.array(range(0, 27, 2))
    valores_x = np.array([1, 0.98, 0.961, 0.944, 0.927, 0.910, 0.894,
                          0.879, 0.865, 0.851, 0.837, 0.852, 0.812, 0.80])
    x = valores_x[valores_beta == beta]

    # Constantes
    u = 0.66 * (a / d) ** -0.16 * (d / a) ** 0.13
    C = 0.60 * a / d * np.cos(np.deg2rad(beta)) ** 1.5

    # Funciones para determinar 'H'
    def v(H): return Q / (Br * H)
    def Ho(H): return H + v(H) ** 2 / (2 * g)
    def h(H): return 2 / 3 * x * Ho(H)
    def b(H): return Q / (2 / 3 * C * u * L * np.sqrt(2 * g * h(H)))
    def fnc(H): return 1.72 * b(H) * H ** 1.5 - Q

    # Determinar el valor de 'H'
    H0 = fnc_valor_inicial(fnc, (0.1, 20))
    H = fsolve(fnc, H0)
    H = H[0]
    B = b(H)
    B = B - (B % 0.50) + 0.50
    B = B[0]

    # Resultados
    parametro_resultados = [
        "Calado zona de aproximación",
        "Ancho de la rejilla",
        "Longitud de la rejilla"
    ]

    simbolo_resultados = [
        "H",
        "B",
        "L"
    ]

    valor_resultados = [
        round(H, 2),
        round(B, 2),
        L
    ]

    unidad_resultados = [
        "m",
        "m",
        "m"
    ]

    resultados = pd.DataFrame({
        "Parámetro": parametro_resultados,
        "Símbolo": simbolo_resultados,
        "Valor": valor_resultados,
        "Unidad": unidad_resultados
    })

    return resultados


def fnc_valor_inicial(fnc, limites):
    """
    Añadir documentación
    """
    # Preprocesamiento
    inicio = limites[0]
    fin = limites[1]

    # Crear posibles valores y evaluarlos
    valores = np.linspace(inicio, fin + 1)
    evaluacion = fnc(valores)

    # Determinar la evaluación mínima para utilizar su correspondiente valor inicial
    indice_eval_min = np.argmin(evaluacion)
    x0 = valores[indice_eval_min]

    return x0
