"""Pruebas unitarias sintéticas para las secciones de Brickzon (ADWIN y visualización).

Este archivo valida de forma 100% aislada e independiente los contratos de:
- ejecutar_adwin (Sección 10.1, celda 75)
- graficar_adwin (Sección 10.2, celda 78)
- Metadatos e integridad de la interpretación de Brickzon (celdas 81 y 116)
No requiere conexión a Internet, datos externos ni ejecución completa de notebooks.
"""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
HYM = ROOT / 'Semana 14' / 'Plantilla_HyM_Data_Drift_Colab_Agent_Oriented.ipynb'


class MockADWIN:
    """Mock/emulador que reproduce estrictamente el contrato de river.drift.ADWIN.

    Contrato:
      - __init__(delta=0.002, clock=32, grace_period=30)
      - update(val: float) -> None
      - drift_detected: bool
      - width: int
    """
    def __init__(self, delta=0.002, clock=32, grace_period=30):
        self.delta = float(delta)
        self.clock = int(clock)
        self.grace_period = int(grace_period)
        self._window = []
        self.drift_detected = False
        self.width = 0
        self._count = 0

    def update(self, val):
        self._window.append(float(val))
        self.width = len(self._window)
        self._count += 1
        self.drift_detected = False

        if self._count < self.grace_period:
            return

        # Evaluación periódica o por volumen acumulado
        if self._count % max(1, self.clock) == 0 or len(self._window) >= 80:
            n = len(self._window)
            if n >= 20:
                mid = n // 2
                w0 = self._window[:mid]
                w1 = self._window[mid:]
                m0 = sum(w0) / len(w0)
                m1 = sum(w1) / len(w1)
                # Detección de drift cuando el desplazamiento estandarizado supera el umbral
                if abs(m0 - m1) > 1.2 and len(w1) >= 8:
                    self.drift_detected = True
                    self._window = list(w1)
                    self.width = len(self._window)


class MockDrift:
    ADWIN = MockADWIN


class MockTqdm:
    """Mock de tqdm que soporta wrapping de iterables y barras manuales."""
    def __init__(self, iterable=None, total=None, desc=None, unit=None,
                 mininterval=None, disable=False, **kwargs):
        self.iterable = iterable
        self.total = total
        self.disable = disable
        self.n = 0
        self.postfix = ''

    def __iter__(self):
        if self.iterable is not None:
            return iter(self.iterable)
        return iter(())

    def update(self, n=1):
        self.n += n

    def set_postfix_str(self, s, refresh=False):
        self.postfix = s

    def close(self):
        pass


def cell(path, idx):
    data = json.loads(path.read_text(encoding='utf-8'))
    return ''.join(data['cells'][idx]['source'])


def definitions(source, names, namespace):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert set(n.name for n in nodes) == set(names), f"Faltan definiciones: {set(names) - set(n.name for n in nodes)}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<brickzon notebook functions>', 'exec'), namespace)
    return namespace


@pytest.fixture(scope='module')
def brickzon():
    ns = {
        'np': np,
        'pd': pd,
        'plt': plt,
        'tqdm': MockTqdm,
        'drift': MockDrift,
    }
    definitions(cell(HYM, 75), {'ejecutar_adwin'}, ns)
    definitions(cell(HYM, 78), {'graficar_adwin'}, ns)
    return ns


def crear_datos_sinteticos(n_ref=100, n_seq=200, drift_en_pos=None, semilla=42):
    rng = np.random.default_rng(semilla)
    ref = pd.DataFrame({
        'ID': np.arange(1000, 1000 + n_ref),
        'Dt_Customer': pd.date_range('2020-01-01', periods=n_ref, freq='D'),
        'MntWines': rng.normal(50.0, 10.0, n_ref),
        'Income': rng.normal(50000.0, 5000.0, n_ref),
        'MntMeatProducts': rng.normal(80.0, 15.0, n_ref),
        'NumWebPurchases': rng.integers(1, 10, n_ref),
    })

    if drift_en_pos is None or drift_en_pos >= n_seq:
        wines = rng.normal(50.0, 10.0, n_seq)
    else:
        wines_estacionario = rng.normal(50.0, 10.0, drift_en_pos)
        wines_alterado = rng.normal(160.0, 10.0, n_seq - drift_en_pos)
        wines = np.concatenate([wines_estacionario, wines_alterado])

    seq = pd.DataFrame({
        'ID': np.arange(2000, 2000 + n_seq),
        'Dt_Customer': pd.date_range('2020-05-01', periods=n_seq, freq='D'),
        'MntWines': wines,
        'Income': rng.normal(50000.0, 5000.0, n_seq),
        'MntMeatProducts': rng.normal(80.0, 15.0, n_seq),
        'NumWebPurchases': rng.integers(1, 10, n_seq),
    })
    return ref, seq


def test_notebook_json_integrity_and_brickzon_metadata():
    """Valida formato v4, celdas y etiquetas owner-brickzon en el notebook."""
    data = json.loads(HYM.read_text(encoding='utf-8'))
    assert data.get('nbformat') == 4, "El notebook debe ser nbformat v4"
    assert data.get('nbformat_minor', 0) >= 2

    # Verificar presencia y etiquetas de celdas clave
    c75 = data['cells'][75]
    c78 = data['cells'][78]
    c81 = data['cells'][81]
    c116 = data['cells'][116]

    assert 'owner-brickzon' in c75.get('metadata', {}).get('tags', [])
    assert 'owner-brickzon' in c78.get('metadata', {}).get('tags', [])
    assert 'def ejecutar_adwin' in ''.join(c75['source'])
    assert 'def graficar_adwin' in ''.join(c78['source'])

    # Verificar interpretación de Brickzon en celda 81
    txt_81 = ''.join(c81['source'])
    assert 'Espacio de interpretación de Brickzon:' not in txt_81, "El placeholder debe haber sido reemplazado"
    assert 'ADWIN (Adaptive Windowing)' in txt_81
    assert 'Dt_Customer' in txt_81
    assert 'cohortes de incorporación' in txt_81
    assert 'Sección 11' in txt_81
    assert 'retardo de detección' in txt_81.lower() or 'detection latency' in txt_81

    # Verificar checklist de entrega en celda 116
    txt_116 = ''.join(c116['source'])
    assert '- [x] **Brickzon:** ADWIN procesado en orden establecido; eventos/configuración trazables; límites temporales explicados.' in txt_116


def test_adwin_scaling_parameters_exclusive_to_reference(brickzon):
    """Verifica que el escalado (mediana e IQR/std) use parámetros estrictamente de referencia."""
    fn = brickzon['ejecutar_adwin']

    ref = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5] * 6,
        'Dt_Customer': pd.date_range('2020-01-01', periods=30),
        'V_iqr': [10.0, 20.0, 30.0, 40.0, 50.0] * 6,  # mediana=30, iqr=20
        'V_std': [10.0] * 15 + [10.0] * 15,          # constante -> escala = 1.0
    })
    seq = pd.DataFrame({
        'ID': [101, 102],
        'Dt_Customer': pd.date_range('2021-01-01', periods=2),
        'V_iqr': [50.0, 70.0],
        'V_std': [20.0, 30.0],
    })

    # Espiar llamadas a update para verificar los valores escalados exactos
    valores_ingresados = {'V_iqr': [], 'V_std': []}

    class SpyADWIN:
        def __init__(self, *args, **kwargs):
            self.drift_detected = False
            self.width = 0
        def update(self, val):
            if current_col in valores_ingresados:
                valores_ingresados[current_col].append(val)

    current_col = 'V_iqr'
    spy_drift = type('SpyDrift', (), {'ADWIN': SpyADWIN})

    ns_spy = dict(brickzon)
    ns_spy['drift'] = spy_drift
    exec(compile(ast.Module(body=[ast.parse(cell(HYM, 75)).body[0]], type_ignores=[]), '<ast>', 'exec'), ns_spy)
    ejecutar_spy = ns_spy['ejecutar_adwin']

    # Para V_iqr: mediana_ref = 30.0, q75 = 40.0, q25 = 20.0 -> iqr = 20.0
    # x1 = 50.0 -> (50 - 30) / 20 = 1.0
    # x2 = 70.0 -> (70 - 30) / 20 = 2.0
    current_col = 'V_iqr'
    ejecutar_spy(seq[['ID', 'Dt_Customer', 'V_iqr']], ref[['ID', 'Dt_Customer', 'V_iqr']], ['V_iqr'], progreso=False)
    assert np.allclose(valores_ingresados['V_iqr'], [1.0, 2.0])

    # Para V_std (constante): mediana_ref = 10.0, iqr = 0, std = 0 -> escala = 1.0
    # x1 = 20.0 -> (20 - 10) / 1.0 = 10.0
    # x2 = 30.0 -> (30 - 10) / 1.0 = 20.0
    current_col = 'V_std'
    ejecutar_spy(seq[['ID', 'Dt_Customer', 'V_std']], ref[['ID', 'Dt_Customer', 'V_std']], ['V_std'], progreso=False)
    assert np.allclose(valores_ingresados['V_std'], [10.0, 20.0])


def test_adwin_detects_synthetic_drift_and_event_contract(brickzon):
    """Prueba que ejecutar_adwin detecta drift inducido y respeta el esquema de eventos."""
    ref, seq = crear_datos_sinteticos(n_ref=100, n_seq=250, drift_en_pos=120)
    fn = brickzon['ejecutar_adwin']

    res, eventos = fn(seq, ref, ['MntWines', 'Income'], clock=16, grace_period=20, progreso=False)

    # Esquema de resumen
    columnas_resumen = ['variable', 'alertas', 'procesadas', 'omitidas', 'estado']
    assert list(res.columns) == columnas_resumen
    assert set(res['variable']) == {'MntWines', 'Income'}

    res_wines = res.loc[res['variable'].eq('MntWines')].iloc[0]
    res_income = res.loc[res['variable'].eq('Income')].iloc[0]

    assert res_wines['alertas'] > 0, "MntWines debe registrar al menos una alerta de drift"
    assert res_wines['procesadas'] == len(seq)
    assert res_wines['omitidas'] == 0
    assert res_wines['estado'] == 'Evaluado'

    assert res_income['alertas'] == 0, "Income (estacionario) no debe emitir alertas"
    assert res_income['procesadas'] == len(seq)
    assert res_income['omitidas'] == 0

    # Esquema de eventos
    columnas_eventos = ['variable', 'indice_0based', 'numero_observacion',
                        'ID', 'fecha', 'valor_original', 'ancho_ventana']
    assert list(eventos.columns) == columnas_eventos
    assert len(eventos) == res_wines['alertas']

    for _, ev in eventos.iterrows():
        assert ev['variable'] == 'MntWines'
        assert ev['numero_observacion'] == ev['indice_0based'] + 1
        assert ev['indice_0based'] >= 120, "Las alertas deben ser posteriores al punto de drift (>= 120)"
        assert ev['valor_original'] > 100.0, "El valor original registrado debe reflejar el régimen alterado"
        assert ev['ancho_ventana'] > 0
        assert isinstance(ev['fecha'], pd.Timestamp)


def test_adwin_strict_immutability(brickzon):
    """Garantiza que ejecutar_adwin no mute en absoluto DataFrames de secuencia ni referencia."""
    ref, seq = crear_datos_sinteticos(n_ref=50, n_seq=60)
    ref_copia = ref.copy(deep=True)
    seq_copia = seq.copy(deep=True)

    fn = brickzon['ejecutar_adwin']
    fn(seq, ref, ['MntWines', 'Income', 'MntMeatProducts'], progreso=False)

    pd.testing.assert_frame_equal(ref, ref_copia)
    pd.testing.assert_frame_equal(seq, seq_copia)


def test_adwin_missing_and_infinite_values_handling(brickzon):
    """Garantiza que NaNs, inf y -inf en secuencia y referencia se manejen sin excepciones."""
    ref, seq = crear_datos_sinteticos(n_ref=50, n_seq=60)

    # Inyectar infinitos y nulos en referencia
    ref.loc[0, 'MntWines'] = np.inf
    ref.loc[1, 'MntWines'] = -np.inf
    ref.loc[2, 'MntWines'] = np.nan

    # Inyectar infinitos y nulos en secuencia
    seq.loc[5, 'MntWines'] = np.nan
    seq.loc[10, 'MntWines'] = np.inf
    seq.loc[15, 'MntWines'] = -np.inf

    fn = brickzon['ejecutar_adwin']
    res, eventos = fn(seq, ref, ['MntWines'], clock=16, grace_period=10, progreso=False)

    res_wines = res.loc[res['variable'].eq('MntWines')].iloc[0]
    assert res_wines['omitidas'] == 3, "Debe omitir exactamente 3 valores no finitos"
    assert res_wines['procesadas'] == len(seq) - 3
    assert res_wines['estado'] == 'Evaluado'


def test_adwin_insufficient_reference_handling(brickzon):
    """Garantiza manejo adecuado cuando la cohorte de referencia tiene menos de 20 observaciones válidas."""
    ref, seq = crear_datos_sinteticos(n_ref=15, n_seq=50)
    fn = brickzon['ejecutar_adwin']

    res, eventos = fn(seq, ref, ['MntWines', 'Income'], progreso=False)

    for _, row in res.iterrows():
        assert row['estado'] == 'Sin referencia suficiente'
        assert row['alertas'] == 0
        assert row['procesadas'] == 0
        assert row['omitidas'] == len(seq)

    assert len(eventos) == 0
    assert list(eventos.columns) == ['variable', 'indice_0based', 'numero_observacion',
                                     'ID', 'fecha', 'valor_original', 'ancho_ventana']


def test_adwin_empty_or_edge_inputs(brickzon):
    """Verifica respuesta adecuada ante listas de columnas vacías o secuencias vacías."""
    ref, seq = crear_datos_sinteticos(n_ref=40, n_seq=50)
    fn = brickzon['ejecutar_adwin']

    # Sin columnas
    res_vacio, ev_vacio = fn(seq, ref, [], progreso=False)
    assert len(res_vacio) == 0 and len(ev_vacio) == 0

    # Secuencia vacía
    seq_vacia = seq.iloc[:0].copy()
    res_seq0, ev_seq0 = fn(seq_vacia, ref, ['MntWines'], progreso=False)
    assert len(res_seq0) == 1
    assert res_seq0.iloc[0]['procesadas'] == 0
    assert len(ev_seq0) == 0


def test_graficar_adwin_non_interactive_agg(brickzon, monkeypatch):
    """Verifica que graficar_adwin se ejecute sin fallos bajo backend Agg no interactivo."""
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    fn = brickzon['graficar_adwin']

    ref, seq = crear_datos_sinteticos(n_ref=50, n_seq=100)
    eventos = pd.DataFrame({
        'variable': ['MntWines', 'MntWines', 'Income'],
        'indice_0based': [40, 70, 85],
        'numero_observacion': [41, 71, 86],
        'ID': [2040, 2070, 2085],
        'fecha': [pd.Timestamp('2020-06-01')] * 3,
        'valor_original': [150.0, 160.0, 75000.0],
        'ancho_ventana': [32, 45, 50],
    })

    # Caso normal con eventos
    fn(seq, eventos, ['MntWines', 'Income'], corte=50, ventana=20)

    # Caso sin eventos
    eventos_vacios = eventos.iloc[:0]
    fn(seq, eventos_vacios, ['MntMeatProducts'], corte=50, ventana=20)

    # Caso secuencia corta donde len(seq) < ventana
    seq_corta = seq.iloc[:10]
    fn(seq_corta, eventos_vacios, ['MntWines'], corte=5, ventana=50)
    plt.close('all')
