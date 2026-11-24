"""Pruebas unitarias e integración para ADWIN y su visualización.

Este archivo valida de forma rigurosa y reproducible:
- Mocks de contrato, escalado robusto e inmutabilidad estricta.
- Integración real con river.drift.ADWIN (versión 0.27.0).
- Detección en señales constantes, escalón 0->1, escalón inverso 1->0.
- Detectores independientes por variable y preservación de índices originales con nulos/inf.
- Estados de evaluabilidad ('Evaluado', 'Sin referencia suficiente', 'Sin secuencia suficiente').
- Validación de parámetros de graficar_adwin (ventana > 0, min_periods, infinitos y secuencias vacías).
- Integridad académica del notebook y extracción de celdas por ID estable.
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
import river
from river import drift as river_drift

ROOT = Path(__file__).resolve().parents[2]
HYM = ROOT / 'Semana 14' / 'G2_Lab_Hym_Data_Drift.ipynb'


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


def cell(path, selector):
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(selector, str):
        matches = [c for c in data['cells'] if c.get('id') == selector]
        assert len(matches) == 1, f'ID de celda ausente o duplicado: {selector}'
        return ''.join(matches[0]['source'])
    return ''.join(data['cells'][selector]['source'])


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
    definitions(cell(HYM, '8a653bd7'), {'ejecutar_adwin'}, ns)
    definitions(cell(HYM, '522ed1e3'), {'graficar_adwin'}, ns)
    return ns


@pytest.fixture(scope='module')
def brickzon_real_river():
    """Fixture de integración directa con river.drift.ADWIN real (River 0.27.0)."""
    assert river.__version__ == '0.27.0', f"Se requiere river 0.27.0, encontrado {river.__version__}"
    ns = {
        'np': np,
        'pd': pd,
        'plt': plt,
        'tqdm': MockTqdm,
        'drift': river_drift,
    }
    definitions(cell(HYM, '8a653bd7'), {'ejecutar_adwin'}, ns)
    definitions(cell(HYM, '522ed1e3'), {'graficar_adwin'}, ns)
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


def test_notebook_json_integrity_and_academic_content():
    """Valida integridad, contenido académico y conservación de la ejecución Colab."""
    data = json.loads(HYM.read_text(encoding='utf-8'))
    assert data.get('nbformat') == 4
    assert data.get('nbformat_minor', 0) >= 2
    ids = [c.get('id') for c in data['cells']]
    assert all(isinstance(cid, str) and cid for cid in ids)
    assert len(ids) == len(set(ids))
    by_id = {c['id']: c for c in data['cells']}
    assert 'def ejecutar_adwin' in ''.join(by_id['8a653bd7']['source'])
    assert 'def graficar_adwin' in ''.join(by_id['522ed1e3']['source'])
    assert 'Dt_Customer' in ''.join(by_id['350eb169']['source'])
    assert 'Adaptive Windowing' in ''.join(by_id['350eb169']['source'])
    assert 'res_adwin' in ''.join(by_id['d8b01ad1']['source'])
    assert 'alertas_adwin' in ''.join(by_id['d8b01ad1']['source'])
    academico = '\n'.join(''.join(c['source']) for c in data['cells'] if c['cell_type'] == 'markdown')
    for concepto in ('marketing_campaign.csv', 'KS', 'Population Stability Index',
                     'chi-cuadrado', 'ADWIN', 'Experimento controlado',
                     'Reproducibilidad, validaciones y entregables'):
        assert concepto.casefold() in academico.casefold()
    assert 'no queda acreditada' in academico
    celdas_codigo = [c for c in data['cells'] if c['cell_type'] == 'code']
    assert len(celdas_codigo) == 25
    assert all(c.get('execution_count') is not None and c.get('outputs') for c in celdas_codigo)
    assert not any(o.get('output_type') == 'error' for c in celdas_codigo for o in c['outputs'])


def test_adwin_scaling_parameters_exclusive_to_reference(brickzon):
    """Verifica que el escalado (mediana e IQR/std) use parámetros estrictamente de referencia."""
    fn = brickzon['ejecutar_adwin']

    ref = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5] * 6,
        'Dt_Customer': pd.date_range('2020-01-01', periods=30),
        'V_iqr': [10.0, 20.0, 30.0, 40.0, 50.0] * 6,  # mediana=30, iqr=20
        'V_std': [10.0] * 15 + [10.0] * 15,          # constante -> escala = 1.0
    })
    # Secuencia suficiente (35 observaciones >= max(20, 30, 32))
    seq = pd.DataFrame({
        'ID': np.arange(101, 136),
        'Dt_Customer': pd.date_range('2021-01-01', periods=35),
        'V_iqr': [50.0, 70.0] + [30.0] * 33,
        'V_std': [20.0, 30.0] + [10.0] * 33,
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
    exec(compile(ast.Module(body=[ast.parse(cell(HYM, '8a653bd7')).body[0]], type_ignores=[]), '<ast>', 'exec'), ns_spy)
    ejecutar_spy = ns_spy['ejecutar_adwin']

    # Para V_iqr: mediana_ref = 30.0, q75 = 40.0, q25 = 20.0 -> iqr = 20.0
    # x1 = 50.0 -> (50 - 30) / 20 = 1.0
    # x2 = 70.0 -> (70 - 30) / 20 = 2.0
    current_col = 'V_iqr'
    ejecutar_spy(seq[['ID', 'Dt_Customer', 'V_iqr']], ref[['ID', 'Dt_Customer', 'V_iqr']], ['V_iqr'], progreso=False)
    assert np.allclose(valores_ingresados['V_iqr'][:2], [1.0, 2.0])

    # Para V_std (constante): mediana_ref = 10.0, iqr = 0, std = 0 -> escala = 1.0
    # x1 = 20.0 -> (20 - 10) / 1.0 = 10.0
    # x2 = 30.0 -> (30 - 10) / 1.0 = 20.0
    current_col = 'V_std'
    ejecutar_spy(seq[['ID', 'Dt_Customer', 'V_std']], ref[['ID', 'Dt_Customer', 'V_std']], ['V_std'], progreso=False)
    assert np.allclose(valores_ingresados['V_std'][:2], [10.0, 20.0])


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
    """B04: Verifica secuencias vacías, completamente inválidas y menores que el mínimo establecido."""
    ref, seq = crear_datos_sinteticos(n_ref=40, n_seq=50)
    fn = brickzon['ejecutar_adwin']

    # Sin columnas
    res_vacio, ev_vacio = fn(seq, ref, [], progreso=False)
    assert len(res_vacio) == 0 and len(ev_vacio) == 0

    # 1. Secuencia vacía (len == 0): devuelve 'Sin secuencia suficiente', 0 omitidas, 0 procesadas
    seq_vacia = seq.iloc[:0].copy()
    res_seq0, ev_seq0 = fn(seq_vacia, ref, ['MntWines'], progreso=False)
    assert len(res_seq0) == 1
    assert res_seq0.iloc[0]['procesadas'] == 0
    assert res_seq0.iloc[0]['omitidas'] == 0
    assert res_seq0.iloc[0]['alertas'] == 0
    assert res_seq0.iloc[0]['estado'] == 'Sin secuencia suficiente'
    assert len(ev_seq0) == 0

    # 2. Secuencia completamente inválida (todas NaN / inf): devuelve 'Sin secuencia suficiente'
    seq_invalida = seq.copy()
    seq_invalida['MntWines'] = np.nan
    seq_invalida.loc[0, 'MntWines'] = np.inf
    seq_invalida.loc[1, 'MntWines'] = -np.inf
    res_inv, ev_inv = fn(seq_invalida, ref, ['MntWines'], progreso=False)
    assert len(res_inv) == 1
    assert res_inv.iloc[0]['procesadas'] == 0
    assert res_inv.iloc[0]['omitidas'] == len(seq_invalida)
    assert res_inv.iloc[0]['alertas'] == 0
    assert res_inv.iloc[0]['estado'] == 'Sin secuencia suficiente'
    assert len(ev_inv) == 0

    # 3. Secuencia menor que el mínimo establecido (n_finitas < max(20, grace_period, clock))
    # Con clock=32, grace_period=30 -> mínimo es 32 observaciones
    seq_menor = seq.iloc[:15].copy()
    res_menor, ev_menor = fn(seq_menor, ref, ['MntWines'], delta=0.002, clock=32, grace_period=30, progreso=False)
    assert len(res_menor) == 1
    assert res_menor.iloc[0]['procesadas'] == 0
    assert res_menor.iloc[0]['omitidas'] == 15
    assert res_menor.iloc[0]['alertas'] == 0
    assert res_menor.iloc[0]['estado'] == 'Sin secuencia suficiente'
    assert len(ev_menor) == 0


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


def test_graficar_adwin_validations_and_edge_cases(brickzon, monkeypatch):
    """B05: Verifica validación de ventana (>0 int), min_periods, infinitos y secuencias vacías."""
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    fn = brickzon['graficar_adwin']
    ref, seq = crear_datos_sinteticos(n_ref=50, n_seq=60)
    eventos = pd.DataFrame(columns=['variable', 'indice_0based', 'numero_observacion',
                                    'ID', 'fecha', 'valor_original', 'ancho_ventana'])

    # Validación de parámetro ventana: debe ser entero positivo
    for v_invalida in [0, -1, -80, '80', 20.5, None]:
        with pytest.raises(ValueError):
            fn(seq, eventos, ['MntWines'], corte=25, ventana=v_invalida)

    # Ventanas de tamaño 1, 2 y 4 (calcula min_periods sin superar ventana)
    for v in [1, 2, 4]:
        fn(seq, eventos, ['MntWines'], corte=25, ventana=v)

    # Secuencia con valores infinitos (+inf, -inf) y NaN
    seq_inf = seq.copy()
    seq_inf.loc[5, 'MntWines'] = np.inf
    seq_inf.loc[15, 'MntWines'] = -np.inf
    seq_inf.loc[25, 'MntWines'] = np.nan
    fn(seq_inf, eventos, ['MntWines'], corte=25, ventana=4)

    # Secuencia vacía
    seq_vacia = seq.iloc[:0].copy()
    fn(seq_vacia, eventos, ['MntWines'], corte=0, ventana=4)
    plt.close('all')


# ==============================================================================
# B02: PRUEBAS DE INTEGRACIÓN DIRECTA CON RIVER 0.27.0 (river.drift.ADWIN)
# ==============================================================================

def test_adwin_real_river_constant_signal_zero_alerts(brickzon_real_river):
    """B02: Señal constante sin cambios con river.drift.ADWIN real no produce alertas."""
    fn = brickzon_real_river['ejecutar_adwin']
    n = 1024
    ref = pd.DataFrame({
        'ID': range(100),
        'Dt_Customer': pd.date_range('2020-01-01', periods=100),
        'constante': [0.5] * 100,
    })
    seq = pd.DataFrame({
        'ID': range(1000, 1000 + n),
        'Dt_Customer': pd.date_range('2021-01-01', periods=n),
        'constante': [0.5] * n,
    })

    res, eventos = fn(seq, ref, ['constante'], delta=0.002, clock=32, grace_period=30, progreso=False)
    row = res.iloc[0]
    assert row['alertas'] == 0, f"Una señal constante no debe generar alertas, obtuvo {row['alertas']}"
    assert row['procesadas'] == n
    assert row['omitidas'] == 0
    assert row['estado'] == 'Evaluado'
    assert len(eventos) == 0


def test_adwin_real_river_step_signals_detection(brickzon_real_river):
    """B02: Detección con river.drift.ADWIN real en señal 0->1 y señal inversa 1->0."""
    fn = brickzon_real_river['ejecutar_adwin']
    n = 1024
    punto_intervencion = 512

    # 1. Señal con 512 observaciones de 0 seguidas de 512 observaciones de 1
    ref_0 = pd.DataFrame({
        'ID': range(100),
        'Dt_Customer': pd.date_range('2020-01-01', periods=100),
        'step_0_1': [0.0] * 100,
    })
    seq_0_1 = pd.DataFrame({
        'ID': range(1000, 1000 + n),
        'Dt_Customer': pd.date_range('2021-01-01', periods=n),
        'step_0_1': [0.0] * punto_intervencion + [1.0] * (n - punto_intervencion),
    })

    res_0_1, ev_0_1 = fn(seq_0_1, ref_0, ['step_0_1'], delta=0.002, clock=32, grace_period=30, progreso=False)
    assert res_0_1.iloc[0]['alertas'] >= 1, "Debe detectar al menos una alerta en señal 0->1"
    assert res_0_1.iloc[0]['procesadas'] == n
    assert res_0_1.iloc[0]['omitidas'] == 0
    assert res_0_1.iloc[0]['estado'] == 'Evaluado'
    for _, ev in ev_0_1.iterrows():
        assert ev['indice_0based'] >= punto_intervencion, "Alerta debe ser posterior al punto de intervención (>= 512)"
        assert ev['numero_observacion'] == ev['indice_0based'] + 1
        assert ev['ancho_ventana'] > 0

    # 2. Señal inversa: 512 observaciones de 1 seguidas de 512 observaciones de 0
    ref_1 = pd.DataFrame({
        'ID': range(100),
        'Dt_Customer': pd.date_range('2020-01-01', periods=100),
        'step_1_0': [1.0] * 100,
    })
    seq_1_0 = pd.DataFrame({
        'ID': range(1000, 1000 + n),
        'Dt_Customer': pd.date_range('2021-01-01', periods=n),
        'step_1_0': [1.0] * punto_intervencion + [0.0] * (n - punto_intervencion),
    })

    res_1_0, ev_1_0 = fn(seq_1_0, ref_1, ['step_1_0'], delta=0.002, clock=32, grace_period=30, progreso=False)
    assert res_1_0.iloc[0]['alertas'] >= 1, "Debe detectar al menos una alerta en señal inversa 1->0"
    assert res_1_0.iloc[0]['procesadas'] == n
    assert res_1_0.iloc[0]['omitidas'] == 0
    assert res_1_0.iloc[0]['estado'] == 'Evaluado'
    for _, ev in ev_1_0.iterrows():
        assert ev['indice_0based'] >= punto_intervencion, "Alerta debe ser posterior al punto de intervención (>= 512)"
        assert ev['numero_observacion'] == ev['indice_0based'] + 1
        assert ev['ancho_ventana'] > 0


def test_adwin_real_river_independent_detectors(brickzon_real_river):
    """B02: river.drift.ADWIN instancia detectores independientes por variable en una misma ejecución."""
    fn = brickzon_real_river['ejecutar_adwin']
    n = 1024
    ref = pd.DataFrame({
        'ID': range(100),
        'Dt_Customer': pd.date_range('2020-01-01', periods=100),
        'constante': [0.0] * 100,
        'step_0_1': [0.0] * 100,
        'step_1_0': [1.0] * 100,
    })
    seq = pd.DataFrame({
        'ID': range(1000, 1000 + n),
        'Dt_Customer': pd.date_range('2021-01-01', periods=n),
        'constante': [0.0] * n,
        'step_0_1': [0.0] * 512 + [1.0] * 512,
        'step_1_0': [1.0] * 512 + [0.0] * 512,
    })

    res, eventos = fn(seq, ref, ['constante', 'step_0_1', 'step_1_0'],
                      delta=0.002, clock=32, grace_period=30, progreso=False)

    assert set(res['variable']) == {'constante', 'step_0_1', 'step_1_0'}
    res_const = res.loc[res['variable'].eq('constante')].iloc[0]
    res_0_1 = res.loc[res['variable'].eq('step_0_1')].iloc[0]
    res_1_0 = res.loc[res['variable'].eq('step_1_0')].iloc[0]

    assert res_const['alertas'] == 0, "Detector constante no debe verse contaminado por otras variables"
    assert res_0_1['alertas'] >= 1
    assert res_1_0['alertas'] >= 1
    assert 'constante' not in set(eventos['variable'])


def test_adwin_real_river_missing_values_preserves_indices(brickzon_real_river):
    """B02: river.drift.ADWIN maneja valores faltantes e infinitos preservando índices originales."""
    fn = brickzon_real_river['ejecutar_adwin']
    n = 1024
    punto_intervencion = 512
    ref = pd.DataFrame({
        'ID': range(100),
        'Dt_Customer': pd.date_range('2020-01-01', periods=100),
        'senal': [0.0] * 100,
    })
    valores = [0.0] * punto_intervencion + [1.0] * (n - punto_intervencion)
    seq = pd.DataFrame({
        'ID': range(2000, 2000 + n),
        'Dt_Customer': pd.date_range('2021-01-01', periods=n),
        'senal': valores,
    })

    # Inyectar valores no finitos (NaN, inf, -inf)
    posiciones_nulas = [10, 50, 100, 200, 600, 700]
    seq.loc[10, 'senal'] = np.nan
    seq.loc[50, 'senal'] = np.inf
    seq.loc[100, 'senal'] = -np.inf
    seq.loc[200, 'senal'] = np.nan
    seq.loc[600, 'senal'] = np.nan
    seq.loc[700, 'senal'] = np.inf

    res, eventos = fn(seq, ref, ['senal'], delta=0.002, clock=32, grace_period=30, progreso=False)
    row = res.iloc[0]
    assert row['omitidas'] == len(posiciones_nulas)
    assert row['procesadas'] == n - len(posiciones_nulas)
    assert row['alertas'] >= 1
    assert row['estado'] == 'Evaluado'

    for _, ev in eventos.iterrows():
        assert ev['indice_0based'] not in posiciones_nulas
        assert ev['indice_0based'] >= punto_intervencion
        assert ev['numero_observacion'] == ev['indice_0based'] + 1
        assert ev['ID'] == seq.loc[ev['indice_0based'], 'ID']
        assert ev['fecha'] == seq.loc[ev['indice_0based'], 'Dt_Customer']
        assert ev['valor_original'] == seq.loc[ev['indice_0based'], 'senal']
