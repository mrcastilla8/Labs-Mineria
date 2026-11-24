"""Pruebas sintéticas de las secciones 9 y 11 de HyM extraídas del notebook."""

import ast
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import river
from river import drift
from scipy import stats
from scipy.stats import chi2_contingency
from tqdm.auto import tqdm


NOTEBOOK = Path(__file__).resolve().parents[1] / 'G2_Lab_Hym_Data_Drift.ipynb'
OBJETIVOS = ['Income', 'MntWines', 'Marital_Status']


@pytest.fixture(scope='module')
def celdas_python():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return {
        celda['id']: ast.parse(''.join(celda['source']), filename=f'{NOTEBOOK.name}:{celda["id"]}')
        for celda in notebook['cells'] if celda['cell_type'] == 'code'
    }


def extraer_definiciones(celda, nombres, entorno):
    nodos = {}
    for nodo in celda.body:
        if isinstance(nodo, ast.FunctionDef):
            nodos[nodo.name] = nodo
        elif isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name):
            nodos[nodo.targets[0].id] = nodo
    seleccion = ast.Module(body=[nodos[nombre] for nombre in nombres], type_ignores=[])
    exec(compile(seleccion, NOTEBOOK.name, 'exec'), entorno)


def ejecutar_celdas(celdas, entorno, *ids):
    for cell_id in ids:
        exec(compile(celdas[cell_id], f'{NOTEBOOK.name}:{cell_id}', 'exec'), entorno)


@pytest.fixture
def entorno_hym(celdas_python):
    entorno = {
        'np': np, 'pd': pd, 'stats': stats, 'chi2_contingency': chi2_contingency,
        'drift': drift, 'tqdm': tqdm,
        'etapa': lambda _nombre: nullcontext(), 'display': lambda _tabla: None,
    }
    for cell_id, nombres in (
        ('46fbd911', ('CFG', 'NUMERICAS', 'CATEGORICAS', 'ADWIN_COLS')),
        ('bbb5871f', ('ajustar_bh', 'valores_finitos')),
        ('fb8c655c', ('contar_categorias', 'calcular_chi', 'probar_chi', 'agrupar_marital', 'resumir_marital_agrupado', 'calcular_chi_marital_agrupado', 'GRUPOS_MARITAL', 'CATEGORIAS_MARITAL_BASE')),
        ('ea8eeb30', ('inyectar_drift', 'probar_inyeccion_drift')),
        ('592938b0', ('resumir_primeras_alertas', 'probar_retraso_adwin')),
    ):
        extraer_definiciones(celdas_python[cell_id], nombres, entorno)
    return entorno


@pytest.mark.parametrize('nombre', ['probar_chi', 'probar_inyeccion_drift', 'probar_retraso_adwin'])
def test_autopruebas_del_notebook(entorno_hym, nombre, capsys):
    entorno_hym[nombre]()
    assert 'PASS' in capsys.readouterr().out


@pytest.mark.parametrize('ref_vacia,cur_vacia', [(True, False), (False, True), (True, True)])
def test_lectura_de_cohortes_vacias(celdas_python, entorno_hym, capsys, ref_vacia, cur_vacia):
    datos = pd.DataFrame({
        'Education': ['Graduation', 'PhD'], 'Marital_Status': ['Single', 'Married'],
    })
    ref = datos.iloc[:0] if ref_vacia else datos
    cur = datos.iloc[:0] if cur_vacia else datos
    tablas = []
    entorno_hym.update(
        df_ref=ref, df_cur=cur, display=tablas.append,
        res_chi=entorno_hym['calcular_chi'](ref, cur, list(datos), progreso=False),
    )
    ejecutar_celdas(celdas_python, entorno_hym, '544bc81d')
    salida = capsys.readouterr().out
    assert 'nan%' not in salida.lower()
    assert 'Mayor aumento' not in salida and 'Mayor disminución' not in salida
    assert len(tablas) == 5
    for etiqueta, sufijo, vacia in (('Referencia', 'ref', ref_vacia),
                                    ('Comparación', 'cur', cur_vacia)):
        assert salida.count(f'{etiqueta}: No evaluable: cohorte vacía') == (2 if vacia else 0)
        assert salida.count(f'dominante(s) en {etiqueta.lower()}') == (0 if vacia else 2)
        for tabla in tablas[:2]:
            assert tabla['delta_pp'].isna().all()
            if vacia:
                assert tabla[f'p_{sufijo}'].isna().all()
            else:
                np.testing.assert_allclose(tabla[f'p_{sufijo}'], [0.5, 0.5])
    assert tablas[-1]['observacion'].str.contains('Cohorte vacía').all()


@pytest.mark.parametrize('elegibles,esperadas', [(0, 0), (1, 0), (2, 1), (10, 4)])
@pytest.mark.parametrize('incluye_casados', [False, True], ids=['sin-Married', 'con-Married'])
def test_intervencion_estado_civil_categorico(entorno_hym, elegibles, esperadas, incluye_casados):
    estados = ['Single'] * elegibles + [None, '__FALTANTE__']
    if incluye_casados:
        estados.append('Married')
    filas = len(estados)
    original = pd.DataFrame({
        'ID': np.arange(50, 50 + filas),
        'Dt_Customer': pd.date_range('2020-01-01', periods=filas),
        'Income': np.linspace(100., 200., filas),
        'MntWines': np.linspace(10., 20., filas),
        'Marital_Status': pd.Categorical(estados),
        'Education': ['Graduation'] * filas,
    }, index=101 + np.arange(filas) * 3)
    original.loc[original.index[-2:], 'Income'] = [0., np.nan]
    original.loc[original.index[-2:], 'MntWines'] = [np.nan, 0.]
    anterior = original.copy(deep=True)
    inyectar = entorno_hym['inyectar_drift']

    alterado, reporte = inyectar(original, semilla=42)
    repetido, reporte_repetido = inyectar(original, semilla=42)
    pd.testing.assert_frame_equal(original, anterior)
    pd.testing.assert_frame_equal(alterado, repetido)
    pd.testing.assert_series_equal(reporte, reporte_repetido)
    pd.testing.assert_frame_equal(alterado.drop(columns=OBJETIVOS), original.drop(columns=OBJETIVOS))
    assert alterado.columns.equals(original.columns)
    assert isinstance(alterado['Marital_Status'].dtype, pd.CategoricalDtype)
    assert ('Married' in alterado['Marital_Status'].cat.categories) == (incluye_casados or esperadas > 0)
    assert alterado['Marital_Status'].iloc[:elegibles].eq('Married').sum() == esperadas
    pd.testing.assert_series_equal(
        alterado['Marital_Status'].iloc[elegibles:].astype('string'),
        original['Marital_Status'].iloc[elegibles:].astype('string'),
    )
    pd.testing.assert_series_equal(alterado['Income'], original['Income'] * 1.25)
    pd.testing.assert_series_equal(alterado['MntWines'], original['MntWines'] * 1.60)
    assert reporte['income_modificados'] == reporte['vinos_modificados_no_cero'] == filas - 2
    assert reporte['estado_civil_elegibles'] == elegibles
    assert reporte['estado_civil_modificados'] == esperadas
    if elegibles:
        assert reporte['estado_civil_efectivo_pct'] == pytest.approx(100 * esperadas / elegibles)
    else:
        assert pd.isna(reporte['estado_civil_efectivo_pct'])


def crear_cohortes_sinteticas(entorno, datos_insuficientes):
    generador = np.random.default_rng(42)
    filas = 512
    datos = {columna: generador.normal(size=filas) for columna in entorno['NUMERICAS']}
    datos.update({columna: np.tile([0, 1], filas // 2) for columna in entorno['CATEGORICAS']})
    datos.update(
        Income=generador.normal(50_000, 200, filas),
        MntWines=generador.normal(100, 2, filas),
        MntMeatProducts=np.full(filas, 25.),
        Marital_Status=np.tile(['Single', 'Married'], filas // 2),
    )
    for columna in ('Income', 'MntWines'):
        datos[columna][[1, 6]] = np.nan
    if datos_insuficientes:
        for columna in entorno['ADWIN_COLS']:
            datos[columna][10:] = np.nan
        datos['Marital_Status'] = ['Single'] + ['Married'] * (filas - 1)

    base = pd.DataFrame(datos)
    fechas = pd.date_range('2020-01-01', periods=2 * filas)
    ref = base.assign(ID=np.arange(filas), Dt_Customer=fechas[:filas])
    cur = base.assign(ID=np.arange(filas, 2 * filas), Dt_Customer=fechas[filas:])
    return ref, cur


def verificar_alertas_controladas(entorno, controlado):
    corte = entorno['corte_control']
    secuencia = entorno['df_seq_control']
    eventos = entorno['alertas_control']
    posteriores = eventos.loc[eventos['indice_0based'].ge(corte)]
    pd.testing.assert_frame_equal(entorno['alerta_post'], posteriores)
    for variable in entorno['ADWIN_COLS']:
        fila = controlado.loc[variable]
        alertas = posteriores.loc[posteriores['variable'].eq(variable)].sort_values('indice_0based')
        if fila.estado_adwin == 'Evaluado':
            assert fila.alertas_adwin_total == eventos['variable'].eq(variable).sum()
            assert fila.alertas_adwin_posteriores == len(alertas)
        else:
            assert alertas.empty
            assert pd.isna(fila.alertas_adwin_total) and pd.isna(fila.alertas_adwin_posteriores)
        if alertas.empty:
            assert fila[['primera_alerta_adwin_indice', 'primera_alerta_adwin_fecha',
                         'retraso_adwin_filas', 'observaciones_adwin_hasta_alerta']].isna().all()
            continue
        primera = alertas.iloc[0]
        indice = int(primera.indice_0based)
        assert fila.primera_alerta_adwin_indice == indice
        assert fila.primera_alerta_adwin_fecha == primera.fecha == secuencia.iloc[indice].Dt_Customer
        assert primera.ID == secuencia.iloc[indice].ID
        assert fila.retraso_adwin_filas == indice - corte
        assert fila.observaciones_adwin_hasta_alerta == np.isfinite(
            secuencia[variable].iloc[corte:indice + 1].to_numpy(dtype=float)
        ).sum()


@pytest.mark.parametrize('datos_insuficientes', [False, True], ids=['detectable', 'no-evaluable'])
def test_escenario_controlado_integral(celdas_python, entorno_hym, datos_insuficientes):
    assert river.__version__ == '0.27.0'
    ref, cur = crear_cohortes_sinteticas(entorno_hym, datos_insuficientes)
    assert len(ref) == len(cur) == 512 and ref['Dt_Customer'].max() < cur['Dt_Customer'].min()
    assert set(ref['ID']).isdisjoint(cur['ID'])
    ref_original, cur_original = ref.copy(deep=True), cur.copy(deep=True)
    entorno_hym.update(df_ref=ref, df_cur=cur, df_seq=pd.concat([ref, cur], ignore_index=True))

    # Celdas reales de los métodos observados, la intervención, el informe y la consolidación.
    ejecutar_celdas(
        celdas_python, entorno_hym, 'bbb5871f', 'ddd4c675', 'fb8c655c', '544bc81d', '8a653bd7',
        'ea8eeb30', '5ce5457b', '592938b0', 'a56fa39e',
    )
    pd.testing.assert_frame_equal(ref, ref_original)
    pd.testing.assert_frame_equal(cur, cur_original)
    alterado = entorno_hym['df_shifted']
    assert alterado.columns.equals(cur.columns)
    pd.testing.assert_frame_equal(alterado.drop(columns=OBJETIVOS), cur.drop(columns=OBJETIVOS))
    pd.testing.assert_series_equal(alterado['Income'], cur['Income'] * 1.25)
    pd.testing.assert_series_equal(alterado['MntWines'], cur['MntWines'] * 1.60)
    cambios = cur['Marital_Status'].ne(alterado['Marital_Status'])
    assert cambios.sum() == (0 if datos_insuficientes else 90)
    assert cur.loc[cambios, 'Marital_Status'].eq('Single').all()
    assert alterado.loc[cambios, 'Marital_Status'].eq('Married').all()

    controlado = entorno_hym['res_control'].set_index('variable')
    resumen = entorno_hym['resumen_final'].set_index(['escenario', 'variable'])
    assert controlado.index.is_unique and resumen.index.is_unique
    observados = resumen.loc['OBSERVADO']
    consolidados = resumen.loc['CONTROLADO']
    assert set(resumen.index.get_level_values('escenario')) == {'OBSERVADO', 'CONTROLADO'}
    assert controlado['escenario'].eq('CONTROLADO').all()
    assert set(controlado.index) == set(consolidados.index) == set(observados.index)
    assert set(controlado.index) == set(entorno_hym['NUMERICAS'] + entorno_hym['CATEGORICAS'])
    np.testing.assert_allclose(consolidados['ks'], controlado.loc[consolidados.index, 'ks'])
    np.testing.assert_allclose(consolidados['psi'], controlado.loc[consolidados.index, 'psi'])
    np.testing.assert_allclose(consolidados['p_chi2_ajustado'],
                               controlado.loc[consolidados.index, 'p_chi2_ajustado'])
    assert observados.loc['Income', 'estado_chi2'] == consolidados.loc['Income', 'estado_chi2'] == 'no_aplicado'
    assert consolidados.loc['Marital_Status', 'estado_ks'] == 'no_aplicado'
    assert consolidados.loc['Year_Birth', 'estado_adwin'] == 'no_aplicado'
    assert controlado.loc['Marital_Status', ['ks', 'psi', 'alertas_adwin_total']].isna().all()
    assert pd.isna(consolidados.loc['Year_Birth', 'alertas_adwin'])
    verificar_alertas_controladas(entorno_hym, controlado)

    if datos_insuficientes:
        assert consolidados.loc['Income', ['estado_ks', 'estado_psi', 'estado_adwin']].eq(
            'calculado_no_evaluable'
        ).all()
        assert controlado.loc['Income', ['ks', 'p_ajustado', 'drift_ks', 'psi',
                                         'alertas_adwin_total', 'alertas_adwin_posteriores']].isna().all()
        assert consolidados.loc['Income', ['ks', 'p_ks', 'psi', 'alertas_adwin']].isna().all()
        assert consolidados.loc['Marital_Status', 'estado_chi2'] == 'calculado_no_evaluable'
        assert controlado.loc['Marital_Status', ['p_chi2_ajustado', 'drift_chi']].isna().all()
        assert entorno_hym['alertas_control'].empty
    else:
        assert observados.loc['Income', 'ks'] == 0
        assert bool(controlado.loc['Income', 'drift_ks'])
        assert controlado.loc['MntWines', 'intensidad_psi'] == 'Alto'
        assert bool(controlado.loc['Marital_Status', 'drift_chi'])
        assert controlado.loc[['Income', 'MntWines'], 'alertas_adwin_posteriores'].gt(0).any()
        assert consolidados.loc['MntMeatProducts', 'estado_adwin'] == 'evaluado'
        assert controlado.loc['MntMeatProducts', 'alertas_adwin_posteriores'] == 0


def test_marital_status_grouped_complementary_contrast(entorno_hym):
    agrupar = entorno_hym['agrupar_marital']
    resumen = entorno_hym['resumir_marital_agrupado']
    calcular = entorno_hym['calcular_chi_marital_agrupado']
    ref = pd.DataFrame({'Marital_Status': ['Married', 'Together', 'Single', 'Divorced', 'Absurd', None] * 20})
    cur = pd.DataFrame({'Marital_Status': ['Married'] * 60 + ['Together'] * 20 + ['Single'] * 10 + ['Divorced'] * 10 + ['YOLO'] * 10})
    assert set(agrupar(ref.Marital_Status)) <= {'Married', 'Together', 'Single', 'Divorced', 'Otros'}
    tabla = resumen(ref, cur)
    assert tabla.index.tolist() == ['Married', 'Together', 'Single', 'Divorced', 'Otros']
    assert tabla.loc['Otros', 'n_ref'] == 40  # 20 categorías raras y 20 valores faltantes
    assert tabla.loc['Otros', 'n_cur'] == 10
    base = pd.DataFrame({'variable': ['Education', 'Marital_Status'], 'p_valor': [0.01, np.nan]})
    salida = calcular(ref, cur, base, alpha=0.05).iloc[0]
    assert salida['variante'] == 'categorias_agrupadas'
    assert salida['grupos'] == 'Married|Together|Single|Divorced|Otros'
    assert salida['supuestos_ok']
    assert np.isfinite(salida['chi2']) and np.isfinite(salida['p_ajustado'])
