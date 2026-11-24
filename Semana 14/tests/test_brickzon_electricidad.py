"""Pruebas unitarias sintéticas para las secciones de Brickzon (Autoencoder y clasificadores).

Valida ExtractorAutoencoder, ProgresoEpocas, BarraXGB, BarraLightGBM, crear_modelo,
ajustar_clasificador, metricas_binarias y la integridad del notebook sin requerir
GPU ni ejecución completa del flujo experimental.
"""
import ast
import contextlib
import copy
import gc
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ELEC = ROOT / 'Semana 14' / 'Laboratorio_Electricidad_Integral.ipynb'


def celda_por_id(nb_o_path, cell_id):
    """Localiza una celda mediante su ID estable y verifica que exista exactamente una coincidencia."""
    if isinstance(nb_o_path, (str, Path)):
        nb = json.loads(Path(nb_o_path).read_text(encoding='utf-8'))
    elif isinstance(nb_o_path, dict):
        nb = nb_o_path
    else:
        raise TypeError("nb_o_path debe ser una ruta Path/str o un diccionario de notebook")
    matches = [c for c in nb.get('cells', []) if c.get('id') == cell_id]
    assert len(matches) == 1, (
        f"Se esperaba exactamente 1 coincidencia para la celda con id='{cell_id}', "
        f"pero se encontraron {len(matches)}."
    )
    return matches[0]


def codigo_celda_id(nb_o_path, cell_id):
    """Extrae el código o texto concatenado de una celda localizada por su ID estable."""
    c = celda_por_id(nb_o_path, cell_id)
    return ''.join(c.get('source', []))


def cell(path, id_o_idx):
    """Extrae el código fuente concatenado localizando preferentemente por ID estable o por índice."""
    if isinstance(id_o_idx, str):
        return codigo_celda_id(path, id_o_idx)
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    return ''.join(nb['cells'][id_o_idx]['source'])


def definitions(source, names, namespace):
    """Extrae y compila exclusivamente las funciones, clases o asignaciones indicadas."""
    tree = ast.parse(source)
    nodes = [
        n for n in tree.body
        if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
        or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
    ]
    extracted_names = set()
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            extracted_names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    extracted_names.add(t.id)
    assert extracted_names == set(names), f"Faltaron definiciones: {set(names) - extracted_names}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<brickzon notebook definitions>', 'exec'), namespace)
    return namespace


class DummyTqdm:
    """Implementación sintética de barra de progreso para pruebas deterministas."""
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get('total', 0)
        self.desc = kwargs.get('desc', '')
        self.unit = kwargs.get('unit', '')
        self.leave = kwargs.get('leave', True)
        self.n = 0
        self.postfix = {}
        self.closed = False

    def update(self, n=1):
        self.n += n

    def set_postfix(self, **kwargs):
        self.postfix.update(kwargs)

    def close(self):
        self.closed = True


class MockTensor:
    """Representación ligera de tensor para el grafo sintético de Keras."""
    def __init__(self, shape, name=None):
        self.shape = shape
        self.name = name


class MockDense:
    """Capa Dense sintética que valida conexiones y propaga dimensiones."""
    def __init__(self, units, activation='linear', name=None):
        self.units = units
        self.activation = activation
        self.name = name

    def __call__(self, x):
        in_dim = x.shape[-1] if hasattr(x, 'shape') else None
        return MockTensor(shape=(None, self.units), name=self.name)


class MockModel:
    """Modelo Keras sintético para el Autoencoder y Encoder."""
    def __init__(self, inputs, outputs, name=None):
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        self.optimizer = None
        self.loss = None
        self.history = None
        self.fitted = False

    def compile(self, optimizer=None, loss=None):
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, x, y, epochs=1, batch_size=256, shuffle=True, validation_split=0.1, verbose=0, callbacks=None):
        self.fitted = True
        cbs = callbacks or []
        for cb in cbs:
            if hasattr(cb, 'on_train_begin'):
                cb.on_train_begin()
        losses = []
        val_losses = []
        for epoch in range(epochs):
            l = 0.05 / (epoch + 1)
            vl = 0.06 / (epoch + 1)
            losses.append(l)
            val_losses.append(vl)
            for cb in cbs:
                if hasattr(cb, 'on_epoch_end'):
                    cb.on_epoch_end(epoch, {'loss': l, 'val_loss': vl})
        for cb in cbs:
            if hasattr(cb, 'on_train_end'):
                cb.on_train_end()
        self.history = SimpleNamespace(history={'loss': losses, 'val_loss': val_losses})
        return self.history

    def predict(self, x, batch_size=256, verbose=0):
        out_units = self.outputs.shape[-1] if hasattr(self.outputs, 'shape') else 32
        rng = np.random.default_rng(111)
        w = rng.normal(0, 0.1, size=(x.shape[1], out_units)).astype(np.float32)
        return (x @ w).astype(np.float32)


class MockTrainingCallback:
    """Base callback de xgboost para aislamiento sintético."""
    pass


class MockXGBClassifier:
    """Clasificador XGBoost sintético con registro de hiperparámetros."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.fitted_ = False
        self.eval_set_received = None

    def get_params(self, deep=True):
        return self.kwargs

    def fit(self, X, y, eval_set=None, verbose=False, **kwargs):
        self.fitted_ = True
        self.eval_set_received = eval_set
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.7, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


class MockLGBMClassifier:
    """Clasificador LightGBM sintético con soporte de callbacks."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.fitted_ = False
        self.eval_set_received = None

    def get_params(self, deep=True):
        return self.kwargs

    def fit(self, X, y, eval_set=None, callbacks=None, **kwargs):
        self.fitted_ = True
        self.eval_set_received = eval_set
        if callbacks:
            for cb in callbacks:
                env = SimpleNamespace(iteration=0, begin_iteration=0, end_iteration=1)
                cb(env)
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.65, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


@pytest.fixture(scope='module')
def brickzon_env():
    """Compila las celdas 39 y 43 en un espacio de nombres con dependencias simuladas."""
    mock_keras = SimpleNamespace(
        callbacks=SimpleNamespace(
            Callback=object,
            EarlyStopping=lambda monitor='val_loss', patience=4, restore_best_weights=True: SimpleNamespace(
                monitor=monitor, patience=patience, restore_best_weights=restore_best_weights
            ),
        ),
        utils=SimpleNamespace(set_random_seed=lambda s: None),
        Input=lambda shape, name=None: MockTensor(shape=(None,) + shape, name=name),
        Model=MockModel,
        optimizers=SimpleNamespace(Adam=lambda learning_rate=0.003: SimpleNamespace(learning_rate=learning_rate)),
    )
    mock_layers = SimpleNamespace(Dense=MockDense)
    mock_xgboost = SimpleNamespace(
        callback=SimpleNamespace(TrainingCallback=MockTrainingCallback),
        XGBClassifier=MockXGBClassifier,
    )
    mock_lightgbm = SimpleNamespace()

    cfg = SimpleNamespace(
        modo='RAPIDO',
        semilla=111,
        epochs_rapido=2,
        epochs_completo=20,
        paciencia_ae=4,
        batch_ae=256,
        n_jobs=2,
        n_componentes_ica=10,
        n_componentes_pls=8,
        umbral_clase=0.5,
    )

    from sklearn.ensemble import RandomForestClassifier

    ns = {
        'np': np,
        'gc': gc,
        'keras': mock_keras,
        'layers': mock_layers,
        'StandardScaler': StandardScaler,
        'RandomForestClassifier': RandomForestClassifier,
        'XGBClassifier': MockXGBClassifier,
        'LGBMClassifier': MockLGBMClassifier,
        'xgboost': mock_xgboost,
        'lightgbm': mock_lightgbm,
        'CFG': cfg,
        'tqdm': DummyTqdm,
        'etapa': lambda name: contextlib.nullcontext(),
        'average_precision_score': average_precision_score,
        'roc_auc_score': roc_auc_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'f1_score': f1_score,
        'accuracy_score': accuracy_score,
        'confusion_matrix': confusion_matrix,
    }

    definitions(codigo_celda_id(ELEC, '45912de6'), {'ProgresoEpocas', 'ExtractorAutoencoder'}, ns)
    definitions(
        codigo_celda_id(ELEC, 'd9c7fffa'),
        {'BarraXGB', 'BarraLightGBM', 'PARES', 'crear_modelo', 'ajustar_clasificador', 'metricas_binarias'},
        ns,
    )
    return ns


# ==============================================================================
# 1. INTEGRIDAD Y CONTENIDO CIENTÍFICO DEL NOTEBOOK
# ==============================================================================

def test_notebook_json_integrity_and_scientific_report():
    """Comprueba integridad JSON, secciones científicas y consistencia con COMPLETO."""
    assert ELEC.is_file(), f"No se encontró el notebook en {ELEC}"
    nb=json.loads(ELEC.read_text(encoding='utf-8'))
    assert nb.get('nbformat')==4 and nb.get('nbformat_minor',0)>=2
    cells=nb.get('cells',[])
    ids=[c.get('id') for c in cells if c.get('id') is not None]
    assert all(isinstance(cid,str) and cid for cid in ids)
    assert len(ids)==len(set(ids))
    for c in cells:
        assert c.get('cell_type') in ('code','markdown') and isinstance(c.get('source'),list)

    src_ae=codigo_celda_id(nb,'8b8cf438')
    src_clf=codigo_celda_id(nb,'0222255d')
    src_concl=codigo_celda_id(nb,'d9714129')
    src_test=codigo_celda_id(nb,'7c7951b4')
    src_interp=codigo_celda_id(nb,'1b89425b')
    src_outliers=codigo_celda_id(nb,'af9ad7ef')
    src_eda=codigo_celda_id(nb,'b2d83e0f')

    assert '### 6.3. Autoencoder' in src_ae and '32' in src_ae and 'TRAIN' in src_ae
    assert '## 7. Clasificadores' in src_clf and 'SMOTE' in src_clf and 'ADASYN' in src_clf
    assert 'TEST' in src_test and 'una sola vez' in src_test
    for term in ['0,256346','0,248981','0,237058','0,230637']:
        assert term in src_interp
    for term in ['0,256346','0,239720','0,231332']:
        assert term in src_outliers
    for term in ['29.660','25,621 %','25,10 %','31,20 %']:
        assert term in src_eda
    for term in ['Autoencoder + SMOTE + XGBoost','0,259516','0,258141','0,21986','0,6839','0,8083','0,1908','0,3843','0,25497','0,0397']:
        assert term in src_concl
    assert 'selección secuencial' in src_concl and 'significancia estadística' in src_concl

    markdown='\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='markdown')
    for forbidden in ['Marco Renato Castilla','Poma','Vera','Aldana','Brickzon','Responsable:','[PENDIENTE','Guía obligatoria para los agentes','Checklist']:
        assert re.search(r'\\b'+re.escape(forbidden)+r'\\b', markdown, re.IGNORECASE) is None, f"Referencia editorial interna persistente: {forbidden}"
    assert 'class ExtractorAutoencoder' in codigo_celda_id(nb,'45912de6')
    src_code_clf=codigo_celda_id(nb,'d9c7fffa')
    assert 'def crear_modelo' in src_code_clf and 'def metricas_binarias' in src_code_clf


def test_notebook_compatibilidad_celdas_vera_y_busqueda_por_id():
    """Verifica compatibilidad con adición de celdas (compatibilidad con celdas adicionales) y robustez de IDs."""
    nb = json.loads(ELEC.read_text(encoding='utf-8'))
    initial_count = len(nb['cells'])

    # IDs estables de secciones y funciones
    target_ids = ['8b8cf438', '45912de6', '0222255d', 'd9c7fffa', 'd9714129']
    for tid in target_ids:
        c = celda_por_id(nb, tid)
        assert c['id'] == tid
        assert isinstance(c['source'], list)

    # Simular la inserción de una celda adicional en capítulo 4
    nb_sim = copy.deepcopy(nb)
    nueva_celda_vera = {
        'cell_type': 'code',
        'execution_count': None,
        'id': 'cambio-outliers',
        'metadata': {},
        'outputs': [],
        'source': [
            "# 4.3. Comparación antes y después del tratamiento de outliers\n",
            "print('Outliers tratados')\n"
        ]
    }
    nb_sim['cells'].insert(20, nueva_celda_vera)
    assert len(nb_sim['cells']) == initial_count + 1

    # Comprobar que todas las celdas se siguen localizando correctamente por ID
    for tid in target_ids:
        c_sim = celda_por_id(nb_sim, tid)
        assert c_sim['id'] == tid
        assert c_sim['source'] == celda_por_id(nb, tid)['source']

    # Comprobar que una celda no existente genera AssertionError (assert len(matches) == 1)
    with pytest.raises(AssertionError, match='Se esperaba exactamente 1 coincidencia'):
        celda_por_id(nb_sim, 'celda_inexistente_12345')

    # Comprobar que una celda duplicada genera AssertionError
    nb_dup = copy.deepcopy(nb_sim)
    nb_dup['cells'].append({'id': '45912de6', 'cell_type': 'code', 'source': ['# duplicada']})
    with pytest.raises(AssertionError, match='Se esperaba exactamente 1 coincidencia'):
        celda_por_id(nb_dup, '45912de6')


# ==============================================================================
# 2. PRUEBAS DEL AUTOENCODER (EXTRACTORAUTOENCODER Y PROGRESOEPOCAS)
# ==============================================================================

def test_extractor_autoencoder_initialization_and_contracts(brickzon_env):
    """Verifica parámetros de inicialización y valores predeterminados."""
    ExtractorAutoencoder = brickzon_env['ExtractorAutoencoder']
    ae_def = ExtractorAutoencoder()
    assert ae_def.semilla == 111
    assert ae_def.modo == 'RAPIDO'
    assert hasattr(ae_def, '_ajustado') and not ae_def._ajustado

    ae_custom = ExtractorAutoencoder(semilla=999, modo='COMPLETO')
    assert ae_custom.semilla == 999
    assert ae_custom.modo == 'COMPLETO'


def test_extractor_autoencoder_fit_transform_shape_and_immutability(brickzon_env):
    """Verifica forma (n, 32), tipo float32, valores finitos y no alteración de inputs."""
    ExtractorAutoencoder = brickzon_env['ExtractorAutoencoder']
    ae = ExtractorAutoencoder(semilla=111, modo='RAPIDO')

    X = np.random.randn(20, 50).astype(np.float32)
    X_original = X.copy()

    fitted = ae.fit(X)
    assert fitted is ae
    assert ae._ajustado is True
    assert np.array_equal(X, X_original), "fit() modificó la matriz de entrada en su lugar."

    z = ae.transform(X)
    assert np.array_equal(X, X_original), "transform() modificó la matriz de entrada en su lugar."
    assert z.shape == (20, 32), f"Forma latente esperada (20, 32), se obtuvo {z.shape}"
    assert z.dtype == np.float32, f"Tipo de datos esperado float32, se obtuvo {z.dtype}"
    assert np.isfinite(z).all(), "La representación latente contiene valores no finitos (NaN o Inf)."

    # fit_transform produce resultado equivalente
    z_ft = ExtractorAutoencoder(semilla=111, modo='RAPIDO').fit_transform(X)
    assert z_ft.shape == (20, 32)
    assert z_ft.dtype == np.float32
    assert np.isfinite(z_ft).all()


def test_extractor_autoencoder_scaler_fitted_strictly_on_train(brickzon_env):
    """Verifica que el StandardScaler se ajuste solo en TRAIN y no se recalcule en VALID/TEST."""
    ExtractorAutoencoder = brickzon_env['ExtractorAutoencoder']
    ae = ExtractorAutoencoder(semilla=111)

    X_train = np.full((15, 40), 10.0, dtype=np.float32)
    ae.fit(X_train)

    mean_train = ae.scaler.mean_.copy()
    scale_train = ae.scaler.scale_.copy()

    # Matriz con media completamente diferente
    X_valid = np.full((10, 40), 200.0, dtype=np.float32)
    ae.transform(X_valid)

    # El scaler no debe haber cambiado sus parámetros aprendidos
    assert np.array_equal(ae.scaler.mean_, mean_train)
    assert np.array_equal(ae.scaler.scale_, scale_train)


def test_extractor_autoencoder_dimension_guard_and_unfitted_call(brickzon_env):
    """Verifica validación de mínimo 32 días y excepción si no se ha ajustado."""
    ExtractorAutoencoder = brickzon_env['ExtractorAutoencoder']
    ae = ExtractorAutoencoder()

    # Menos de 32 columnas
    X_insuficiente = np.ones((10, 25), dtype=np.float32)
    with pytest.raises(ValueError, match='32'):
        ae.fit(X_insuficiente)

    # transform() sin fit previo
    ae_sin_ajuste = ExtractorAutoencoder()
    with pytest.raises(RuntimeError, match='fit'):
        ae_sin_ajuste.transform(np.ones((10, 40), dtype=np.float32))


def test_extractor_autoencoder_fit_invalido_invalida_estado_previo(brickzon_env):
    """Verifica que un nuevo fit() inválido desajuste el modelo y bloquee transform() con RuntimeError."""
    ExtractorAutoencoder = brickzon_env['ExtractorAutoencoder']
    ae = ExtractorAutoencoder(semilla=111, modo='RAPIDO')

    # 1. Ajuste exitoso inicial
    X_valido = np.random.randn(20, 40).astype(np.float32)
    ae.fit(X_valido)
    assert ae._ajustado is True
    z = ae.transform(X_valido)
    assert z.shape == (20, 32)

    # 2. Re-fit inválido con menos de 32 columnas
    X_invalido_col = np.ones((10, 20), dtype=np.float32)
    with pytest.raises(ValueError, match='32'):
        ae.fit(X_invalido_col)
    assert ae._ajustado is False, "Tras un fit() fallido, _ajustado debe ser False"
    with pytest.raises(RuntimeError, match='fit'):
        ae.transform(X_valido)

    # 3. Nuevo ajuste exitoso y posterior re-fit inválido con valores no finitos (NaN)
    ae.fit(X_valido)
    assert ae._ajustado is True
    X_invalido_nan = X_valido.copy()
    X_invalido_nan[2, 3] = np.nan
    with pytest.raises(ValueError, match='finitos|NaN'):
        ae.fit(X_invalido_nan)
    assert ae._ajustado is False
    with pytest.raises(RuntimeError, match='fit'):
        ae.transform(X_valido)

    # 4. Nuevo ajuste exitoso y posterior re-fit con matriz no 2D (1D)
    ae.fit(X_valido)
    assert ae._ajustado is True
    with pytest.raises(ValueError, match='2D'):
        ae.fit(np.ones(50, dtype=np.float32))
    assert ae._ajustado is False
    with pytest.raises(RuntimeError, match='fit'):
        ae.transform(X_valido)

    # 5. Nuevo ajuste exitoso y posterior re-fit con datos no numéricos o None
    ae.fit(X_valido)
    assert ae._ajustado is True
    with pytest.raises(TypeError):
        ae.fit(None)
    assert ae._ajustado is False
    with pytest.raises(RuntimeError, match='fit'):
        ae.transform(X_valido)

    ae.fit(X_valido)
    assert ae._ajustado is True
    with pytest.raises(TypeError):
        ae.fit([['abc'] * 35] * 5)
    assert ae._ajustado is False
    with pytest.raises(RuntimeError, match='fit'):
        ae.transform(X_valido)


def test_progreso_epocas_callback(brickzon_env):
    """Verifica ciclo de vida y actualización de ProgresoEpocas."""
    ProgresoEpocas = brickzon_env['ProgresoEpocas']
    cb = ProgresoEpocas(total=10)
    assert cb.total == 10
    assert cb.barra is None

    cb.on_train_begin()
    assert cb.barra is not None
    assert cb.barra.total == 10

    cb.on_epoch_end(0, {'loss': 0.1234, 'val_loss': 0.2345})
    assert cb.barra.n == 1
    assert 'loss' in cb.barra.postfix and 'val_loss' in cb.barra.postfix

    # Manejo robusto cuando logs es None
    cb.on_epoch_end(1, None)
    assert cb.barra.n == 2

    cb.on_train_end()
    assert cb.barra.closed is True


# ==============================================================================
# 3. PRUEBAS DE CLASIFICADORES (FÁBRICA, BARRAXGB, BARRALIGHTGBM, AJUSTAR)
# ==============================================================================

def test_crear_modelo_rapido_and_completo(brickzon_env):
    """Verifica la parametrización de clasificadores base en modos RAPIDO y COMPLETO."""
    crear_modelo = brickzon_env['crear_modelo']
    cfg = brickzon_env['CFG']

    # Modo RAPIDO
    cfg.modo = 'RAPIDO'
    rf_rap = crear_modelo('random_forest')
    assert rf_rap.n_estimators == 80
    assert rf_rap.max_depth == 10
    assert rf_rap.min_samples_leaf == 2
    assert rf_rap.random_state == 111
    assert rf_rap.n_jobs == 2

    xgb_rap = crear_modelo('xgboost')
    assert xgb_rap.n_estimators == 80
    assert xgb_rap.max_depth == 6
    assert xgb_rap.learning_rate == 0.05
    assert xgb_rap.subsample == 0.85
    assert xgb_rap.colsample_bytree == 0.85
    assert xgb_rap.tree_method == 'hist'
    assert xgb_rap.random_state == 111

    lgb_rap = crear_modelo('lightgbm')
    assert lgb_rap.n_estimators == 90
    assert lgb_rap.max_depth == -1
    assert lgb_rap.num_leaves == 31
    assert lgb_rap.learning_rate == 0.05
    assert lgb_rap.random_state == 111

    # Modo COMPLETO
    cfg.modo = 'COMPLETO'
    rf_comp = crear_modelo('random_forest')
    assert rf_comp.n_estimators == 200

    xgb_comp = crear_modelo('xgboost')
    assert xgb_comp.n_estimators == 200

    lgb_comp = crear_modelo('lightgbm')
    assert lgb_comp.n_estimators == 250

    # Restaurar modo RAPIDO
    cfg.modo = 'RAPIDO'


def test_crear_modelo_raises_on_unknown(brickzon_env):
    """Verifica excepción clara para nombres de clasificador no soportados."""
    crear_modelo = brickzon_env['crear_modelo']
    with pytest.raises(ValueError, match='Clasificador desconocido: catboost'):
        crear_modelo('catboost')
    with pytest.raises(ValueError, match='Clasificador desconocido: None'):
        crear_modelo(None)


def test_barra_xgb_callback(brickzon_env):
    """Verifica callbacks nativos de XGBoost para avance por árbol."""
    BarraXGB = brickzon_env['BarraXGB']
    cb = BarraXGB(total=80)
    assert cb.total == 80
    assert cb.barra is None

    dummy_model = SimpleNamespace()
    ret_begin = cb.before_training(dummy_model)
    assert ret_begin is dummy_model
    assert cb.barra is not None

    # after_iteration debe avanzar la barra y retornar False (no cancelar entrenamiento)
    continuar = cb.after_iteration(dummy_model, epoch=0, evals_log={})
    assert continuar is False
    assert cb.barra.n == 1

    ret_end = cb.after_training(dummy_model)
    assert ret_end is dummy_model
    assert cb.barra.closed is True


def test_barra_lightgbm_callback(brickzon_env):
    """Verifica callbacks nativos de LightGBM para avance por árbol."""
    BarraLightGBM = brickzon_env['BarraLightGBM']
    cb = BarraLightGBM(total=90)
    assert cb.total == 90
    assert cb.barra is None
    assert cb.order == 10
    assert cb.before_iteration is False

    # Primera iteración crea la barra
    env1 = SimpleNamespace(iteration=0, begin_iteration=0, end_iteration=90)
    cb(env1)
    assert cb.barra is not None
    assert cb.barra.n == 1

    # Cierre automático al finalizar
    env_final = SimpleNamespace(iteration=89, begin_iteration=0, end_iteration=90)
    cb(env_final)
    assert cb.barra.n == 2
    assert cb.barra.closed is True

    # Método close explícito
    cb.close()


def test_ajustar_clasificador_dispatch(brickzon_env):
    """Verifica ajuste y paso de parámetros eval_set para los tres clasificadores."""
    ajustar_clasificador = brickzon_env['ajustar_clasificador']
    Z_tr = np.random.randn(40, 32).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=40)
    Z_va = np.random.randn(10, 32).astype(np.float32)
    y_va = np.random.randint(0, 2, size=10)

    # Random Forest
    rf = ajustar_clasificador('random_forest', Z_tr, y_tr)
    assert hasattr(rf, 'classes_')

    # XGBoost con validación
    xgb = ajustar_clasificador('xgboost', Z_tr, y_tr, Z_va, y_va)
    assert xgb.fitted_ is True
    assert xgb.eval_set_received is not None

    # LightGBM con validación y callback
    lgb = ajustar_clasificador('lightgbm', Z_tr, y_tr, Z_va, y_va)
    assert lgb.fitted_ is True
    assert lgb.eval_set_received is not None


# ==============================================================================
# 4. PRUEBAS DE MÉTRICAS BINARIAS
# ==============================================================================

def test_metricas_binarias_keys_and_numerical_contracts(brickzon_env):
    """Verifica presencia de las 7 métricas y tipos numéricos requeridos."""
    metricas_binarias = brickzon_env['metricas_binarias']
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.3, 0.75, 0.4])

    res = metricas_binarias(y_true, y_proba, threshold=0.5)

    claves_esperadas = {
        'pr_auc', 'roc_auc', 'precision_fraude', 'recall_fraude',
        'f1_fraude', 'accuracy', 'matriz_confusion'
    }
    assert set(res.keys()) == claves_esperadas

    for k in ['pr_auc', 'roc_auc', 'precision_fraude', 'recall_fraude', 'f1_fraude', 'accuracy']:
        val = res[k]
        assert isinstance(val, float)
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0

    cm = res['matriz_confusion']
    assert isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2
    assert sum(cm[0]) + sum(cm[1]) == len(y_true)


def test_metricas_binarias_perfect_and_imbalanced_cases(brickzon_env):
    """Verifica resultados exactos en casos sintéticos conocidos."""
    metricas_binarias = brickzon_env['metricas_binarias']

    # Predicción perfecta
    y_perfect = np.array([0, 0, 1, 1])
    proba_perfect = np.array([0.05, 0.1, 0.9, 0.95])
    res_perf = metricas_binarias(y_perfect, proba_perfect, threshold=0.5)
    assert res_perf['pr_auc'] == 1.0
    assert res_perf['roc_auc'] == 1.0
    assert res_perf['accuracy'] == 1.0
    assert res_perf['f1_fraude'] == 1.0
    assert res_perf['matriz_confusion'] == [[2, 0], [0, 2]]

    # Predicción con división por cero (sin predicciones positivas de fraude)
    y_imbal = np.array([0, 0, 0, 1])
    proba_baja = np.array([0.1, 0.1, 0.2, 0.2])  # Todo por debajo de threshold 0.5
    res_cero = metricas_binarias(y_imbal, proba_baja, threshold=0.5)
    assert res_cero['precision_fraude'] == 0.0
    assert res_cero['recall_fraude'] == 0.0
    assert res_cero['f1_fraude'] == 0.0
    assert res_cero['accuracy'] == 0.75
    assert res_cero['matriz_confusion'] == [[3, 0], [1, 0]]


def test_metricas_binarias_input_validations(brickzon_env):
    """Verifica validación estricta de probabilidades finitas y en [0, 1]."""
    metricas_binarias = brickzon_env['metricas_binarias']
    y = np.array([0, 1])

    # Probabilidades mayores a 1.0
    with pytest.raises(AssertionError):
        metricas_binarias(y, [0.2, 1.2])

    # Probabilidades negativas
    with pytest.raises(AssertionError):
        metricas_binarias(y, [-0.1, 0.8])

    # Probabilidades con NaN o Inf
    with pytest.raises(AssertionError):
        metricas_binarias(y, [0.3, np.nan])
    with pytest.raises(AssertionError):
        metricas_binarias(y, [0.3, np.inf])

    # Vector 2D plano aceptado correctamente
    proba_2d = np.array([[0.2], [0.8]])
    res_2d = metricas_binarias(y, proba_2d)
    assert res_2d['accuracy'] == 1.0


# ==============================================================================
# 5. PRUEBAS DE PARES Y PROTOCOLO DE EVALUACIÓN
# ==============================================================================

def test_protocolo_pares_extractor_clasificador(brickzon_env):
    """Verifica el acuerdo operativo de pares extractor x clasificador."""
    PARES = brickzon_env['PARES']
    assert PARES == {
        'autoencoder': 'xgboost',
        'ica': 'random_forest',
        'pls': 'lightgbm'
    }
