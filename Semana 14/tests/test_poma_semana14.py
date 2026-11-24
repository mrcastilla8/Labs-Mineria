"""Pruebas unitarias sintéticas para las secciones de Poma (no ejecutan notebooks)."""
import ast
import contextlib
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
tqdm = lambda iterable, **kwargs: iterable  # barra sustituida en pruebas aisladas

ROOT = Path(__file__).resolve().parents[2]
ELEC = ROOT / 'Semana 14' / 'Laboratorio_Electricidad_Integral.ipynb'
HYM = ROOT / 'Semana 14' / 'Plantilla_HyM_Data_Drift_Colab_Agent_Oriented.ipynb'

def cell(path, cell_id):
    nb=json.loads(path.read_text(encoding='utf-8'))
    if path == ELEC:
        matches=[c for c in nb['cells'] if c.get('id')==cell_id]
        assert len(matches)==1, f"Se esperaba una celda con id={cell_id}; encontradas={len(matches)}"
        selected=matches[0]
    else:
        # HyM conserva sus referencias por índice; este cambio aplica solo a Electricidad.
        selected=nb['cells'][cell_id]
    return ''.join(selected['source'])

def definitions(source, names, namespace):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert set(n.name for n in nodes) == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<selected notebook functions>', 'exec'), namespace)
    return namespace

@pytest.fixture(scope='module')
def electric():
    from scipy.interpolate import PchipInterpolator
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression
    ns = {'np': np, 'pd': pd, 'tqdm': tqdm, 'warnings': warnings, 'PchipInterpolator': PchipInterpolator,
          'IterativeImputer': IterativeImputer, 'LinearRegression': LinearRegression}
    definitions(cell(ELEC, '76a25cb9'), {'validar_calendario_eda','muestrear_matrices_proporcional'}, ns)
    source = cell(ELEC, '702c77ed')
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INTERPOLACIONES' for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<electricidad Poma>', 'exec'), ns)
    return ns

@pytest.fixture(scope='module')
def hym():
    ns = {'Path': Path, 'csv': csv, 'pd': pd, 'np': np, 'plt': plt, 'tqdm': tqdm,
          'display': lambda *_args, **_kwargs: None,
          'CFG': {'ruta_csv': None, 'nombre_csv': 'marketing_campaign.csv', 'mostrar_graficos': True,
                  'max_graficos_por_metodo': 4}, 'EN_COLAB': False,
          'COLUMNAS_REQUERIDAS': REQUIRED, 'NUMERICAS': NUMERICAS, 'CATEGORICAS': CATEGORICAS}
    ns['etapa'] = lambda _name: contextlib.nullcontext()
    definitions(cell(HYM, 21), {'localizar_csv', 'cargar_datos'}, ns)
    definitions(cell(HYM, 24), {'auditar_dataset'}, ns)
    definitions(cell(HYM, 28), {'preparar_dataset'}, ns)
    definitions(cell(HYM, 42), {'resumir_cohortes'}, ns)
    definitions(cell(HYM, 45), {'graficar_eda_cohortes'}, ns)
    return ns

NUMERICAS = ['Year_Birth','Income','Kidhome','Teenhome','MntWines','MntFruits','MntMeatProducts',
             'MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases',
             'NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth']
CATEGORICAS = ['Education','Marital_Status','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4',
               'AcceptedCmp5','Complain','Response']
REQUIRED = ['ID','Dt_Customer'] + NUMERICAS + CATEGORICAS + ['Recency','Z_CostContact','Z_Revenue']

@pytest.mark.parametrize('method', ['interpolar_vecinos','interpolar_lineal','interpolar_pchip'])
def test_interpolators_contracts_and_edge_cases(electric, method):
    x = np.array([[1,np.nan,3,4], [np.nan,2,np.nan,6], [np.nan]*4, [5,5,5,5], [np.nan,-5,np.nan,np.nan],
                  [-4,np.nan,-2,np.nan], [np.nan,7,7,np.nan], [2,3,4,5]], dtype=np.float32)
    before = x.copy(); t = np.array([0,1,3,8], dtype=np.float64)
    fn = electric[method]
    out = fn(x, t, mostrar=False) if method != 'interpolar_vecinos' else fn(x, chunk=2, mostrar=False)
    assert out.shape == x.shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert np.array_equal(x, before, equal_nan=True)
    assert np.array_equal(out[np.isfinite(x)], x[np.isfinite(x)])
    assert np.all(out[2] == 0)
    if method in ('interpolar_lineal','interpolar_pchip'): assert np.all(out[4] == -5)
    else: assert out[4].tolist() == [0.,-5.,0.,0.]
    if method == 'interpolar_vecinos': assert out[6].tolist() == [0.,7.,7.,0.]
    else: assert np.all(out[6] == 7)
    assert np.array_equal(fn(x, t, mostrar=False) if method != 'interpolar_vecinos' else fn(x, chunk=2, mostrar=False), out)

def test_eda_calendar_uses_known_irregular_dates(electric):
    fechas=pd.to_datetime(['2020/01/03','2020/01/06','2020/01/10'],format='%Y/%m/%d',errors='raise')
    calendario=electric['validar_calendario_eda'](fechas,[0,3,7])
    assert calendario.equals(pd.DatetimeIndex(fechas))
    assert not any(fecha.year==1970 for fecha in calendario)
    with pytest.raises(ValueError,match='longitud'):
        electric['validar_calendario_eda'](fechas,[0,3])
    with pytest.raises(ValueError,match='crecientes'):
        electric['validar_calendario_eda'](fechas[::-1],[0,3,7])
    with pytest.raises(ValueError,match='no coinciden'):
        electric['validar_calendario_eda'](fechas,[0,2,7])
    eda=cell(ELEC,'76a25cb9'); plots=cell(ELEC,'73a62726')
    assert "pd.to_datetime(fechas, format='%Y/%m/%d', errors='raise')" in eda
    assert "'fecha':fechas_calendario" in eda
    assert all(etiqueta in eda for etiqueta in ("('Q1',q1","('Q3',q3","('límite inferior IQR'","('límite superior IQR'"))
    assert 'muestrear_matrices_proporcional((X_train_raw,X_valid_raw),CFG.semilla,max_total=200_000)' in eda
    assert 'no selecciona modelos' in eda
    assert 'ax.plot(fechas_calendario' in plots and 'pd.to_datetime(t_dias)' not in plots


def test_eda_sampling_is_proportional_reproducible_and_bounded(electric):
    train=np.zeros((300,1000),dtype=np.float32)
    valid=np.ones((100,1000),dtype=np.float32)
    sample=electric['muestrear_matrices_proporcional']((train,valid),semilla=27,max_total=200_000)
    repeat=electric['muestrear_matrices_proporcional']((train,valid),semilla=27,max_total=200_000)
    assert [len(x) for x in sample]==[150_000,50_000]
    assert sum(map(len,sample))==200_000
    assert all(np.array_equal(a,b) for a,b in zip(sample,repeat))
    smaller=electric['muestrear_matrices_proporcional']((train[:10],valid[:10]),semilla=27,max_total=200_000)
    assert [len(x) for x in smaller]==[10_000,10_000]


def test_neighbor_policy_and_temporal_validation(electric):
    f=electric['interpolar_vecinos']
    assert f([[1,np.nan,3]],mostrar=False)[0,1] == 2
    assert f([[np.nan,2,np.nan]],mostrar=False).tolist() == [[0.,2.,0.]]
    with pytest.raises(ValueError): electric['interpolar_lineal'](np.ones((2,3)),[0,1],False)
    with pytest.raises(ValueError): electric['interpolar_pchip'](np.ones((2,3)),[0,1,1],False)
    with pytest.raises(ValueError): f(np.ones(4))

def test_iterative_fit_only_once_and_preserves_observed(electric):
    x=np.array([[1,2,np.nan],[2,3,4],[3,np.nan,6],[4,5,8],[5,6,10]],dtype=np.float32)
    model=electric['ImputadorIterativoOriginal'](9)
    train=model.fit_transform(x); fitted=model.modelo; iterations=fitted.n_iter_
    valid=np.array([[np.nan,7,9],[8,np.nan,12]],dtype=np.float32); out=model.transform(valid)
    assert np.isfinite(train).all() and np.isfinite(out).all()
    assert model.modelo is fitted and model.modelo.n_iter_ == iterations
    assert np.array_equal(out[np.isfinite(valid)],valid[np.isfinite(valid)])
    with pytest.raises(RuntimeError): electric['ImputadorIterativoOriginal']().transform(valid)

def raw_fixture(n=42):
    d={c: np.arange(n,dtype=float)+1 for c in NUMERICAS}
    d.update({c:['Graduation' if i%2 else 'Master' for i in range(n)] for c in CATEGORICAS})
    d.update({'ID':np.arange(n)+100,'Dt_Customer':['01-01-2020']*n,
              'Recency':np.arange(n),'Z_CostContact':[3]*n,'Z_Revenue':[11]*n})
    d['Income'][0]=np.nan; d['Income'][1]=-1000
    d['Education'][2]='   '; d['Dt_Customer'][3]='fecha-invalida'
    return pd.DataFrame(d)

@pytest.mark.parametrize('sep,bom', [('\t',False),(',',False),('\t',True)])
def test_csv_delimiters_bom_and_schema(hym,tmp_path,sep,bom):
    raw=raw_fixture(); path=tmp_path/'campaign.csv'
    raw.to_csv(path,index=False,sep=sep,encoding='utf-8-sig' if bom else 'utf-8')
    loaded=hym['cargar_datos'](path)
    assert set(REQUIRED).issubset(loaded.columns) and loaded.shape==raw.shape
    broken=tmp_path/'broken.csv'; broken.write_text('ID,Dt_Customer\n1,01-01-2020\n',encoding='utf-8')
    with pytest.raises(ValueError,match='faltan'): hym['cargar_datos'](broken)
    with pytest.raises(FileNotFoundError,match='No existe'): hym['cargar_datos'](tmp_path/'missing.csv')

def test_csv_locator_reports_missing_and_prefers_explicit_path(hym,tmp_path,monkeypatch):
    target=tmp_path/'campaign.tsv'; target.write_text('ready',encoding='utf-8')
    hym['CFG']['ruta_csv']=str(target)
    assert hym['localizar_csv']()==target
    missing=tmp_path/'does-not-exist.csv'
    hym['CFG']['ruta_csv']=str(missing)
    with pytest.raises(FileNotFoundError,match='ruta CSV solicitada') as exc: hym['localizar_csv']()
    assert str(missing) in str(exc.value)
    monkeypatch.chdir(tmp_path)
    (tmp_path/'marketing_campaign.csv').write_text('alternativo',encoding='utf-8')
    with pytest.raises(FileNotFoundError,match='does-not-exist.csv'): hym['localizar_csv']()

def test_raw_audit_is_descriptive_and_non_mutating(hym):
    raw=raw_fixture(); raw['Income']=raw['Income'].astype(object); raw.loc[5,'Income']='dato-invalido'; raw.loc[6,'Education']='   '; raw.loc[7,'Education']=''
    raw=pd.concat([raw,raw.iloc[[0]]],ignore_index=True); before=raw.copy(deep=True)
    quality,stats=hym['auditar_dataset'](raw)
    pd.testing.assert_frame_equal(raw,before)
    assert quality.loc[quality.columna.eq('Income'),'nulos'].iloc[0]==2
    assert quality.loc[quality.columna.eq('Income'),'pct_nulos'].iloc[0]>0
    assert quality.loc[quality.columna.eq('Education'),'vacios_o_espacios_texto'].iloc[0]==3
    assert quality.loc[quality.columna.eq('Income'),'no_convertibles_numericos'].iloc[0]==1
    assert 'Income' in stats.index and raw.ID.duplicated().sum()==1

def test_dataset_cleaning_is_conservative_and_auditable(hym):
    raw=raw_fixture(); raw['Income']=raw['Income'].astype(object); raw.loc[5,'Income']='dato-invalido'; raw.loc[6,'Income']=np.inf; raw.loc[7,'Income']=-np.inf
    raw=pd.concat([raw,raw.iloc[[4]]],ignore_index=True); input_copy=raw.copy(deep=True)
    clean,report=hym['preparar_dataset'](raw)
    pd.testing.assert_frame_equal(raw,input_copy)
    assert len(clean)==41 and clean.ID.is_unique and clean.Dt_Customer.notna().all()
    assert report['duplicados_exactos_eliminados']==1 and report['fechas_invalidas_eliminadas']==1
    assert clean.loc[clean.ID.eq(101),'Income'].iloc[0]==-1000
    assert clean.loc[clean.ID.eq(102),'Education'].iloc[0]=='__FALTANTE__'
    assert clean.Income.isna().sum()==4 and report['conversiones_numericas_invalidas']['Income']==1
    assert report['no_finitos_convertidos_a_nan']['Income']==2
    broken=pd.concat([raw_fixture(),raw_fixture().iloc[[0]].assign(Income=999)],ignore_index=True)
    with pytest.raises(ValueError,match='IDs repetidos'): hym['preparar_dataset'](broken)
    too_small=raw_fixture(10)
    with pytest.raises(ValueError,match='se requieren al menos 40'): hym['preparar_dataset'](too_small)

def test_cohort_summary_and_shared_plot_bins(hym,monkeypatch):
    monkeypatch.setattr(plt,'show',lambda *_args,**_kwargs:None)
    histogram_inputs=[]; original_hist=plt.Axes.hist
    def track_hist(self,values,*args,**kwargs):
        histogram_inputs.append(np.asarray(values,dtype=float).copy())
        return original_hist(self,values,*args,**kwargs)
    monkeypatch.setattr(plt.Axes,'hist',track_hist)
    a=pd.DataFrame({'Income':[1.,2.,np.nan,3.],'MntWines':[4,4,4,4],
        'MntMeatProducts':[1,2,3,4],'NumWebPurchases':[1,2,3,4],
        'Education':['A','A','A','A'],'Marital_Status':['S','S','S','S']})
    b=pd.DataFrame({'Income':[5.,np.nan],'MntWines':[8,9],'MntMeatProducts':[5,6],
        'NumWebPurchases':[7,8],'Education':['B','B'],'Marital_Status':['M','M']})
    summary=hym['resumir_cohortes'](a,b).set_index('variable')
    assert summary.loc['Income','n_total_ref']==4 and summary.loc['Income','n_validos_ref']==3
    assert summary.loc['Income','n_faltantes_cur']==1 and summary.loc['Income','pct_faltantes_cur']==50
    a_plot=pd.DataFrame({'Income':[1.,2.,np.nan,3.,np.inf],'MntWines':[4,4,4,4,4],
        'MntMeatProducts':[1,2,3,4,5],'NumWebPurchases':[1,2,3,4,5],
        'Education':['A']*5,'Marital_Status':['S']*5})
    b_plot=pd.DataFrame({'Income':[5.,np.nan,6.,-np.inf],'MntWines':[8,9,10,11],
        'MntMeatProducts':[5,6,7,8],'NumWebPurchases':[7,8,9,10],
        'Education':['B']*4,'Marital_Status':['M']*4})
    diag=hym['graficar_eda_cohortes'](a_plot,b_plot,{'mostrar_graficos':True,'max_graficos_por_metodo':4})
    assert len(diag['bins'])==4 and np.array_equal(diag['bins']['MntWines'],diag['bins']['MntWines'])
    assert np.array_equal(histogram_inputs[0],np.array([1.,2.,3.]))
    assert np.array_equal(histogram_inputs[1],np.array([5.,6.]))
    assert all(np.isfinite(values).all() for values in histogram_inputs)
    cats=diag['proporciones']['Education']
    assert 'A' in cats.index and 'B' in cats.index and cats.loc['A','Comparación']==0
    assert cats['Referencia'].sum()==pytest.approx(1) and cats['Comparación'].sum()==pytest.approx(1)
    plt.close('all')
