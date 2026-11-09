# Laboratorio de forecasting — alquiler de bicicletas en Washington D. C. (2011–2012)

## Propósito y reglas de trabajo

Esta guía convierte los requerimientos de `Sesion_forecasting.pdf` y los recursos SQL/DAX del laboratorio en tareas verificables. El informe analiza alquileres horarios y usa las predicciones precalculadas disponibles. **En esta etapa no se entrenan modelos de ML.**

`Bike_Sharing_Base.pbix` ya contiene las seis páginas `Corporativo`, `Mapa de calor`, `Tendencia`, `Correlación`, `Modelos` y `Predicción`, además de algunos visuales iniciales. Cada integrante debe:

1. Abrir una copia del PBIX base y guardarla inmediatamente con el nombre individual que le corresponde.
2. Editar la página que ya existe y que tiene asignada; conservar, configurar o reemplazar los visuales iniciales según estas instrucciones.
3. Crear una página solo si la asignada realmente falta. No duplicar la primera página como paso rutinario.
4. No editar páginas asignadas a otros integrantes ni guardar sobre el PBIX base o el archivo de otro compañero.

Los datos están importados en el PBIX. Trabaja con el modelo de la plantilla; no cambies el servidor local, no vuelvas a importar tablas, no ejecutes scripts SQL, ni cambies esquema, relaciones o medidas comunes sin coordinarlo con quien integra.

## Referencias y modelo común

- Requisitos académicos: `Sesion_forecasting.pdf`, en la carpeta raíz del laboratorio.
- Definiciones SQL: `1.- dim_fact.sql`, `2.- dim_fecha.sql`, `4.- view.sql` y `Bike_Sharing_Instalacion/sql/`.
- Medidas DAX y guía de visuales: `Bike_Sharing_Instalacion/PowerBI/DAX/Medidas.dax` y `PAGINAS_Y_VISUALES.md`.
- Validaciones de conteos, rangos y claves: `Bike_Sharing_Instalacion/outputs/diagnostico_sql.json` y `REPORTE_SQL.md`.

### Visuales iniciales que ya están en el PBIX base

| Página | Visuales iniciales observados | Acción general |
| --- | --- | --- |
| `Corporativo` | Tarjeta, gráfico de líneas y matriz | Poma los configura para los roles indicados y añade los visuales que faltan. |
| `Mapa de calor` | Matriz | Marco la configura con los campos y el gradiente pedidos. |
| `Tendencia` | Gráfico de líneas y segmentador | Aldana configura el segmentador y sustituye/recrea como Python los gráficos requeridos; añade el navegador. |
| `Correlación` | Dispersograma | Marco lo sustituye por ACF y PACF; el PDF no pide dispersograma. |
| `Modelos` | Gráfico de líneas y segmentador | Brickzon adapta el filtro y sustituye/configura la línea como visual Python de comparación TEST. |
| `Predicción` | Gráfico de líneas | Vera lo configura como visual Python del pronóstico futuro y añade la tabla y segmentadores que falten. |

Los tipos listados se comprobaron en el contenido PBIR embebido del PBIX base. Antes de borrar un visual, verifica que sea el visual inicial citado y no un avance propio. No elimines visuales de otra página.

### Tablas, columnas y relaciones

| Objeto | Columnas relevantes |
| --- | --- |
| `DIM_FECHA` | `FECHA_ID`, `FECHA`, `DIA_MES`, `DIA_SEMANA`, `NOMBRE_DIA_SEMANA`, `NOMBRE_MES`, `MES_ANIO`, `ANIO` |
| `DIM_HORA` | `HORA_ID`, `HORA`, `HORA_INT` |
| `DIM_WEATHER` | `WEATHER_ID`, `WEATHER` |
| `vw_FACT_BIKE_SHARING` | `FECHA_ID`, `HORA_ID`, `WEATHER_ID`, `USERS`, `TIPO_CONJUNTO`, `ID_PREDICCION`, `DATE_TIME` |
| `vw_PREDICCION_BIKE_SHARING` | `ID_PREDICCION`, `MODEL_FORECASTER`, `MODEL_EXOGENEAS`, `MODEL_EXOGENEAS_CatBoost`, `MODEL_EXOGENEAS_LGBM`, `MODEL_EXOGENEAS_LSTM` |
| `vw_FORECASTING_BIKE_SHARING` | `DATE_TIME`, `FECHA_ID`, `HORA_ID`, `MODEL_FORECASTER`, `MODEL_EXOGENEAS`, `MODEL_EXOGENEAS_LGBM`, `MODEL_EXOGENEAS_CatBoost`; LSTM sin valores futuros |

Relaciones esperadas y comprobadas en el modelo: `DIM_FECHA[FECHA_ID]` con fact y forecasting; `DIM_HORA[HORA_ID]` con fact y forecasting; `DIM_WEATHER[WEATHER_ID]` con fact; `vw_PREDICCION_BIKE_SHARING[ID_PREDICCION]` (lado 1, clave única) con `vw_FACT_BIKE_SHARING[ID_PREDICCION]` (lado muchos). Conserva las relaciones activas y su dirección ya definida en la plantilla. Para Predicción, Año/Mes deben filtrar forecast por las relaciones de `FECHA_ID`; no crees otra relación.

`vw_FACT_BIKE_SHARING[TIPO_CONJUNTO]` vale `TRAIN` o `TEST`. La vista `vw_PREDICCION_BIKE_SHARING` aporta las predicciones de evaluación TEST; la vista `vw_FORECASTING_BIKE_SHARING` aporta el horizonte futuro. Nunca las cruces para presentar reales y predicciones de un mismo periodo si no comparten la misma fecha/hora observada.

### Medidas comunes ya documentadas

Revisa primero si existen en el PBIX. `PowerBI/DAX/Medidas.dax` define `Usuarios`, `Total registros`, `Promedio de alquileres por hora`, `Usuarios mes`, `Usuarios acumulados año`, `Usuarios mes anterior`, `Variación mensual %` y `MAE Forecaster`, `MAE Exogenas`, `MAE CatBoost`, `MAE LightGBM`, `MAE LSTM`. Las medidas MAE iteran filas fact TEST y recuperan el modelo con `RELATED`; dependen de la relación por `ID_PREDICCION`.

## Preparación compartida para visuales Python

Aplica esta sección a Aldana (Tendencia), Marco (Correlación) y Brickzon/Vera cuando usen los visuales Python descritos abajo.

1. Instala una distribución local de Python compatible con Power BI Desktop y con `pandas`, `matplotlib` y, para ACF/PACF, `statsmodels`.
2. Instala las dependencias en el intérprete que usará Desktop. Desde PowerShell, por ejemplo:

   ```powershell
   py -m pip install pandas matplotlib
   py -m pip install statsmodels
   ```

   `statsmodels` requiere `numpy` y pip instala su dependencia. Los scripts actuales no importan `numpy` directamente; instálalo por separado solo si el intérprete muestra que falta (`py -m pip install numpy`).
3. En Power BI Desktop abre **Archivo > Opciones y configuración > Opciones > Scripts de Python**, selecciona la carpeta/intérprete correcto y confirma. Reinicia Desktop si la configuración lo solicita. Al crear la primera visual Python, acepta **Enable**. No cambies el intérprete después de instalar paquetes sin repetir la comprobación.
4. En cada visual, agrega los campos indicados en su sección al área `Values`. Para fechas, selecciona la columna sin jerarquía automática. Power BI entrega al script un `dataset` tabular y elimina filas duplicadas en el conjunto seleccionado: incluye la granularidad necesaria (`DATE_TIME`) y no agregues columnas que colapsen observaciones.
5. Pega el script de la sección o el recurso Python indicado y pulsa ejecutar después de cada cambio. Si aparece error, confirma primero el intérprete configurado, paquetes instalados, nombres de columnas y valores nulos/duplicados; registra el mensaje exacto. No rellenes huecos con predicciones inventadas.

Dependencias directas: Tendencia usa `pandas` y `matplotlib`; Correlación requiere además `statsmodels`. `numpy` es dependencia transitiva de `statsmodels` para los scripts actuales. Las librerías y la ruta Python son locales y no quedan garantizadas por compartir el PBIX.

## Asignación definitiva

| Integrante | Página(s) | Archivo de entrega |
| --- | --- | --- |
| Poma | Corporativo | `Bike_Sharing_Poma.pbix` |
| Marco | Mapa de calor y Correlación | `Bike_Sharing_Marco.pbix` |
| Aldana | Tendencia | `Bike_Sharing_Aldana.pbix` |
| Brickzon | Modelos | `Bike_Sharing_Brickzon.pbix` |
| Vera | Predicción | `Bike_Sharing_Vera.pbix` |

## Instrucciones por integrante

### Poma — Corporativo

**Objetivo.** Resumen temporal ejecutivo del Ejercicio 1.

**Preparación y visuales existentes.** Edita la página existente `Corporativo`. Marca `DIM_FECHA` como tabla de fechas con `FECHA`; ordena `NOMBRE_MES` por `MES_ANIO` (esta columna es el número de mes). Conserva y configura la tarjeta, línea y matriz iniciales; añade los segmentadores, la segunda línea, la tarjeta que falte y el título. Crea un visual solo si no hay uno utilizable para ese rol. Añade el cuadro de texto `EJERCICIO 1`.

**Segmentadores obligatorios.** `DIM_FECHA[ANIO]`, título `Año`, y `DIM_FECHA[NOMBRE_MES]`, título `Mes`. Ambos desplegables, encabezado del segmentador desactivado y opción de selección múltiple/Seleccionar todo activada. Ordena meses cronológicamente. El PDF pide Top N para esos filtros mediante el identificador de fecha relacionado con fact, pero no da un N reproducible: conserva el N si el PBIX realmente ya lo tiene; si no, deja solo ese valor anotado como pendiente y continúa con el resto de la página.

**Visualizaciones obligatorias.**

1. **Matriz mensual** (configura la existente). Filas `DIM_FECHA[NOMBRE_MES]` (alias `Mes`). Valores académicos del PDF `.user_moth` (alias `Usuarios`), `.users_prev_month` (alias `Usuarios mes previo`) y `.%var_users_month` (alias `%Variación`). Estilo `Minimal`; barras de datos en `%Variación`; positivo `#A0D1FF`. El PDF da el color negativo como `##EFB5B9en`, que es inválido; no lo copies literalmente, conserva el color predeterminado o usa un valor hexadecimal válido y registra cuál. Formatea conteos con separador de miles y variación como porcentaje.
2. **Línea de comparación mensual** (configura la existente). X: `DIM_FECHA[NOMBRE_MES]`; valores `.user_moth` y `.users_prev_month`, alias `Usuarios` y `Usuarios mes previo`. Quita los títulos de ejes. Título `Comparación de tendencias Año Actual - previo`.
3. **Línea de tendencia y promedio**. Añade una línea si no hay otra utilizable. X: `DIM_FECHA[FECHA]`; valores `.sum_users` y `.promedio_movil`; título `Tendencia y promedio móvil`. Usa fecha ordenada, no jerarquía automática.
4. **Tarjeta anual** (configura la inicial si cumple). Valor `.users_total_year`; sin etiqueta de categoría; unidades de visualización `None`; tamaño 20; título `Total de usuarios del último año`, texto blanco y fondo `#118DFF`.
5. **Tarjeta de variación anual**: valor `.%var_users_yearr`; título `%Var Año previo`. El PDF proporciona la expresión académica, reproducida aquí sin corregirla:

   ```DAX
   .DIA = MAX('DIM_FECHA'[DIA_MES])
   .sum_users = SUM('vw_FACT_BIKE_SHARING'[USERS])
   .user_moth = TOTALMTD([.sum_users], 'DIM_FECHA'[FECHA])
   .users_total_year = TOTALYTD([.sum_users], 'DIM_FECHA'[FECHA])
   .users_prev_month = TOTALMTD([.user_moth], PREVIOUSMONTH('DIM_FECHA'[FECHA]))
   .users_prev_year = TOTALMTD([.users_total_year], PREVIOUSYEAR('DIM_FECHA'[FECHA]))
   .%var_users_yearr = DIVIDE([.users_total_year] - [.users_prev_year], [.users_prev_year])
   .%var_users_month = DIVIDE([.user_moth] - [.users_prev_month], [.users_prev_month])
   .promedio_movil = DIVIDE([.user_moth], [.DIA])
   ```

   Son las expresiones de origen académico del PDF, no las medidas técnicas compartidas/validadas. Reprodúcelas provisionalmente como aparecen, sin corregir su lógica ni cambiar sus nombres por fórmulas propias. Si Desktop rechaza el punto inicial del nombre de una medida, conserva el cuerpo de fórmula literal, usa un nombre que Desktop acepte y registra el cambio de nombre; no cambies la expresión. La utilización de `TOTALMTD` en `.users_prev_year`, descrita como acumulado anual, puede ser una errata; si no compila o el resultado parece incoherente, registra el error y deja esa validación pendiente, pero completa los demás visuales. El nombre “promedio móvil” tampoco describe claramente `.user_moth / .DIA`; preséntalo como la métrica del laboratorio, no como promedio móvil estadísticamente validado. `Medidas.dax` tiene cálculos compartidos de intención parecida, pero no reemplazan silenciosamente estas expresiones académicas.

**Interacción de Mes.** El PDF pide desactivar la interacción de `Mes` con la etiqueta/tarjeta, la matriz y la línea de comparación mensual. Configura sin interacción la matriz y la línea mensual. Si no queda claro cuál tarjeta llama “gráfico etiqueta”, no detengas el resto: prueba/documenta la interpretación aplicada a la tarjeta y deja solo ese punto anotado para integración. Prueba con más de un mes/año.

**Aceptación y presentación.** Título `EJERCICIO 1`, dos segmentadores, matriz, dos líneas y dos tarjetas. Mes ordenado; conteos enteros con miles; variación mensual porcentual; medida anual académica creada o error registrado, sin darla por validada si no se comprobó. Visuales alineados y no recortados. No sustituyas visuales requeridos por otros gráficos.

**Entrega:** `Bike_Sharing_Poma.pbix`.

### Marco — Mapa de calor

Edita la página existente `Mapa de calor`; no cambies su nombre ni dupliques Corporativo. Configura la matriz inicial y añade el texto `EJERCICIO 2`.

**Matriz obligatoria:** filas `DIM_FECHA[DIA_MES]`, columnas `DIM_HORA[HORA]`, valores suma de `vw_FACT_BIKE_SHARING[USERS]`. En el valor suma aplica formato condicional de color de fondo `Gradient`; ajusta el lienzo para ver la matriz completa. Ordena `HORA` por `HORA_INT` si hace falta.

El resultado representa suma de usuarios por número de día del mes y hora. El PDF especifica `DIA_MES`; la guía técnica interna propone día de semana. Usa provisionalmente `DIA_MES`, registra la discrepancia y no dejes que impida completar la matriz. No presentes la escala como valor individual por alquiler: es suma agregada.

**Aceptación:** una sola matriz requerida, todos los valores visibles, horas en orden y gradiente aplicado a la suma de `USERS`. No agregues un segundo mapa con día de semana sin autorización.

### Marco — Correlación

Edita la página existente `Correlación`; no agregues navegador de páginas aquí (el PDF lo ubica en Tendencia). Elimina/reemplaza el dispersograma inicial porque el PDF no lo solicita como entregable. Añade título `EJERCICIO 4`.

**Dos visuales Python obligatorios:** en cada uno agrega `vw_FACT_BIKE_SHARING[DATE_TIME]` y `vw_FACT_BIKE_SHARING[USERS]` a `Values`, con `DATE_TIME` como fecha/hora sin jerarquía. Primer visual: PACF, 72 rezagos, `method='ywm'`, título `Gráfico autocorrelación parcial`. Segundo visual: ACF, 72 rezagos, título `Gráfico autocorrelación`. Usa `PowerBI/scripts_python/PACF.py` y `ACF.py`; siguen la secuencia fecha/hora, orden temporal y frecuencia horaria, y limitan lags a `min(72, (n-1)//2)` para una serie corta.

Con las 17.544 horas completas se esperan 72 rezagos. Si un filtro heredado reduce la serie, el script puede mostrar menos; si falta una hora/valor, los scripts revisados dan error en vez de rellenar datos. Quita o documenta filtros que recorten horas antes de afirmar que se muestran los 72 rezagos. Configura títulos visibles, fondo azul (o color acordado) y texto blanco. Confirma que ambos visuales ejecutan con las librerías de la sección Python.

**Aceptación:** solo ACF y PACF como visuales requeridos, `DATE_TIME` único y `USERS` real; títulos legibles y rezagos comprobados. Guarda y registra cualquier error local de Python.

**Entrega de Marco:** `Bike_Sharing_Marco.pbix`.

### Aldana — Tendencia

Edita la página existente `Tendencia`. Tiene línea y segmentador iniciales: configura el segmentador y sustituye/configura la línea inicial como primer visual Python. Añade el segundo Python y el navegador de páginas. No dupliques la primera página.

**Segmentador obligatorio:** `vw_FACT_BIKE_SHARING[TIPO_CONJUNTO]`, desplegable, selección múltiple/Seleccionar todo; encabezado oculto y título general `Conjunto`.

**Visual Python 1 — perfil semanal:** agrega `DATE_TIME`, `DIM_FECHA[DIA_SEMANA]`, `USERS` y `DIM_HORA[HORA]` a `Values`. Calcula media, cuantil 25 y cuantil 75 de `USERS` por día de semana y hora. Título interno `Promedio de usuarios a largo de la semana`; X `Día y hora de la semana`; Y `Número de usuarios`; agrega leyenda. El PDF pide etiquetas Mon–Sun, pero el SQL define `DIA_SEMANA` como 1=domingo, 2=lunes, ..., 7=sábado. Reordena los grupos como 2,3,4,5,6,7,1 antes de poner etiquetas Mon–Sun, para que no se etiquete domingo como lunes. Ejecuta el script tras cada cambio.

Para implementar esa reordenación en Python, calcula primero las series agrupadas por `DIA_SEMANA` y `HORA`, reindexa los resultados en orden `[2, 3, 4, 5, 6, 7, 1]` y conserva las horas en orden ascendente. En cada serie usa `plot`; coloca los ticks en `0, 24, 48, 72, 96, 120, 144` y los rótulos Mon–Sun. Usa `DATE_TIME` para conservar la fila horaria única en el dataframe, aunque la agregación final sea por día y hora.

```python
import pandas as pd
import matplotlib.pyplot as plt

dataset["DATE_TIME"] = pd.to_datetime(dataset["DATE_TIME"], errors="raise")
order = [2, 3, 4, 5, 6, 7, 1]  # SQL: lunes...sábado, domingo
hours = sorted(dataset["HORA"].dropna().unique())
idx = pd.MultiIndex.from_product([order, hours], names=["DIA_SEMANA", "HORA"])
users = dataset.groupby(["DIA_SEMANA", "HORA"])["USERS"]
mean = users.mean().reindex(idx).reset_index(drop=True)
q25 = users.quantile(0.25).reindex(idx).reset_index(drop=True)
q75 = users.quantile(0.75).reindex(idx).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 4))
mean.plot(ax=ax, label="Promedio")
q25.plot(ax=ax, linestyle="dashed", color="gray", label="Cuantil 25")
q75.plot(ax=ax, linestyle="dashed", color="silver", label="Cuantil 75")
ax.set(title="Promedio de usuarios a largo de la semana",
       xticks=[i * 24 for i in range(7)],
       xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
       xlabel="Día y hora de la semana", ylabel="Número de usuarios")
ax.legend()
plt.show()
```

**Visual Python 2 — serie horaria:** agrega `DATE_TIME` y `USERS`, ordenados por fecha/hora, y grafica `USERS`; título dentro del gráfico `Número de usuarios`. Desactiva el título del contenedor Power BI para evitar duplicarlo.

```python
import pandas as pd
import matplotlib.pyplot as plt

dataset["DATE_TIME"] = pd.to_datetime(dataset["DATE_TIME"], errors="raise")
dataset = dataset.sort_values("DATE_TIME")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(dataset["DATE_TIME"], dataset["USERS"])
ax.set(title="Número de usuarios")
plt.show()
```

**Discrepancia de columnas en el PDF:** una línea dice conservar `DATE_TIME` y `HORA`, pero el script que sigue ejecuta `dataset['USERS'].plot()`. Para que el código funcione, usa `DATE_TIME` y `USERS` en este segundo visual; `HORA` no sustituye la serie de usuarios. Registra esta diferencia, sin afirmar que el PDF y el código coinciden.

**Navegador obligatorio:** Inserta **Insertar > Botones > Navegador > Navegador de páginas** en Tendencia. Deshabilita `Corporativo` y `Mapa de calor` en las opciones del navegador y deja las otras páginas navegables. Verifícalo de nuevo en el PBIX final.

**Aceptación:** promedio y cuantiles por las 168 combinaciones día/hora, orden semanal correcto; serie de `USERS` horaria; filtro TRAIN/TEST operativo; navegador con las dos páginas excluidas. Instala y configura `pandas` y `matplotlib` según la sección Python. No declares que se muestra la semana completa si hay filtros que la recortan.

**Entrega:** `Bike_Sharing_Aldana.pbix`.

### Brickzon — Modelos

**Objetivo.** Visualizar la comparación entre alquileres reales del periodo TEST y predicciones precalculadas de los mismos instantes (Ejercicio 5). No entrenar modelos.

El PBIX base contiene una línea y un segmentador. Configura el segmentador como `TIPO_CONJUNTO = TEST`; convierte/reemplaza la línea inicial por un visual Python, que es el tipo pedido por el PDF. Añade cuadro de texto `EJERCICIO 5` si falta.

**Origen y alineamiento comprobado.** Usa `vw_FACT_BIKE_SHARING` para `DATE_TIME`, `USERS`, `TIPO_CONJUNTO` e `ID_PREDICCION`; usa `vw_PREDICCION_BIKE_SHARING` para las cinco columnas MODEL. SQL genera el ID TEST a partir de `FECHA_ID` y `HORA_ID`; la vista de predicciones genera el mismo ID desde `DATE_TIME` (fecha `yyyyMMdd` + hora). La comprobación SQL del laboratorio encontró claves únicas en la vista de predicciones, 2.928 filas TEST, 2.928 `DATE_TIME` distintos, 2.928 claves emparejadas y cero faltantes/duplicadas. La relación activa es `vw_PREDICCION_BIKE_SHARING[ID_PREDICCION]` 1 → `vw_FACT_BIKE_SHARING[ID_PREDICCION]` *.

No uses directamente una columna de la tabla de predicciones como serie suponiendo que el filtro temporal del lado fact se propaga hacia el lado 1. Usa medidas iteradas desde las filas fact; el propio DAX MAE del proyecto demuestra el patrón compatible: `AVERAGEX(FILTER(fact,...), ... RELATED(pred[MODEL]))`. Crea una medida de valor TEST por modelo; por ejemplo:

```DAX
TEST MODEL_FORECASTER =
SUMX(
    FILTER(
        'vw_FACT_BIKE_SHARING',
        'vw_FACT_BIKE_SHARING'[TIPO_CONJUNTO] = "TEST"
    ),
    RELATED('vw_PREDICCION_BIKE_SHARING'[MODEL_FORECASTER])
)
```

Repite la medida para `MODEL_EXOGENEAS`, `MODEL_EXOGENEAS_CatBoost`, `MODEL_EXOGENEAS_LGBM` y `MODEL_EXOGENEAS_LSTM`, cambiando el nombre de la medida y la columna de `RELATED`. Esto devuelve la suma de predicciones relacionadas con las filas fact del contexto actual; como la validación encontró un hecho por cada `DATE_TIME` TEST, en cada punto temporal devuelve la predicción de esa misma hora. La relación 1:* y unicidad comprobada permiten `RELATED` en la fila iterada; no se depende de propagación inversa del filtro.

**Verificación previa obligatoria del emparejamiento:** filtra una tabla temporal de validación a `TIPO_CONJUNTO = TEST` y agrega `DATE_TIME`, `ID_PREDICCION`, `USERS` y las cinco medidas TEST. Confirma 2.928 filas, 2.928 fechas/IDs distintos, medida no vacía por cada modelo y que para cada fila su `ID_PREDICCION` coincide con la clave del modelo. El número 2.928 aplica sin filtros adicionales de fecha/hora; si se filtra el periodo, el conteo esperado disminuye. Las cifras de control SQL están en `outputs/diagnostico_sql.json` y `outputs/REPORTE_SQL.md`. Quita la tabla temporal antes de entregar, salvo que el profesor la solicite.

**Visual de comparación obligatorio:** crea parámetro de campos `Modelos` con las cinco medidas `TEST MODEL_*`, no con columnas crudas de la vista ni con las columnas de pronóstico futuro. El parámetro de medidas va en Values; configúralo como selector desplegable de selección única. En el visual Python agrega `vw_FACT_BIKE_SHARING[DATE_TIME]`, `vw_FACT_BIKE_SHARING[USERS]` y el parámetro de medidas. Usa `DATE_TIME` de la fact como eje/índice; `USERS` como real y la medida seleccionada como predicción. Ordena por tiempo y grafica ambas series con leyenda `Real` y `Predicción seleccionada`; título del objeto `Evaluación de Modelos predictivos`, fondo azul y texto blanco. En Python, selecciona el nombre de columna que no sea `DATE_TIME` ni `USERS` como predicción; no supongas que se llamará `MODEL_*` si Power BI la etiqueta con el nombre de la medida. Instala `pandas` y `matplotlib` según sección Python.

Esquema del script para el visual (el nombre del campo dinámico proviene del parámetro y debe comprobarse en el dataframe del visual):

```python
import pandas as pd
import matplotlib.pyplot as plt

dataset["DATE_TIME"] = pd.to_datetime(dataset["DATE_TIME"], errors="raise")
dataset = dataset.sort_values("DATE_TIME")
pred_cols = [c for c in dataset.columns if c not in {"DATE_TIME", "USERS"}]
if len(pred_cols) != 1:
    raise ValueError(f"Se esperaba una medida TEST seleccionada; columnas: {pred_cols}")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(dataset["DATE_TIME"], dataset["USERS"], label="Real")
ax.plot(dataset["DATE_TIME"], dataset[pred_cols[0]], label="Predicción seleccionada")
ax.set(title="Evaluación de Modelos predictivos", xlabel="Fecha y hora", ylabel="Usuarios")
ax.legend()
plt.show()
```

No uses `drop_duplicates` para ocultar duplicados. Si el visual recibe más de una fila por `DATE_TIME`, detente y revisa la relación/agrupación; los datos de esta vista deben dar una observación TEST por hora.

**Diferencia académica/SQL:** el Ejercicio 5 del PDF apunta el parámetro a `vw_FORECASTING_BIKE_SHARING` y luego describe una comparación con reales. Esa vista es futuro 2013 y no comparte observaciones reales; la vista que contiene cinco predicciones de evaluación es `vw_PREDICCION_BIKE_SHARING`. Para satisfacer la comparación real/predicción sin cruzar horizontes, sigue el procedimiento TEST anterior y deja esta discrepancia documentada. No presentes el forecast futuro como evaluación.

**Opcional, no exigido por el PDF:** añadir tarjetas/tabla con `MAE Forecaster`, `MAE Exogenas`, `MAE CatBoost`, `MAE LightGBM` y `MAE LSTM` del DAX común. Si se agrega, rotular TEST; no sustituye ni altera la línea comparativa obligatoria.

**Aceptación:** visual Python con real y predicción alineados por hora; selector cambia de modelo; tabla temporal de verificación confirma los 2.928 pares sin filtros adicionales; no hay agregación que repita una única predicción para varias horas. Distingue TEST de futuro.

**Entrega:** `Bike_Sharing_Brickzon.pbix`.

### Vera — Predicción

Edita la página existente `Predicción`. Configura/reemplaza la línea inicial como visual Python del horizonte futuro. Añade los segmentadores si no existen, la tabla y el título `EJERCICIO 6` si falta.

**Segmentadores Año/Mes:** usa `DIM_FECHA[ANIO]` y `DIM_FECHA[NOMBRE_MES]`, relacionados con forecasting por `FECHA_ID`; estilo desplegable. Comprueba filtrando un año/mes con resultados que las fechas visibles cambian. El PDF dice que el Top N interno debe usar `vw_FORECASTING_BIKE_SHARING[FECHA_ID]` en lugar del ID de fact. Si la plantilla ya contiene N, consérvalo y cambia la columna “By” a `FECHA_ID` del forecast. Si no existe configuración N o falta el valor N, no inventes uno: registra solo ese parámetro como pendiente y completa el selector, gráfico y tabla.

**Parámetro y visual futuro:** crea parámetro de campos `Forecasting` con los cuatro campos poblados de `vw_FORECASTING_BIKE_SHARING`: `MODEL_FORECASTER`, `MODEL_EXOGENEAS`, `MODEL_EXOGENEAS_LGBM`, `MODEL_EXOGENEAS_CatBoost`. Segmentador desplegable, selección única. En el visual Python agrega `vw_FORECASTING_BIKE_SHARING[DATE_TIME]` y el campo `Forecasting`; usa la fecha como índice horario, detecta la columna seleccionada y grafícala. Título `Evaluación de Modelos predictivos`, fondo azul y texto blanco.

El script del visual futuro recibe `DATE_TIME` y la columna seleccionada por `Forecasting`. Convierte la fecha, ordena por tiempo, identifica la única columna distinta de `DATE_TIME` y la grafica; no agregues una columna `USERS` que no existe en la vista.

```python
import pandas as pd
import matplotlib.pyplot as plt

dataset["DATE_TIME"] = pd.to_datetime(dataset["DATE_TIME"], errors="raise")
dataset = dataset.sort_values("DATE_TIME")
forecast_cols = [c for c in dataset.columns if c != "DATE_TIME"]
if len(forecast_cols) != 1:
    raise ValueError(f"Se esperaba un modelo futuro seleccionado: {forecast_cols}")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(dataset["DATE_TIME"], dataset[forecast_cols[0]], label=forecast_cols[0])
ax.set(title="Evaluación de Modelos predictivos", xlabel="Fecha y hora", ylabel="Pronóstico")
ax.legend()
plt.show()
```

**Límite del PDF y de los datos:** el ejemplo de código del profesor también intenta graficar `USERS`, pero `vw_FORECASTING_BIKE_SHARING` no tiene observaciones reales `USERS`; los reales terminan el 31-12-2012 y el forecast empieza el 01-01-2013. En esta página muestra el pronóstico futuro sin línea real superpuesta. No uses `vw_PREDICCION_BIKE_SHARING` para completar esa serie ni describas predicciones futuras como TEST.

**Tabla obligatoria:** visual de tipo **Tabla**, columnas `vw_FORECASTING_BIKE_SHARING[DATE_TIME]` y campo de parámetro `Forecasting`; estilo `Minimal`, orden ascendente por fecha/hora. No uses matriz salvo autorización del profesor.

**Verificación del horizonte:** sin filtros de fecha adicionales, la tabla y el gráfico deben cubrir 2.160 horas desde `2013-01-01 00:00` hasta `2013-03-31 23:00`; cada uno de los cuatro modelos del parámetro tiene 2.160 valores no nulos. Los filtros Año/Mes pueden reducir los conteos. La vista tiene una columna LSTM, pero cero valores LSTM para el horizonte futuro; no la agregues ni rellenes.

**Aceptación:** selector cambia entre cuatro series; gráfico y tabla muestran la misma serie/fechas; filtros año/mes afectan el forecast; rango y conteo completos se verifican antes de filtros. El único pendiente admisible es un N que no figure en la plantilla ni en instrucciones del profesor.

**Entrega:** `Bike_Sharing_Vera.pbix`.

## Seguimiento de entregables

Estado inicial: **Pendiente**. Cambia a completado solo tras abrir y revisar el PBIX individual. Los pendientes académicos se anotan sin detener visuales independientes.

| Integrante | Página(s) | Visuales obligatorios | Archivo | Criterio resumido | Estado inicial |
| --- | --- | --- | --- | --- | --- |
| Poma | Corporativo | Segmentadores Año/Mes; matriz; 2 líneas; tarjeta acumulado año; tarjeta variación anual | `Bike_Sharing_Poma.pbix` | Fórmulas del PDF registradas, filtros/orden e interacciones probados | Pendiente |
| Marco | Mapa de calor y Correlación | Matriz DIA_MES×HORA con gradiente; Python PACF y ACF | `Bike_Sharing_Marco.pbix` | Página mapa completa; dos visuales Python ejecutan y lags verificados | Pendiente |
| Aldana | Tendencia | 2 visuales Python; slicer conjunto; navegador | `Bike_Sharing_Aldana.pbix` | Perfil 168 grupos ordenado; serie horaria, filtro y navegador probados | Pendiente |
| Brickzon | Modelos | Python real vs predicción TEST; selector de 5 medidas | `Bike_Sharing_Brickzon.pbix` | 2.928 emparejamientos verificados sin agregación errónea | Pendiente |
| Vera | Predicción | Python forecast; segmentadores Año/Mes; tabla | `Bike_Sharing_Vera.pbix` | 2.160 valores en horizonte futuro; LSTM excluido | Pendiente |

### Checklist individual

**Poma — Corporativo**

- [ ] Guardó su copia como `Bike_Sharing_Poma.pbix` y editó la página existente.
- [ ] Configuró calendario, orden de mes y segmentadores Año/Mes.
- [ ] Configuró matriz, comparación mensual, tendencia/promedio y ambas tarjetas.
- [ ] Usó fórmulas del PDF para promedio y variación anual; registró errores/semántica sin reemplazarlas.
- [ ] Aplicó interacciones identificables y anotó la interpretación pendiente de “gráfico etiqueta”.
- [ ] Probó filtros, formato y guardó/reabrió el PBIX.

**Marco — Mapa de calor y Correlación**

- [ ] Guardó como `Bike_Sharing_Marco.pbix` y editó ambas páginas existentes.
- [ ] Configuró matriz `DIA_MES × HORA`, suma de `USERS`, gradiente y orden de hora.
- [ ] Reemplazó el dispersograma por ACF/PACF Python; no agregó navegador a Correlación.
- [ ] Confirmó intérprete Python, dependencias, fechas únicas y rezagos representados.
- [ ] Registró la diferencia académica `DIA_MES` vs día de semana y guardó/reabrió.

**Aldana — Tendencia**

- [ ] Guardó como `Bike_Sharing_Aldana.pbix` y editó Tendencia existente.
- [ ] Configuró el segmentador `TIPO_CONJUNTO` y el navegador solo en esta página.
- [ ] Creó/configuró el perfil semanal y reordenó `DIA_SEMANA` 2,3,4,5,6,7,1.
- [ ] Creó/configuró la serie horaria con `DATE_TIME` y `USERS`.
- [ ] Instaló/configuró Python, `pandas`, `matplotlib`; registró la contradicción del PDF.
- [ ] Probó visuales/filtros y guardó/reabrió.

**Brickzon — Modelos**

- [ ] Guardó como `Bike_Sharing_Brickzon.pbix` y editó Modelos existente.
- [ ] Configuró filtro TEST y medidas de predicción iteradas con `RELATED`.
- [ ] Creó parámetro de medidas y visual Python con `DATE_TIME`, real y predicción seleccionada.
- [ ] Verificó temporalmente `ID_PREDICCION`, 2.928 filas, fechas únicas y cero predicciones faltantes.
- [ ] Documentó la discrepancia PDF sobre la vista futura; no entrenó modelos.
- [ ] Dejó MAE como opcional o la añadió claramente como complemento.
- [ ] Guardó/reabrió y probó al menos dos modelos del selector.

**Vera — Predicción**

- [ ] Guardó como `Bike_Sharing_Vera.pbix` y editó Predicción existente.
- [ ] Agregó Año/Mes si faltaban y confirmó que filtran Forecast por relación de fecha.
- [ ] Conservó N preconfigurado o registró únicamente el N ausente como pendiente.
- [ ] Configuró selector de cuatro modelos, visual Python y tabla (no matriz).
- [ ] Comprobó 2.160 horas entre enero y marzo de 2013 sin filtros restrictivos.
- [ ] Excluyó LSTM futuro y reales TEST de la página de forecast.
- [ ] Guardó/reabrió y probó el selector y filtros.

## Flujo Git del equipo

Todos trabajan directamente en `master`; no se crearán ramas individuales. Cada integrante publica solo su PBIX individual. Git no fusiona el contenido binario de PBIX. Sincroniza antes de empezar y antes de publicar; no uses `git push --force`.

Desde la raíz del repositorio, antes de editar:

```powershell
git status --short
git pull --rebase origin master
git status --short
```

Guarda como el archivo propio. Para publicarlo, agrega solo esa ruta, revisa el staging y luego confirma/publica:

```powershell
git add -- "Semana 12/Lab Alquiler de bicicletas en la ciudad de Washington D.C. durante los años 2011 y 2012/OUTPUT(POWERBI QUE PIDE EL PROFE)/Bike_Sharing_Marco.pbix"
git diff --cached --name-only
git commit -m "feat(powerbi): actualizar avance de Marco"
git pull --rebase origin master
git push origin master
```

Sustituye `Marco` por tu nombre. Si `git diff --cached --name-only` lista un archivo ajeno, retíralo del staging —sin borrar su contenido local— y revisa de nuevo:

```powershell
git restore --staged -- "ruta/al/archivo-ajeno"
git diff --cached --name-only
```

Si otro integrante publicó mientras trabajabas, el `pull --rebase` posterior al commit incorpora su commit antes del push. Si aparece un conflicto sobre el mismo PBIX, no elijas una versión ni uses force: detente y coordina con integración. No subas scripts, temporales, PBIX base ni archivos de otros. No se hacen commits ni pushes como parte de esta edición del README.

Archivos esperados (no se asume que ya existan): `Bike_Sharing_Base.pbix`, `Bike_Sharing_Poma.pbix`, `Bike_Sharing_Marco.pbix`, `Bike_Sharing_Aldana.pbix`, `Bike_Sharing_Brickzon.pbix`, `Bike_Sharing_Vera.pbix` y `Bike_Sharing_Final.pbix`.

## Integración final en Power BI Desktop

La integración requiere Desktop. Conserva los PBIX recibidos y crea `Bike_Sharing_Final.pbix` con **Guardar como** desde `Bike_Sharing_Base.pbix`.

1. Comprueba que están los cinco archivos individuales y que abren. En cada uno revisa únicamente la página asignada.
2. Conserva en el final un solo modelo importado con los seis objetos comunes y relaciones de base. No sustituyas el modelo completo por el de otro PBIX ni cambies la conexión local `Renato\SQLEXPRESS`.
3. Para cada página, usa la página destino existente del final. Como el final también parte de la base, no pegues encima visuales iniciales equivalentes: configura/reemplaza los marcadores iniciales con la versión final o pega solo los visuales adicionales y retira el duplicado inicial. En el PBIX de origen selecciona los visuales terminados y copia/pega en la página final correspondiente. Desktop permite copiar visuales de un informe a otro, pero la operación no fusiona el modelo ni debe asumirse que transfiere propiedades de página, filtros, interacciones, marcadores o dependencias. Reconfigura esas propiedades y consulta la [documentación oficial de copiar visuales entre informes](https://learn.microsoft.com/en-us/power-bi/create-reports/desktop-report-view#copy-and-paste-visuals-between-reports).
4. Revisa medidas y parámetros. Conserva una sola medida común. El parámetro de campos crea una tabla auxiliar: comprueba que la tabla/campos (o medidas para Modelos) existan en el modelo final; recrea el parámetro con los mismos nombres si la referencia no queda enlazada al pegar.
5. Comprueba relaciones activas, campos y cardinalidades; `DIM_FECHA` como tabla calendario y orden `NOMBRE_MES` por `MES_ANIO`.
6. Revisa filtros visuales/página/informe: TEST solo en Modelos; dimensiones fecha hacia Forecast en Predicción; Año/Mes; interacción de Mes en Corporativo; `TIPO_CONJUNTO` y navegador en Tendencia.
7. Orden de páginas: `Corporativo`, `Mapa de calor`, `Tendencia`, `Correlación`, `Modelos`, `Predicción`. Confirma nombres, visibilidad, títulos, formatos, tipografía, colores, números, fechas y que nada quede recortado.
8. Valida resultados: Corporativo con filtros; mapa completo; scripts Python ejecutados; Modelo emparejado por ID/hora y conteo 2.928 sin filtros restrictivos; Predicción con 2.160 horas 2013-01-01 a 2013-03-31, cuatro modelos no nulos. Anota cualquier filtro que reduzca los conteos.
9. Contrasta cada visual con `Sesion_forecasting.pdf` y las diferencias documentadas aquí. Guarda, cierra y vuelve a abrir el PBIX final antes de declararlo integrado.

Los PBIX son archivos binarios y Git no ofrece una fusión segura. Copiar/pegar visuales es una función soportada de Desktop, pero no es equivalente a transferir automáticamente toda la página ni a fusionar modelos. No se propone fusión automática PBIR: el flujo acordado entrega PBIX independientes y un final PBIX.

## Diferencias o parámetros pendientes, sin detener las demás tareas

1. **Top N:** el PDF no fija N claramente. Conservarlo si está realmente preconfigurado; si no, completar los demás visuales y registrar solo este valor pendiente.
2. **Interacción Mes en Corporativo:** “gráfico etiqueta” no identifica de forma inequívoca una tarjeta. Aplicar las interacciones no ambiguas y registrar la tarjeta interpretada.
3. **Mapa de calor:** PDF `DIA_MES`; guía interna día de semana. Usar `DIA_MES` provisionalmente.
4. **Promedio móvil/variación anual:** reproducir las expresiones del PDF sin corrección tácita; `TOTALMTD` en el acumulado de año y la denominación “promedio móvil” son académicamente ambiguos. Validar resultado antes de declararlas correctas.
5. **Modelos:** el PDF apunta al forecast futuro para comparar con reales, pero esos periodos no se solapan. Usar la vista TEST y alinear por `ID_PREDICCION`; registrar la discrepancia.
6. **Predicción futura:** el código del PDF agrega `USERS`, ausente del horizonte forecast. Mostrar el pronóstico futuro solo, sin reales superpuestos.
7. **Tendencia:** la descripción del segundo gráfico nombra `HORA`, pero el script requiere `USERS`; usar ambos campos que el script necesita (`DATE_TIME`, `USERS`) y dejar la discrepancia registrada.

## Fuera del alcance

- No entrenar modelos ni crear predicciones simuladas.
- No modificar/truncar/reimportar la base ni cambiar archivos SQL, DAX o Python.
- No sobrescribir el PBIX base o de otro integrante; no crear ramas ni usar `git push --force`.
- No declarar exitosa una comprobación de Desktop, Python, filtros o alineamiento que no se haya ejecutado.
