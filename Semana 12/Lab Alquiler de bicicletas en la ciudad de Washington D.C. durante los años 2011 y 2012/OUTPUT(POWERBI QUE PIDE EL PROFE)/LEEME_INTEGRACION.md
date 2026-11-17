# Bike Sharing — Informe final integrado (PBIP)

**Estado:** Integración programática realizada y auditada estructuralmente. **No verificada en Power BI Desktop.** No se modificó ningún proyecto original.

## Apertura

1. Extraer **todo** el ZIP a una carpeta nueva, sin mezclar con copias anteriores.
2. Abrir el archivo `Bike_Sharing_Final.pbip` en Power BI Desktop. Las carpetas `Bike_Sharing_Final.Report` y `Bike_Sharing_Final.SemanticModel` deben permanecer en el mismo directorio que el `.pbip`.
3. El modelo utiliza tablas importadas de SQL Server local `Renato\SQLEXPRESS`, base `DMT_BIKE_SHARING`. Si Desktop solicita actualizar, verificar que el servidor SQL esté accesible, las credenciales sean correctas y que se puedan importar los datos. **No se incluye `cache.abf` ni archivos locales `.pbi`.**
4. Verificar la configuración de Python de Power BI Desktop para los visuales Python de **Tendencia, Correlación, Modelos y Predicción**. Las dependencias exactas dependen de los scripts incorporados; las entregas anteriores utilizaron `pandas`, `numpy`, `matplotlib` y `statsmodels`.
5. Revisar las seis páginas en orden, ejecutar los filtros/segmentadores y confirmar que no haya errores, imágenes vacías ni gráficos mal vinculados.
6. Si todo funciona, guardar una copia mediante **Archivo > Guardar como** en formato `.pbix`, con nombre `Bike_Sharing_Final.pbix`.

## Páginas fusionadas

| Página | Procedencia | Visuales | Preservación |
|---|---|---:|---|
| Corporativo | Poma | 8 | Copia byte a byte |
| Mapa de calor | Marco | 8 | Copia byte a byte |
| Tendencia | Aldana | 6 | Copia byte a byte |
| Correlación | Marco | 7 | Copia byte a byte |
| Modelos | Brickzon | 4 | Copia byte a byte |
| Predicción | Vera | 6 | Copia byte a byte |

## Un único modelo semántico

- Se adoptó el modelo semántico de **Poma** para preservar sus medidas y configuraciones del panel Corporativo.
- Se añadieron las cinco medidas de Brickzon `TEST MODEL_*` en `DIM_FECHA`, **sin reemplazar las medidas académicas ni las columnas existentes de Poma**.
- Se añadió la tabla de parámetro de campos `Modelos` de Brickzon, y la tabla `Forecasting` de Vera, con las referencias correspondientes en `model.tmdl`.
- Se fusionaron entradas de metadatos lingüísticos `en-US` sin colisiones.
- Las relaciones compartidas y los nombres de las tablas/columnas base se preservaron desde Poma. Las páginas de los otros integrantes no se reescribieron.
- Se generaron identificadores nuevos para el informe y modelo final. La referencia `definition.pbir` apunta a `../Bike_Sharing_Final.SemanticModel`.

## Comprobaciones automáticas realizadas

- 6 páginas en orden y 39 visuales.
- 50 archivos JSON parseados sin error.
- Todas las páginas son idénticas, archivo por archivo, a la página correspondiente entregada por el integrante.
- Referencias explícitas de columnas y medidas de los visuales: ninguna referencia ausente en la definición del modelo.
- Parámetros `Modelos` y `Forecasting`: todos los 9 destinos `NAMEOF()` están definidos.
- Medidas: 26 medidas existentes de Poma en `vw_FACT_BIKE_SHARING` más las 5 medidas TEST de Brickzon en `DIM_FECHA`.
- Sin duplicación de modelos, cachés `.abf` ni configuraciones `.pbi` particulares de otro equipo.

## Validaciones pendientes en Power BI Desktop

- Comprobar que el modelo importado cargue correctamente y que sus consultas funcionen; el ZIP no incluye datos importados.
- Corporativo: tarjetas, matriz, medidas y año/mes (revisión de semántica de medidas académicas si corresponde).
- Mapa de calor: 24 horas de 00 a 23, días 1 a 31, suma de usuarios, gradiente y filtros.
- Tendencia: ambos gráficos Python, segmentador y navegador de páginas.
- Correlación: ACF y PACF, ambos visuales Python renderizados sin error y 72 rezagos en serie completa.
- Modelos: 2.928 observaciones TEST con predicciones alineadas por ID/hora; parámetro `Modelos` operable.
- Predicción: horizonte de 2.160 horas y cuatro series futuras; parámetro `Forecasting`, segmentadores y tabla, sin reales del horizonte futuro.
- Verificar orden del menú, interacciones, filtros y que no se recorten las páginas.

**Si se produce un error:** no reescribir todas las páginas. Registrar página y visual afectados, guardar el detalle técnico del error y probar primero las medidas y parámetros de ese visual. El paso de validez JSON a funcionamiento en Desktop debe demostrarse, no presumirse.

## Archivos

- `Bike_Sharing_Final.pbip`: entrada del proyecto.
- `Bike_Sharing_Final.Report/`: páginas, visuales y recursos del informe.
- `Bike_Sharing_Final.SemanticModel/`: definición TMDL unificada.
- `VALIDACION_INTEGRACION.json`: auditoría automática de la integración.

No se ha realizado ningún commit ni push y no se ha generado un PBIX; este último debe crearse en Power BI Desktop una vez verificado el PBIP.
