# Laboratorio de forecasting — Bike Sharing Washington D.C.

## Objetivo

Analizar el alquiler de bicicletas en Washington D.C. durante 2011–2012, explorar su comportamiento temporal y comparar las predicciones precalculadas entregadas por el profesor.

## Dataset y tecnologías

El dataset registra alquileres horarios y variables de calendario, clima y operación. Se usarán Power BI Desktop y archivos PBIX para el dashboard, SQL Server como origen de datos al preparar la plantilla, Git/GitHub para compartir avances y Python para algunos gráficos. Los datos quedan importados en el PBIX; los gráficos Python pueden requerir Python y bibliotecas locales en cada computadora.

## Empezar desde la plantilla

1. Sincroniza el repositorio con `master` y abre `Bike_Sharing_Base.pbix` en Power BI Desktop.
2. Usa **Guardar como** y crea tu archivo individual: `Bike_Sharing_Marco.pbix`.
3. Trabaja solo en las páginas que te corresponden. No guardes cambios sobre la plantilla base ni sobre el PBIX de otra persona.
4. Los datos de la plantilla deben visualizarse desde el contenido importado. No actualices ni cambies la conexión a `Renato\SQLEXPRESS`: esa instancia existe únicamente en la computadora de quien preparó el archivo.

## Páginas por integrante

| Integrante | Páginas asignadas |
| --- | --- |
| Marco | Mapa de calor y Correlación |
| Aldana | Tendencia |
| Brickzon | Modelos |
| Vera | Predicción |
| Poma | Corporativo |

Todos comienzan desde el mismo `Bike_Sharing_Base.pbix` y guardan su avance en su propio archivo.

## Publicar avances

Todos trabajan directamente en `master`; no se utilizarán ramas ni Pull Requests. Antes de empezar y antes de publicar, sincroniza tu copia local:

```powershell
git pull --rebase origin master
git status --short
```

Agrega únicamente tu PBIX, crea un commit con un mensaje descriptivo y publica:

```powershell
git add -- "Semana 12/Lab Alquiler de bicicletas en la ciudad de Washington D.C. durante los años 2011 y 2012/OUTPUT(POWERBI QUE PIDE EL PROFE)/Bike_Sharing_Marco.pbix"
git commit -m "feat(powerbi): actualizar avance de Marco"
git push origin master
```

En los comandos anteriores, sustituye `Marco` por tu nombre antes de ejecutarlos. Cada persona solo hace commits de su propio PBIX. Está prohibido usar `git push --force`. No agregues scripts SQL, PDF, notebooks, archivos temporales ni archivos de otros integrantes. Antes de publicar, comprueba que no haya cambios ajenos en `git status`.

## Reglas para evitar conflictos

- No se sobrescribe `Bike_Sharing_Base.pbix` ni el PBIX individual de otra persona.
- No cambies nombres de tablas, columnas, relaciones ni medidas comunes sin coordinarlo con el equipo.
- Sincroniza `master` antes de publicar tus cambios.
- Los PBIX son binarios: Git no puede fusionar visualizaciones automáticamente. Coordina la edición para evitar publicar simultáneamente cambios incompatibles.

## Integrar el informe final

La integración se hará en Power BI Desktop, copiando las visualizaciones de las páginas de cada PBIX individual al archivo final `Bike_Sharing_Final.pbix`. Revisa también las medidas nuevas, filtros de página o de informe y otras configuraciones: podrían necesitar trasladarse o recrearse manualmente. No crees el PBIX final hasta que el equipo termine sus páginas.