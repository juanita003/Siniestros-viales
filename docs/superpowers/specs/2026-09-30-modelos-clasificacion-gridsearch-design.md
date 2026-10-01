# Diseño: nuevo balanceo y modelos de clasificación con GridSearchCV

**Notebook:** `Preparacion_de_datos_Siniestros_Viales_Envigado.ipynb` (se ejecuta en Google Colab)
**Fecha:** 2026-09-30

## Objetivo

1. Reemplazar el balanceo actual (SMOTENC a 3 clases) por un problema **binario** SOLO DAÑOS vs HERIDOS.
2. Reemplazar la sección 11 (split 80/20) por hiperparametrización con `GridSearchCV` y validación cruzada (`cv=10`, `f1_macro`) para 6 modelos: Árbol, RandomForest, XGBoost, KNN, SVM y Red Neuronal.
3. Verificar el mejor modelo con un balanceo dentro de cada fold (enfoque C) para medir el sesgo del balanceo global.

## Decisiones

| Tema | Decisión |
|---|---|
| Variable objetivo | `GRAVEDAD` (no se renombra a `Alerta`) |
| Clase MUERTOS | Se elimina (0.3 %, unos 130 casos) |
| Balanceo | DAÑOS se reduce al 50 % (`RandomUnderSampler`) y HERIDOS sube con `SMOTENC` hasta igualarlo, para 50/50 (unos 34.8k registros) |
| Evaluación | Sin split 70/30 u 80/20: `GridSearchCV(cv=10, scoring='f1_macro')` sobre `data` balanceada |
| Verificación | El mejor modelo se reevalúa con datos reales y balanceo dentro de cada fold (pipeline imblearn) |

## Cambios por sección

### Celda 3: carga de datos
Corregir el nombre del archivo: `Accidentalidad_Municipio_de__Envigado_20260929.csv`.

### Celda 0 y textos de las secciones 6 y 8
Actualizar las menciones a "3 clases" o MUERTOS: el objetivo pasa a ser binario (SOLO DAÑOS / HERIDOS). Las correlaciones de la sección 6 no cambian de código.

### Sección 8: balanceo (se reemplaza)
1. Gráfica de distribución antes del balanceo (se mantiene).
2. `data = data[data['GRAVEDAD'] != 'MUERTOS']` y `remove_unused_categories()`.
3. `X_real`, `Y_real` = copia sin balancear (binaria), que se usa en la verificación.
4. `RandomUnderSampler(sampling_strategy={'SOLO DAÑOS': n_daños // 2}, random_state=42)`.
5. `SMOTENC(categorical_features=indices_categoricos, random_state=42)`, que sube HERIDOS hasta igualar DAÑOS.
6. Se reconstruye `data`, se grafica después del balanceo y se muestra `value_counts()`.
7. Markdown de resultado con los conteos finales.

### Secciones 9 y 10
El código no cambia. `labelencoder` queda `{HERIDOS: 0, SOLO DAÑOS: 1}` y los Excel se regeneran con los datos binarios.

### Sección 11: modelos (se reemplaza completa)
Se eliminan las celdas actuales (split 80/20, `crear_modelo`, `DummyClassifier`, configuraciones KNN/SVM/NN).

**Configuración común** (plantilla del usuario):
```python
from sklearn.model_selection import GridSearchCV
scoring = 'f1_macro'
cv = 10
X = data.drop('GRAVEDAD', axis=1)
Y = data['GRAVEDAD']
medidas = pd.DataFrame(index=['f1 de la CV'])
```

**Bloque por modelo** (markdown corto + código): definir el grid, ejecutar `GridSearchCV(estimator, param_grid, scoring, cv, return_train_score=True, refit=True, n_jobs=-1)`, hacer `fit(X, Y)`, guardar `best_estimator_`, imprimir `best_params_`, imprimir Train CV / Test CV de `best_index_` y asignar `medidas['<Modelo>'] = grid.best_score_`.

| Modelo | Estimador base | Grid |
|---|---|---|
| Tree | `DecisionTreeClassifier(random_state=42)` | `criterion` [entropy, gini], `min_samples_leaf` [2, 10, 50, 100], `max_depth` [None, 10, 20, 50] |
| RF | `RandomForestClassifier(random_state=42, n_jobs=-1)` | `n_estimators` [50, 100, 200], `criterion` [gini, entropy], `max_depth` [None, 10, 20], `min_samples_leaf` [2, 10], `max_features` ['sqrt'] |
| XGBoost | `XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42, n_jobs=-1)` | `n_estimators` [100, 200], `max_depth` [3, 6, 9], `learning_rate` [0.05, 0.1, 0.3], `subsample` [0.8, 1.0] |
| KNN | `KNeighborsClassifier()` | `n_neighbors` [5, 15, 27], `weights` [uniform, distance] |
| SVM | `SVC(cache_size=512)` | `kernel` ['rbf'], `C` [1, 10] |
| NN | `MLPClassifier(early_stopping=True, max_iter=300, random_state=42)` | `hidden_layer_sizes` [(15,), (64, 32)], `activation` [relu, logistic] |

Solo el árbol se dibuja: `plot_tree(modelTree, feature_names=X.columns.values, class_names=labelencoder.classes_, max_depth=3, rounded=True, filled=True, fontsize=8)` con `figsize=(20, 10)`.

Nota en el markdown de SVM: es el modelo más lento (puede tardar 20-40 min en Colab).

### 11.7 Comparación
Tabla `medidas` (6 columnas) y gráfica de barras. El mejor modelo es `medidas.loc['f1 de la CV'].idxmax()`.

### 11.8 Verificación (enfoque C)
- Datos: `X_real`, `Y_real` codificado con `labelencoder`.
- `imblearn.pipeline.Pipeline`: `RandomUnderSampler` (DAÑOS al 50 % del fold, mediante una función `sampling_strategy`), `SMOTENC` (categóricas por nombre), `ColumnTransformer` (`MinMaxScaler` en `HORA_DIA` y `RESULTADO DE BEODEZ`; `OneHotEncoder(handle_unknown='ignore', drop='first')` en las categóricas) y `clone(mejor_modelo)`.
- `cross_val_score(pipeline, X_real, y_real, cv=StratifiedKFold(10, shuffle=True, random_state=42), scoring='f1_macro')`.
- Se imprime el f1 con balanceo global frente al f1 con datos reales.

### 11.9 Conclusión (markdown)
Mejor modelo, señales de overfitting (Train vs Test CV) y diferencia entre el f1 global y el real, explicada por la fuga de SMOTENC antes de la CV. El texto se escribe después de ejecutar con los resultados reales; sin ellos se deja una plantilla con las preguntas a responder.

## Fuera de alcance
- Cambios en las secciones 1 a 7, más allá de los textos mencionados.
- Métricas adicionales (matriz de confusión, ROC) y búsqueda aleatoria (`RandomizedSearchCV`).

## Verificación de la implementación
- El notebook es JSON válido y abre en Jupyter/Colab.
- No quedan referencias a MUERTOS como clase del modelo ni a `train_test_split` en la sección 11.
- Si sklearn, imblearn y xgboost están disponibles en local, se ejecutan las secciones 8 a 11 con un `cv` reducido como prueba rápida. Si no, el usuario ejecuta en Colab.
