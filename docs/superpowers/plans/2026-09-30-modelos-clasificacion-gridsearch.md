# Balanceo binario y modelos con GridSearchCV — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pasar el problema a binario (SOLO DAÑOS vs HERIDOS) con un nuevo balanceo, y reemplazar la sección 11 por 6 modelos hiperparametrizados con `GridSearchCV` (`cv=10`, `f1_macro`) más una verificación honesta del mejor modelo.

**Architecture:** El notebook `Preparacion_de_datos_Siniestros_Viales_Envigado.ipynb` se edita con scripts Python que manipulan el JSON y ubican las celdas por `id` (no por índice). Un script de verificación extrae las celdas de código, reduce `cv` y ejecuta el notebook de la sección 1 a la 11 en local.

**Tech Stack:** pandas, scikit-learn, imbalanced-learn (`RandomUnderSampler`, `SMOTENC`, `Pipeline`), xgboost, matplotlib. Ejecución final: Google Colab.

**Spec:** `docs/superpowers/specs/2026-09-30-modelos-clasificacion-gridsearch-design.md`

**Rutas:**
- `REPO` = `C:/Users/samue/OneDrive/Documentos/Universidad/8° Universidad/Analitica de datos/Siniestros-viales`
- `NB` = `REPO/Preparacion_de_datos_Siniestros_Viales_Envigado.ipynb`
- `TMP` = scratchpad de la sesión (`C:/Users/samue/AppData/Local/Temp/claude/C--Users-samue/d0c4a10f-10a9-44c6-afe4-affad92a6f6e/scratchpad`)

Los scripts auxiliares viven en `TMP` (no se commitean). Solo se commitea el notebook.

---

## Mapa de celdas (ids actuales)

| Índice | id | Qué es | Acción |
|---|---|---|---|
| 0 | `9b4e99fe` | Título / fases | Editar texto |
| 3 | `36fd3c1a` | Carga CSV | Corregir nombre de archivo |
| 70 | `864a322f` | MD sección 8 | Reemplazar |
| 71 | `b872eeba` | Gráfica antes del balanceo | Conservar |
| 72 | `ddef7481` | SMOTENC 3 clases | Reemplazar |
| 73 | `ed152827` | Reconstruir data | Reemplazar |
| 74 | `a69cb8bb` | MD resultado | Reemplazar |
| 90–104 | `E55bjvJaYSPD` … `YS34SSDje6R3` | Sección 11 actual | Eliminar y poner celdas nuevas |
| 105 | `f306da60` | `# **FIN**` | Conservar |

Las celdas 63 y 66 (sección 7) mencionan MUERTOS, pero en ese punto los datos todavía tienen 3 clases, así que su texto sigue siendo correcto y no se toca.

---

### Task 1: Entorno local y utilidades

**Files:**
- Create: `TMP/nbtools.py`
- Create: `TMP/verificar.py`

- [ ] **Step 1: Crear venv e instalar dependencias**

```bash
cd "$TMP" && python -m venv venv && ./venv/Scripts/python -m pip install -q pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost openpyxl
./venv/Scripts/python -c "import sklearn, imblearn, xgboost; print(sklearn.__version__, imblearn.__version__, xgboost.__version__)"
```
Expected: se imprimen las tres versiones sin error.

- [ ] **Step 2: Crear `TMP/nbtools.py`**

```python
import json, uuid

def cargar(ruta):
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)

def guardar(nb, ruta):
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write('\n')

def _lineas(src):
    src = src.strip('\n')
    lineas = src.split('\n')
    return [l + '\n' for l in lineas[:-1]] + [lineas[-1]]

def md(src):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': _lineas(src)}

def code(src):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'metadata': {},
            'execution_count': None, 'outputs': [], 'source': _lineas(src)}

def indice(nb, cell_id):
    for i, c in enumerate(nb['cells']):
        if c.get('id') == cell_id:
            return i
    raise KeyError(cell_id)

def reemplazar_rango(nb, id_inicio, id_fin, nuevas):
    """Reemplaza las celdas desde id_inicio hasta id_fin (inclusive) por `nuevas`."""
    i, j = indice(nb, id_inicio), indice(nb, id_fin)
    nb['cells'][i:j + 1] = nuevas

def set_source(nb, cell_id, src):
    nb['cells'][indice(nb, cell_id)]['source'] = _lineas(src)
```

- [ ] **Step 3: Crear `TMP/verificar.py`** (prueba de humo: ejecuta todas las celdas de código con `cv` reducido)

```python
import sys, os, re, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close('all')

REPO = sys.argv[1]
CV = int(sys.argv[2]) if len(sys.argv) > 2 else 2
os.chdir(REPO)
nb = json.load(open('Preparacion_de_datos_Siniestros_Viales_Envigado.ipynb', encoding='utf-8'))

ns = {}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if 'ydata_profiling' in src or 'profile_data' in src:   # perfilado opcional: se omite en local
        continue
    src = '\n'.join(l for l in src.splitlines() if not l.lstrip().startswith(('%', '!')))
    src = re.sub(r'^cv = 10\b', f'cv = {CV}', src, flags=re.M)
    src = src.replace('n_splits=10', f'n_splits={CV}')
    src = re.sub(r'\.to_excel\(.*\)', '', src)   # no regenerar los Excel en la prueba
    print(f'--- celda {i}', flush=True)
    exec(compile(src, f'celda_{i}', 'exec'), ns)

# Comprobaciones finales
d = ns['data']
assert set(ns['labelencoder'].classes_) == {'HERIDOS', 'SOLO DAÑOS'}, ns['labelencoder'].classes_
assert d['GRAVEDAD'].value_counts().nunique() == 1, 'clases no balanceadas'
assert list(ns['medidas'].columns) == ['Tree', 'RF', 'XGBoost', 'KNN', 'SVM', 'NN'], ns['medidas'].columns
assert 'f1_real' in ns
print('OK', ns['medidas'].round(4).to_dict('records'), 'f1_real', round(ns['f1_real'].mean(), 4))
```

- [ ] **Step 4: Ejecutar la verificación sobre el notebook actual (debe fallar)**

Run: `cd "$TMP" && ./venv/Scripts/python verificar.py "$REPO" 2`
Expected: FAIL. Puede fallar en la celda 3 (`FileNotFoundError ..._20260806.csv`) o en el assert de clases. Esto confirma que la prueba detecta el estado actual.

---

### Task 2: Celda 0 (título) y celda 3 (CSV)

**Files:** Modify: `NB` (ids `9b4e99fe`, `36fd3c1a`)

- [ ] **Step 1: Crear y ejecutar `TMP/task2.py`**

```python
import sys; sys.path.insert(0, '.')
from nbtools import cargar, guardar, set_source
RUTA = sys.argv[1]
nb = cargar(RUTA)

set_source(nb, '9b4e99fe', """# Preparación de Datos — Siniestros Viales, Municipio de Envigado

**Fuente:** Datos Abiertos Colombia — Accidentalidad Municipio de Envigado
**Variable objetivo:** `GRAVEDAD` — se modela como problema de **clasificación binaria** (SOLO DAÑOS / HERIDOS); la clase MUERTOS se descarta en el paso 8
**Registros:** > 43.000 siniestros únicos · **Variables predictoras finales:** 8

Fases desarrolladas:
1. Integración de los datos
2. Eliminar variables irrelevantes/redundantes
3. Descripción estadística de los datos
4. Limpieza de datos atípicos
5. Limpieza de datos nulos
6. Análisis de correlaciones para redundancias
7. Análisis de correlaciones para irrelevancias
8. Balanceo de datos
9. Transformaciones (discretización / normalización)
10. Guardar los datos preparados
11. Modelos de clasificación con hiperparametrización (GridSearchCV + validación cruzada)""")

set_source(nb, '36fd3c1a', """# Cargamos los datos
data = pd.read_csv("Accidentalidad_Municipio_de__Envigado_20260929.csv", low_memory=False)
print("Filas x Columnas:", data.shape)
data.head()""")

guardar(nb, RUTA)
print('ok')
```

Run: `cd "$TMP" && ./venv/Scripts/python task2.py "$NB"`
Expected: `ok`

- [ ] **Step 2: Verificar**

Run: `grep -c "20260929.csv" "$NB"`
Expected: `1`

---

### Task 3: Sección 8 (nuevo balanceo)

**Files:** Modify: `NB` (ids `864a322f`, `ddef7481`–`a69cb8bb`)

- [ ] **Step 1: Crear y ejecutar `TMP/task3.py`**

````python
import sys; sys.path.insert(0, '.')
from nbtools import cargar, guardar, set_source, reemplazar_rango, md, code
RUTA = sys.argv[1]
nb = cargar(RUTA)

set_source(nb, '864a322f', """# 8. Balanceo de la variable objetivo (Clasificación)

`GRAVEDAD` está muy desbalanceada: cerca del 79.5 % de los siniestros son "SOLO DAÑOS", el 20.2 % "HERIDOS" y apenas el 0.3 % "MUERTOS". Un modelo entrenado así tendería a ignorar las clases minoritarias. Decisiones:

1. **Se elimina la clase MUERTOS:** con el 0.3 % de los registros no hay suficientes casos reales para aprender un patrón confiable, y sobremuestrearla implicaría inventar casi todos sus registros. El problema pasa a ser **binario: SOLO DAÑOS vs HERIDOS**.
2. **Submuestreo de SOLO DAÑOS al 50 %** (`RandomUnderSampler`): se descarta al azar la mitad de los registros de la clase mayoritaria.
3. **Sobremuestreo de HERIDOS** con **SMOTENC** (variante de SMOTE que soporta variables categóricas + numéricas mezcladas) hasta igualar a SOLO DAÑOS, para una proporción 50/50.""")

nuevas = [
    code("""# Eliminamos la clase MUERTOS: el problema pasa a ser binario
data = data[data['GRAVEDAD'] != 'MUERTOS'].copy()
data['GRAVEDAD'] = data['GRAVEDAD'].astype(str).astype('category')
print("Shape sin MUERTOS:", data.shape)
data['GRAVEDAD'].value_counts()"""),
    code("""from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

X = data.drop(columns=['GRAVEDAD'])
Y = data['GRAVEDAD'].astype(str)

# Copia de los datos reales (binarios, sin balancear) para la verificación de la sección 11
X_real = X.copy()
Y_real = Y.copy()

# 1) Submuestreo: SOLO DAÑOS se reduce a la mitad
n_danos = (Y == 'SOLO DAÑOS').sum()
rus = RandomUnderSampler(sampling_strategy={'SOLO DAÑOS': int(n_danos // 2)}, random_state=42)
X_sub, Y_sub = rus.fit_resample(X, Y)
print("Después del submuestreo:", Y_sub.value_counts().to_dict())

# 2) Sobremuestreo: HERIDOS sube con SMOTENC hasta igualar a SOLO DAÑOS
columnas_categoricas = ['DÍA DE LA SEMANA', 'CLASE DE ACCIDENTE', 'AREA', 'MES',
                         'CAUSA_AGRUPADA', 'BARRIO_AGRUPADO']
indices_categoricos = [X.columns.get_loc(c) for c in columnas_categoricas]
sm = SMOTENC(categorical_features=indices_categoricos, random_state=42)
X_bal, Y_bal = sm.fit_resample(X_sub, Y_sub)

print("Shape antes:", X.shape, "-> Shape después:", X_bal.shape)"""),
    code("""# Reconstruimos el dataframe balanceado.
# Se mezclan las filas: SMOTENC agrega los registros sintéticos al final y la validación
# cruzada de GridSearchCV (StratifiedKFold sin shuffle) los concentraría en los últimos folds.
data = pd.DataFrame(X_bal, columns=X.columns).reset_index(drop=True)
data['GRAVEDAD'] = pd.Series(Y_bal).reset_index(drop=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data['GRAVEDAD'].value_counts().plot(kind='bar', color=['#4C72B0', '#DD8452'])
plt.title('GRAVEDAD después del balanceo (submuestreo + SMOTENC)')
plt.show()
data['GRAVEDAD'].value_counts()"""),
    md("""**Resultado:** se eliminó MUERTOS, SOLO DAÑOS se redujo a la mitad y HERIDOS se completó con registros sintéticos de SMOTENC hasta igualarlo; las dos clases quedan al 50 % (conteos exactos en la celda anterior). Se conservan `X_real` / `Y_real` (datos reales, binarios y sin balancear) para la verificación final de la sección 11."""),
]
reemplazar_rango(nb, 'ddef7481', 'a69cb8bb', nuevas)

guardar(nb, RUTA)
print('ok')
````

Run: `cd "$TMP" && ./venv/Scripts/python task3.py "$NB"`
Expected: `ok`

- [ ] **Step 2: Verificar estructura**

Run: `grep -c "RandomUnderSampler(sampling_strategy" "$NB"; grep -c "X_real = X.copy()" "$NB"`
Expected: `1` y `1`

---

### Task 4: Sección 11, configuración y modelos de árboles (Tree, RF, XGBoost)

**Files:** Modify: `NB` (elimina ids `E55bjvJaYSPD`–`YS34SSDje6R3` y pone las celdas nuevas antes de `f306da60`)

- [ ] **Step 1: Crear `TMP/celdas11.py`** (solo define las celdas; se completa en la Task 5)

````python
from nbtools import md, code

intro = [
    md("""# 11. Modelos de clasificación con hiperparametrización

Se entrenan seis modelos: árbol de decisión, Random Forest, XGBoost, KNN, SVM y red neuronal. En lugar de dividir en entrenamiento/prueba (70/30), cada modelo se ajusta con **`GridSearchCV`** usando **validación cruzada de 10 folds** y la medida **f1 macro**.

Se usa el dataset numérico `data` de la sección 9 (balanceado, normalizado, con dummies y label encoder). Para cada modelo se revisa el overfitting/underfitting comparando el f1 promedio en entrenamiento (Train CV) y en validación (Test CV)."""),
    code("""# Configuración hiperparametrización
from sklearn.model_selection import GridSearchCV
scoring = 'f1_macro'
cv = 10

# Configuración de datos
X = data.drop('GRAVEDAD', axis=1)  # Variables predictoras
Y = data['GRAVEDAD']               # Variable objetivo

# Medida de evaluación del mejor modelo
medidas = pd.DataFrame(index=['f1 de la CV'])
print("X:", X.shape, "| registros por clase:", dict(zip(labelencoder.classes_, np.bincount(Y))))"""),
]

arbol = [
    md("""## 11.1. Árbol de decisión"""),
    code("""from sklearn.tree import DecisionTreeClassifier
modelTree = DecisionTreeClassifier(random_state=42)

# Definir los hiperparámetros
criterion = ['entropy', 'gini']      # Índice de información
min_samples_leaf = [2, 10, 50, 100]  # Cantidad de registros por hoja
max_depth = [None, 10, 20, 50]       # Niveles de profundidad

# Hiperparametrización
param_grid = dict(criterion=criterion, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
grid = GridSearchCV(estimator=modelTree, param_grid=param_grid, scoring=scoring, cv=cv,
                    return_train_score=True, refit=True, n_jobs=-1)
grid.fit(X, Y)

# Mejor modelo
modelTree = grid.best_estimator_

# Mejores parámetros
print(grid.best_params_)

# Revisar overfitting y underfitting
print("Train CV:", grid.cv_results_["mean_train_score"][grid.best_index_])
print("Test CV:", grid.cv_results_["mean_test_score"][grid.best_index_])

# Comparación de medidas
medidas['Tree'] = grid.best_score_
medidas"""),
    code("""# Mejor árbol (se dibujan solo los 3 primeros niveles para que sea legible)
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(modelTree, feature_names=X.columns.values, class_names=labelencoder.classes_,
          max_depth=3, rounded=True, filled=True, fontsize=8)
plt.show()"""),
]

rf = [
    md("""## 11.2. Random Forest

Respecto a la primera versión: `n_estimators` sube a 50-200 (1 a 4 árboles no forman un bosque útil), `max_depth` usa 10/20/None (profundidades de 1 a 7 tienden a subajustar) y se quita `class_weight` porque los datos ya están balanceados."""),
    code("""from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier(random_state=42, n_jobs=-1)

# Definir los hiperparámetros
n_estimators = [50, 100, 200]       # Cantidad de árboles
criterion = ['gini', 'entropy']     # Índice de información
max_depth = [None, 10, 20]          # Niveles de profundidad
min_samples_leaf = [2, 10]          # Cantidad de registros por hoja
max_features = ['sqrt']             # Variables candidatas en cada división

# Hiperparametrización
param_grid = dict(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                  min_samples_leaf=min_samples_leaf, max_features=max_features)
grid = GridSearchCV(estimator=modelRF, param_grid=param_grid, scoring=scoring, cv=cv,
                    return_train_score=True, refit=True, n_jobs=-1)
grid.fit(X, Y)

# Mejor modelo
modelRF = grid.best_estimator_
print(grid.best_params_)

# Revisar overfitting y underfitting
print("Train CV:", grid.cv_results_["mean_train_score"][grid.best_index_])
print("Test CV:", grid.cv_results_["mean_test_score"][grid.best_index_])

# Comparación de medidas
medidas['RF'] = grid.best_score_
medidas"""),
]

xgb = [
    md("""## 11.3. XGBoost

Ensamble de árboles construidos de forma secuencial: cada árbol corrige los errores de los anteriores. `learning_rate` controla cuánto aporta cada árbol y `subsample` la fracción de registros que ve cada uno."""),
    code("""from xgboost import XGBClassifier
modelXGB = XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42, n_jobs=-1)

# Definir los hiperparámetros
n_estimators = [100, 200]           # Cantidad de árboles
max_depth = [3, 6, 9]               # Profundidad de cada árbol
learning_rate = [0.05, 0.1, 0.3]    # Tasa de aprendizaje
subsample = [0.8, 1.0]              # Fracción de registros por árbol

# Hiperparametrización
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth,
                  learning_rate=learning_rate, subsample=subsample)
grid = GridSearchCV(estimator=modelXGB, param_grid=param_grid, scoring=scoring, cv=cv,
                    return_train_score=True, refit=True, n_jobs=-1)
grid.fit(X, Y)

# Mejor modelo
modelXGB = grid.best_estimator_
print(grid.best_params_)

# Revisar overfitting y underfitting
print("Train CV:", grid.cv_results_["mean_train_score"][grid.best_index_])
print("Test CV:", grid.cv_results_["mean_test_score"][grid.best_index_])

# Comparación de medidas
medidas['XGBoost'] = grid.best_score_
medidas"""),
]
````

- [ ] **Step 2: Crear y ejecutar `TMP/task4.py`** (aplica intro + árboles; la Task 5 agrega el resto)

```python
import sys; sys.path.insert(0, '.')
from nbtools import cargar, guardar, reemplazar_rango
from celdas11 import intro, arbol, rf, xgb
RUTA = sys.argv[1]
nb = cargar(RUTA)
reemplazar_rango(nb, 'E55bjvJaYSPD', 'YS34SSDje6R3', intro + arbol + rf + xgb)
guardar(nb, RUTA)
print('ok')
```

Run: `cd "$TMP" && ./venv/Scripts/python task4.py "$NB"`
Expected: `ok`

- [ ] **Step 3: Verificar**

Run: `grep -c "train_test_split\|crear_modelo\|DummyClassifier" "$NB"`
Expected: `0`

---

### Task 5: Sección 11, KNN, SVM, NN, comparación y verificación

**Files:** Modify: `TMP/celdas11.py`, `NB` (inserta antes de `f306da60`)

- [ ] **Step 1: Agregar al final de `TMP/celdas11.py`**

````python
knn = [
    md("""## 11.4. KNN"""),
    code("""from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier()

# Definir los hiperparámetros
n_neighbors = [5, 15, 27]           # Cantidad de vecinos
weights = ['uniform', 'distance']   # Peso del voto de cada vecino

# Hiperparametrización
param_grid = dict(n_neighbors=n_neighbors, weights=weights)
grid = GridSearchCV(estimator=modelKNN, param_grid=param_grid, scoring=scoring, cv=cv,
                    return_train_score=True, refit=True, n_jobs=-1)
grid.fit(X, Y)

# Mejor modelo
modelKNN = grid.best_estimator_
print(grid.best_params_)

# Revisar overfitting y underfitting
print("Train CV:", grid.cv_results_["mean_train_score"][grid.best_index_])
print("Test CV:", grid.cv_results_["mean_test_score"][grid.best_index_])

# Comparación de medidas
medidas['KNN'] = grid.best_score_
medidas"""),
]

svm = [
    md("""## 11.5. SVM

Es el modelo más costoso (SVC escala casi cuadráticamente con el número de registros), por eso se prueba un grid reducido. Puede tardar entre 20 y 40 minutos en Colab."""),
    code("""from sklearn.svm import SVC
modelSVM = SVC(cache_size=512)

# Definir los hiperparámetros
kernel = ['rbf']    # Función kernel
C = [1, 10]         # Penalización por error

# Hiperparametrización
param_grid = dict(kernel=kernel, C=C)
grid = GridSearchCV(estimator=modelSVM, param_grid=param_grid, scoring=scoring, cv=cv,
                    return_train_score=True, refit=True, n_jobs=-1)
grid.fit(X, Y)

# Mejor modelo
modelSVM = grid.best_estimator_
print(grid.best_params_)

# Revisar overfitting y underfitting
print("Train CV:", grid.cv_results_["mean_train_score"][grid.best_index_])
print("Test CV:", grid.cv_results_["mean_test_score"][grid.best_index_])

# Comparación de medidas
medidas['SVM'] = grid.best_score_
medidas"""),
]

nn = [
    md("""## 11.6. Red neuronal"""),
    code("""from sklearn.neural_network import MLPClassifier
modelNN = MLPClassifier(early_stopping=True, max_iter=300, random_state=42)

# Definir los hiperparámetros
hidden_layer_sizes = [(15,), (64, 32)]   # Neuronas por capa oculta
activation = ['relu', 'logistic']        # Función de activación

# Hiperparametrización
param_grid = dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation)
grid = GridSearchCV(estimator=modelNN, param_grid=param_grid, scoring=scoring, cv=cv,
                    return_train_score=True, refit=True, n_jobs=-1)
grid.fit(X, Y)

# Mejor modelo
modelNN = grid.best_estimator_
print(grid.best_params_)

# Revisar overfitting y underfitting
print("Train CV:", grid.cv_results_["mean_train_score"][grid.best_index_])
print("Test CV:", grid.cv_results_["mean_test_score"][grid.best_index_])

# Comparación de medidas
medidas['NN'] = grid.best_score_
medidas"""),
]

comparacion = [
    md("""## 11.7. Comparación de modelos"""),
    code("""print(medidas.T.sort_values('f1 de la CV', ascending=False))
medidas.T.sort_values('f1 de la CV').plot(kind='barh', legend=False)
plt.xlabel('f1 macro (validación cruzada, 10 folds)')
plt.title('Comparación de modelos')
plt.show()

modelos = {'Tree': modelTree, 'RF': modelRF, 'XGBoost': modelXGB,
           'KNN': modelKNN, 'SVM': modelSVM, 'NN': modelNN}
mejor_nombre = medidas.loc['f1 de la CV'].idxmax()
mejor_modelo = modelos[mejor_nombre]
print("Mejor modelo:", mejor_nombre, "| f1 CV:", round(medidas.loc['f1 de la CV', mejor_nombre], 4))"""),
]

verificacion = [
    md("""## 11.8. Verificación del mejor modelo con datos reales

En las secciones anteriores el balanceo (submuestreo + SMOTENC) se hizo **antes** de la validación cruzada. Los registros sintéticos de HERIDOS se crean a partir de vecinos que luego caen en otros folds, así que el f1 obtenido puede ser **optimista** (fuga de información).

Para medirlo, se reevalúa el mejor modelo con los datos reales (`X_real`, `Y_real`) y un `Pipeline` de imblearn que balancea **solo dentro del fold de entrenamiento**. El fold de validación siempre contiene accidentes reales."""),
    code("""from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline

y_real = labelencoder.transform(Y_real)
clase_danos = labelencoder.transform(['SOLO DAÑOS'])[0]

def mitad_danos(y):
    # SOLO DAÑOS se reduce a la mitad, calculado sobre el fold de entrenamiento
    return {clase_danos: int((y == clase_danos).sum() // 2)}

categoricas = ['DÍA DE LA SEMANA', 'CLASE DE ACCIDENTE', 'AREA', 'MES',
               'CAUSA_AGRUPADA', 'BARRIO_AGRUPADO']
numericas = ['HORA_DIA', 'RESULTADO DE BEODEZ']
preparacion = ColumnTransformer([
    ('numericas', MinMaxScaler(), numericas),
    ('categoricas', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricas)
])

pipeline_real = Pipeline([
    ('submuestreo', RandomUnderSampler(sampling_strategy=mitad_danos, random_state=42)),
    ('smotenc', SMOTENC(categorical_features=[X_real.columns.get_loc(c) for c in categoricas],
                        random_state=42)),
    ('preparacion', preparacion),
    ('modelo', clone(mejor_modelo)),
])

cv_real = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
f1_real = cross_val_score(pipeline_real, X_real, y_real, cv=cv_real, scoring='f1_macro', n_jobs=-1)

f1_global = medidas.loc['f1 de la CV', mejor_nombre]
print(f"Mejor modelo: {mejor_nombre}")
print(f"f1 CV con balanceo global:                 {f1_global:.4f}")
print(f"f1 CV con datos reales (balanceo por fold): {f1_real.mean():.4f} ± {f1_real.std():.4f}")
print(f"Diferencia:                                {f1_global - f1_real.mean():.4f}")"""),
]

conclusion = [
    md("""## 11.9. Conclusiones

__CONCLUSION__

**Nota sobre el balanceo:** Árbol, Random Forest, XGBoost y SVM podrían manejar el desbalance sin remuestrear los datos (`class_weight='balanced'` en sklearn o `scale_pos_weight` en XGBoost), entrenando directamente con los datos reales. KNN y la red neuronal (`MLPClassifier` no acepta pesos por clase) sí requieren el balanceo. Se usó el mismo balanceo para los seis modelos para compararlos en igualdad de condiciones."""),
]
````

La marca `__CONCLUSION__` se reemplaza en la Task 7 con el texto basado en los resultados reales.

- [ ] **Step 2: Crear y ejecutar `TMP/task5.py`**

```python
import sys; sys.path.insert(0, '.')
from nbtools import cargar, guardar, indice
from celdas11 import knn, svm, nn, comparacion, verificacion, conclusion
RUTA = sys.argv[1]
nb = cargar(RUTA)
i = indice(nb, 'f306da60')   # celda "# **FIN**"
nb['cells'][i:i] = knn + svm + nn + comparacion + verificacion + conclusion
guardar(nb, RUTA)
print('ok')
```

Run: `cd "$TMP" && ./venv/Scripts/python task5.py "$NB"`
Expected: `ok`

---

### Task 6: Prueba de humo (cv=2)

- [ ] **Step 1: Ejecutar la verificación**

Run: `cd "$TMP" && ./venv/Scripts/python verificar.py "$REPO" 2`
Expected: termina con `OK [...] f1_real 0.xxxx`, sin excepciones. Si una celda falla, corregir su fuente en `celdas11.py` o `task3.py`, restaurar el notebook con `git checkout -- Preparacion_de_datos_Siniestros_Viales_Envigado.ipynb` y repetir las Tasks 2 a 5.

- [ ] **Step 2: Comprobar que no queden referencias viejas**

Run: `cd "$TMP" && ./venv/Scripts/python -c "import json,sys; nb=json.load(open(sys.argv[1],encoding='utf-8')); s=''.join(''.join(c['source']) for c in nb['cells'][-30:]); print('MUERTOS' in s, 'train_test_split' in s, nb['cells'][-1]['id'])" "$NB"`
Expected: `False False f306da60`

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add Preparacion_de_datos_Siniestros_Viales_Envigado.ipynb && git commit -m "Balanceo binario y modelos con GridSearchCV (Tree, RF, XGBoost, KNN, SVM, NN)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Ejecución completa (cv=10) y conclusiones

- [ ] **Step 1: Ejecutar con cv=10 en segundo plano** (16 núcleos locales; puede tardar 20-60 min)

Run: `cd "$TMP" && ./venv/Scripts/python verificar.py "$REPO" 10 > corrida_cv10.log 2>&1`
Expected: la última línea del log empieza con `OK`. Anotar los `best_params_`, Train/Test CV de cada modelo, la tabla `medidas` y `f1_real`.

- [ ] **Step 2: Reemplazar `__CONCLUSION__` con los resultados reales**

Con un script `TMP/task7.py` (usar `set_source` sobre la celda que contiene `__CONCLUSION__`), escribir 4 viñetas en español:
1. Mejor modelo y su f1 CV, con el ranking de los seis.
2. Overfitting: modelos con la mayor brecha Train CV − Test CV (citar valores).
3. Underfitting: modelos con Train y Test bajos (si los hay).
4. f1 con balanceo global vs `f1_real` y su diferencia, explicada por la fuga de SMOTENC.

Indicar que los valores salen de una ejecución local con las mismas semillas y que en Colab pueden variar levemente.

- [ ] **Step 3: Verificar y commit**

Run: `grep -c "__CONCLUSION__" "$NB"` → Expected: `0`

```bash
cd "$REPO" && git add Preparacion_de_datos_Siniestros_Viales_Envigado.ipynb && git commit -m "Conclusiones de los modelos con resultados de la CV

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
