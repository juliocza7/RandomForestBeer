import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import missingno as msno
from seaborn import heatmap
import seaborn as sns

from scipy.stats import uniform
from scipy.stats import pearsonr as pearson

from sklearn.model_selection import train_test_split
from sklearn import impute
from sklearn import compose
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


data = pd.read_csv('/Users/julio/Documents/UFF/Disciplinas/AprendizadoMaquina/TrabajoPractico_Beer/datasetbeer/recipeData.csv', encoding="ISO-8859-1")

pd.set_option("display.max_columns", None)
print(data.sample(5))

data = data.drop(data[data['ABV']>20].index)
data = data.drop(data[data['IBU']>200].index)

##############################################
#impimir valores nulls
print(data.info(verbose=False))
msno.matrix(data.sample(500))
plt.savefig('CantidadNulos.png')
plt.show()


data.head(10)

df=data[['ABV','IBU','Style']]
final=df.groupby('Style').filter(lambda x: len(x) > 10)
df=final.dropna()
df.head()
df.info()
df['Style'].unique()

#diferentes estilos (grafica)
final1=df.groupby('Style').filter(lambda x: len(x) > 1150)
plt.figure(figsize=(20,10)) 
g = sns.lmplot(x='ABV',y='IBU',data=final1, hue='Style')
plt.savefig('EstilosCervezas.png')
plt.show(g)


'''
#frequencia da distribuição ABV
sns.distplot(data.ABV.dropna(),bins=10)
plt.xlabel('ABV')
plt.ylabel('Frequency')
plt.title('Distribution of ABV')
plt.savefig('DistribucionABV.png')
plt.show()

#frequencia da distribuição IBU
sns.distplot(data['IBU'].dropna())
plt.xlabel('IBU')
plt.ylabel('Frequency')
plt.title('Distribution of IBU')
plt.savefig('DistribucionIBU.png')
plt.show()
'''

data['Style'].describe()

#estilos
plt.figure(figsize=(40,5))
graf=sns.countplot(data=(data.head(50)),y='Style')
plt.xlabel('Quantidade de Cervejas')
plt.savefig('CantidadporEstilo.png')
plt.show(graf)


#Removendo observações que não são cervejas
beers = data.loc[~data.Style.isin(['Traditional Perry', 'French Cider', 'Pyment (Grape Melomel)',
                                   'Apple Wine', 'New England Cider', 'Metheglin', 'Open Category Mead',
                                   'Cyser (Apple Melomel)', 'Fruit Cider', 'English Cider', 'Sweet Mead',
                                   'Other Specialty Cider or Perry', 'Wheatwine', 'Semi-Sweet Mead',
                                   'Other Fruit Melomel', 'Dry Mead', 'Common Cider', 'Braggot'])]
beers = beers.loc[~beers.Style.isna()]

print(f"Numero de Classes: {len(set(beers.Style))}")

#Mostrando os estilos mais comuns
df_stylecat = beers.Style.value_counts().to_frame()
df_stylecat['percentage'] = df_stylecat.Style * 100 / sum(df_stylecat.Style)
print(df_stylecat)

#Colunas com alto percentagem de valores nulos
nulls1 = (beers.isnull().sum() / beers.shape[0] * 100.00)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Porcentagem de Nulos (%)', fontsize=12)
plt.title('Porcentagem de Valores Nulos', fontsize=14)
nulls1.sort_values(ascending=False).plot(kind='bar')
plt.tight_layout()
plt.savefig('PorcentajeValoresNulos.png')
plt.show()


#Determinando correlação
##############################################################################
#Limpando valores nulos
beers = beers[beers['IBU'].notnull()]
clean_beers = beers[beers['ABV'].notnull()]

print (clean_beers['Style'].value_counts()[:10])

(clean_beers['Style'].value_counts()[:10]).plot(kind = 'bar')
plt.savefig('estiloslimpios.png')
plt.show()

styles = (clean_beers['Style'].value_counts()[:10]).keys()

print(styles)

#correlação
fig, axes = plt.subplots(4, 3, sharex = True, sharey = True, figsize=(10,10))  
fig, global_ax = plt.subplots(figsize=(12,12)) 
x_max = clean_beers['IBU'].max() 
y_max = clean_beers['ABV'].max() 

for style, ax in zip(styles, axes.ravel()):
    ibu_data = clean_beers['IBU'][clean_beers['Style'] == style].values
    abv_data = clean_beers['ABV'][clean_beers['Style'] == style].values
    ax.set_title(style)
    ax.plot(ibu_data, abv_data, marker = 'o', linestyle = '')
    ax.legend(numpoints=1, loc='lower right', fontsize = 10)
    global_ax.plot(ibu_data, abv_data, marker = 'o', label = style, linestyle = '')

global_ax.legend(numpoints=1, loc='lower right', fontsize = 10)
plt.xlabel('IBU')
plt.ylabel('ABV')
plt.savefig('CorrelacionEstilos.png')
plt.show()
###########################################################################



#colunas com uma alta carnilidade
categoricals = list(beers.select_dtypes(include=object).columns)
categoricals = [x for x in categoricals if x not in ['Name', 'URL']]
res = {col: len(beers[col].value_counts()) for col in categoricals}
vals = pd.DataFrame.from_dict(list(res.items())).T
vals.columns = categoricals
vals = vals.iloc[1]
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Número de Valores Exclusivos', fontsize=12)
plt.title('Número de Valores Categóricos Exclusivos', fontsize=14)
vals.sort_values(ascending=False).plot(kind='bar')
plt.tight_layout()
plt.savefig('Cardinalidad.png')
plt.show()

#para colunas com altos valores nulos, criamos uma nova coluna com o a etiqueta de perdidos or não
beers['PrimingMethod_missing'] = beers.PrimingMethod.isna().astype('float')
beers['MashThickness_missing'] = beers.MashThickness.isna().astype('float')
beers['PitchRate_missing'] = beers.PitchRate.isna().astype('float')

#modificamos a gravedade especifica 
beers['Gravity_diff'] = beers.FG - beers.OG


#tratamento
def make_pipeline(classifier=None, scaled=False):    
#criamos o processador dos dads para o ajuste do classificador
    continuous_var = ['MashThickness', 'BoilGravity', 'PrimaryTemp',
                      'OG', 'FG', 'IBU', 'Color', 'BoilTime',
                      'Efficiency', 'Gravity_diff']
    categorical_var = ['BrewMethod']
    fill_zeros_var = ['ABV']
    boolean_var = ['PrimingMethod_missing',
                   'MashThickness_missing', 'PitchRate_missing']
    if scaled:
        continuous_transformer = Pipeline(steps=[
            ('imputer', impute.SimpleImputer(strategy='median')),
            ('normalize', preprocessing.StandardScaler())])
        fill_zeros_transformer = Pipeline(steps=[
            ('imputer', impute.SimpleImputer(missing_values=0, strategy='median')),
            ('normalize', preprocessing.StandardScaler())
        ])
    else:
        continuous_transformer = Pipeline(steps=[
            ('imputer', impute.SimpleImputer(strategy='median'))])
        fill_zeros_transformer = Pipeline(steps=[
            ('imputer', impute.SimpleImputer(missing_values=0, strategy='median'))])
    categorical_transformer = Pipeline(steps=[
        ('imputer', impute.SimpleImputer(strategy='most_frequent')),
        ('encoder', preprocessing.OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = compose.ColumnTransformer(
        transformers=[
            ('continuous_var', continuous_transformer, continuous_var),
            ('categorical_var', categorical_transformer, categorical_var),
            ('fill_zeros_var', fill_zeros_transformer, fill_zeros_var),
            ('boolean_var', 'passthrough', boolean_var)
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier)])
    return pipeline




##############################################
#impimir valores nulos nuevamente para comparar
print(beers.info(verbose=False))
msno.matrix(beers.sample(500))
plt.savefig('CantidadNulos_fig2.png')
plt.show()
#############################################

#####################################
######################################
#######################################
#O dataset esta desbalanceado e utilizamos Weighted F1-score

y = beers['StyleID']
X = beers.drop(columns=['Style', 'StyleID'])

# Dividindo em conjuntos train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=42, stratify=y)


#Random Forests
rf_classifier = RandomForestClassifier(n_estimators=150, min_samples_leaf=3)
pipeRF = make_pipeline(rf_classifier, scaled=False)
pipeRF.fit(X_train, y_train)

#Score Inicial do Traino com Random Forests
y_pred = pipeRF.predict(X_train)
f1_train = f1_score(y_train, y_pred, average='weighted')
print(f"{f1_train:.4f} F1-score em train dataset - Random Forests")

#Resultados Cross Validation com Random Forests (Scoring com weighted F1)
f1_scorer = make_scorer(f1_score, average='weighted')

#Resultado de Cross Validation com Random Forests (Scoring com weighted F1)
print('Cross validation scores - Random Forests:')
k_fold = StratifiedKFold(n_splits=5)
cross_val_score(pipeRF,
                X_train,
                y_train,
                scoring=f1_scorer,
                cv=k_fold)



#########################
##########################
###########################
#contruindo um novo modelamento baseado nas etiquetas da asisiação Beer Judge Certification Program (BJCP)
bjcp = pd.read_csv('/Users/julio/Documents/UFF/Disciplinas/AprendizadoMaquina/TrabajoPractico_Beer/datasetbeer/beerMapping.csv', encoding="ISO-8859-1")
mapping = {row['StyleID']: row['BJCP Mapping'] for i, row in bjcp.iterrows()}
beers['StyleCategory'] = beers['StyleID'].map(mapping)
beers = beers.loc[~beers['StyleCategory'].isna()]
print(f"Número de Classes: {len(set(beers['StyleCategory']))}")

# Mostrando os estilos mais comuns
df_stylecat = beers.StyleCategory.value_counts().to_frame()
df_stylecat['percentage'] = df_stylecat.StyleCategory * 100 / sum(df_stylecat.StyleCategory)
df_stylecat

#Train/test dividindo os dados
y = beers['StyleCategory']
X = beers.drop(columns=['Style', 'StyleID', 'StyleCategory'])

# Dividindo em conjuntos train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=42, stratify=y)

# Hyperparametros compartilhados no modelo
n_iter = 10
cv = 3

#Random Forest
pipeRF = make_pipeline(RandomForestClassifier(n_jobs=-1), scaled=False)
rf_params = rf_param = {
    "classifier__bootstrap": [True, False],
    "classifier__max_depth": [10, 20, 30, 40, 60, 80, 100, None],
    "classifier__max_features": ["auto", "log2", None],
    "classifier__min_samples_leaf": [1, 2, 3, 4, 5, 6],
    "classifier__n_estimators": [25, 50, 100, 135, 180, 230]
}

rf_random_cv = RandomizedSearchCV(pipeRF, rf_params, n_iter=25, cv=3,
                                  scoring='f1_weighted', n_jobs=-1)

rf_random_cv.fit(X_train, y_train)

rf_cv_results = pd.DataFrame(rf_random_cv.cv_results_).sort_values('mean_test_score', ascending=False)
rf_cv_results

rf_random_cv.best_params_

y_test_preds = rf_random_cv.predict(X_test)

f1_test = f1_score(y_test, y_test_preds, average='weighted')
print(f"{f1_test:.4f} F1-score em test dataset")

# Cria a matriz de confusion
cm = confusion_matrix(y_test, y_test_preds, labels=np.unique(y_test))

# Armazena as etiquetas das categorias
label, _ = np.unique(y_test, return_counts=True)

# Cria um dataframe e calcula os porcentagens por filas 
confusion_mat = pd.DataFrame(cm, columns=label, index=label)
confusion_mat = confusion_mat.divide(confusion_mat.sum(axis=1), axis=0) * 100

plt.figure(figsize=(20, 12))
ax = heatmap(confusion_mat, annot=True, annot_kws={"size": 8}, cmap="Blues")
plt.savefig('ConfusionMatriz.png')
plt.show()

print(classification_report(y_test, y_test_preds))

# Armazenar reporte de dados
cr_df = pd.DataFrame(classification_report(y_test, y_test_preds, output_dict=True)).transpose()
plt.figure(figsize=(12,6))
plt.scatter(cr_df.support, cr_df['f1-score'])
plt.xlabel('Suporte')
plt.ylabel('f1-score')
plt.title('Relação positiva leve entre o suporte de um estilo e seu f1-score')
plt.savefig('Classificacion.png')
plt.show()

#Armazena o novo dataset que foi tratado para trabalhar acima dele
beers.to_csv('novodataset.csv')