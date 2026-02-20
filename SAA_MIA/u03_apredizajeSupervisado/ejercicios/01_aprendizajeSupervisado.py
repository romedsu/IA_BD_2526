# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 11:05:43 2026

@author: Mañana
"""

'''
1 Crear una funcion que construya un DataFrame a partir del fichero 
cotizacion.csv.
Se deberá realizar usando el formato de creación de DataFrames de clave-valor,
 de la forma:
pd.DataFrame({'col1':serie1, 'col2':serie2}) siendo col1 las columnas y serie
 los datos de cada columna
 a) Crear la funcion cotizaciones
b) leer el fichero indicando separador (;) , los miles (.) y decimal (,)
c) Eliminar la columna 'Nombre'
d) Crear DataFrame usando clave:valor y asignarle columnas 'Minimo','Maximo','Media'
e) Retornar el dataframe creado e imprimirlo desde fuera de la función
'''
import pandas as pd

#ruta=r'C:\Users\Mañana\OneDrive - Consejería de Educación\Documentos\rm\SAA_MIA\u03_apredizaje_supervisado\ejercicios\cotizacion.csv'

ruta='datasets/cotizacion.csv'

def cotizaciones(ruta):
    datos=pd.read_csv(ruta,sep=';',thousands='.',decimal=',')

    datos=datos.drop(columns=['Nombre'])

    dfDatos=pd.DataFrame({
        'Minimo':datos.min(),
        'Maximo':datos.max(),
        'Media':datos.mean()
        })

    # print(dfDatos.head())

    return(dfDatos)
   
    
print(cotizaciones(ruta))
   


#%% TITANIC (1ªparte)

'''
2 Analiza el archivo csv y busca en la documentación cómo usar los distintos métodos:
a) Genera un dataframe con los datos de titanic.csv
b) Imprima dimensiones, tamaño, indice y las 10 ultimas lineas del dataframe
c) Datos del pasajero con identificador 148 con loc[] (ojo indexacion desde cero)
d) Mostrar por pantalla las filas pares usando iloc[range(...)]
e) Nombres de personas de primera clase 'Pclass==1' ordenadas alfabeticamente
'''
# import pandas as pd

rutaTitanic='datasets/titanic.csv'

datosTitanic=pd.read_csv(rutaTitanic,sep=',')

dfTitanic=pd.DataFrame(
    datosTitanic,
    index=datosTitanic.index,
    columns=datosTitanic.columns)

print(f"DIMENSIONES:\n {dfTitanic.shape}")
print(f'TAMAÑO:\n {dfTitanic.size}')
print(f'INDEX:\n {dfTitanic.index}')
print(f'ULTIMOS 10: {dfTitanic.tail(10)}')

print(f'PASAJERO 148:\n {dfTitanic.loc[147]}')

#PASAJEROS PARES
pasajerosPares=dfTitanic.iloc[range(0,len(dfTitanic),2)]
print(f'PASAJEROS PARES:\n {pasajerosPares}')

# print(f'PASAJEROS PARES {dfTitanic.iloc[range(0,len(dfTitanic),2)]}')

#PRIMERA CLASE
primeraClase=dfTitanic.loc[dfTitanic['Pclass']==1]

#ORDENAR por nombre
primeraClase=primeraClase.sort_values('Name',ascending=True)

#Sumamos 1 al index, para que coincidan el index y el id del pasajero
primeraClase.index=primeraClase.index+1
print(f'PASAJEROS 1ªCLASE:\n {primeraClase}')

# primeraClase=dfTitanic.loc[dfTitanic['Pclass']==1,'Name'].sort_values()
# primeraClase.index=primeraClase.index+1


#%% TITANIC (2ªparte)

'''
3 Continuando con el archivo csv de titanic, investiga cómo usar los métodos:
a) Imprimir porcentaje de supervivientes con .value_counts(normalize=True)*100
b) Porcentaje de supervivientes de cada clase, usando .groupby('Pclass')['Survived']
c) Eliminar los pasajeros con edad desconocida con dropna()
d) Edad media (mean) de las mujeres que viajaban de cada clase, ['Sex']=='female'....
e) Añadir columna booleana 'Young' para ver si pasajero era menor de edad
f) Mostrar todas las columnas del dataframe usando pd.set_option(...)
'''
#PORCENTAJE SUPERVIVIENTES
supervivientes=dfTitanic['Survived'].value_counts(normalize=True)*100

print(f'\nSUPERVIVIENTES:\n {supervivientes}\n\n')


#PORCENTAJE de cada CLASE (con GROUP BY)
supervivientesClase=dfTitanic.groupby('Pclass')['Survived'].value_counts(normalize=True)*100
print(f'\nSUPERVIVIENTES x CLASE:\n {supervivientesClase}\n\n')

#ELIMINAR NaN de columna (con una copia)
dfTitanicUp=dfTitanic.dropna(subset=['Age']).copy()
print(dfTitanicUp)


#MEDIA de MUJERES por CLASE
media= dfTitanicUp[dfTitanicUp['Sex']=='female'].groupby('Pclass')['Age'].mean()

print(f"\nMEDIA: {media}")


#AÑADIR COLUMNA (booleana)
dfTitanicUp['Young']=dfTitanicUp['Age']<18

print(f"\nMENORES:\n {dfTitanicUp['Young']}\n\n")


# TODAS LAS COLUMNAS DE DF
# configuracion de como mostrar por terminal (todas las filas -> max_rows)
pd.set_option('display.max_columns', None)

print(dfTitanicUp)

#%% EJER 4

'''
4 Los ficheros emisiones-201X.csv contienen datos sobre las emisiones contaminantes 
('MAGNITUD') en la ciudad de Madrid.
a) Generar el DataFrame 'emisiones' con los datos de los 4 ficheros conjuntos usando .concat
    
b) Filtrar las columnas para quedarse con Estacion, Magnitud, Año y Mes de los dias D01, D02, etc
    
c) Reestructurar el DataFrame para que los valores de los contaminantes de las
    columnas de los dias aparezcan en una única columna usando .melt(id_vars=[...]))
    
d) (import datetime) Crear columna FECHA con la concatenación del año, mes, día.
    (usar método emisiones.DIA.str.strip() y emisiones.ANO.astype(str)))
    
e) Eliminar filas con fechas no validas usando numpy.isnat(...)
'''

import pandas as pd

# A) GENERAR DATAFRAME con todos los datasets
data2016=pd.read_csv('datasets/emisiones2016.csv', sep=';')
data2017=pd.read_csv('datasets/emisiones2017.csv', sep=';')
data2018=pd.read_csv('datasets/emisiones2018.csv', sep=';')
data2019=pd.read_csv('datasets/emisiones2019.csv', sep=';')


'''
df2016=pd.DataFrame(data2016,
                    index=data2016.index,
                    columns=data2016.columns)

df2017=pd.DataFrame(data2017,
                    index=data2017.index,
                    columns=data2017.columns)

df2018=pd.DataFrame(data2018,
                    index=data2018.index,
                    columns=data2018.columns)

df2019=pd.DataFrame(data2019,
                    index=data2019.index,
                    columns=data2019.columns)
'''
df2016=pd.DataFrame(data2016)

df2017=pd.DataFrame(data2017)

df2018=pd.DataFrame(data2018)

df2019=pd.DataFrame(data2019)


emisiones=pd.concat([df2016,df2017,df2018,df2019])

print(f'\nEMISIONES DFs:\n {emisiones.head()}\n')


# B) FILTRAR columna ESTACION,MAGNITUD, AÑO, MES y DIAS (D01,D02..)
columnas=['ESTACION','MAGNITUD','ANO','MES']
print(f'\nCOLUMNAS:\n {columnas}\n')

# añadir todas las columnas que empiezan por D (dias)
columnas.extend([col for col in emisiones.columns if col.startswith('D')])

emisiones=emisiones[columnas]

print(f'\nEMISIONES filtradas:\n {emisiones.head()}\n')


# C) REESTTRUCTURAR DF, varias columnas en una nueva ( .melt() ) (fundir))
# id_vars --> columnas originales
# var_name --> nombre nueva columna que agrupa las columnas (Si no se especifica, usa todas las que no están en id_vars)
# value_name --> nombre de la nueva columna que contendrá los datos
emisionesReest= emisiones.melt(
    id_vars=['ESTACION', 'MAGNITUD', 'ANO', 'MES'],
    var_name='DIA',
    value_name='VALOR')

print(f'\nEMISIONES reestructuradas:\n {emisionesReest.head()}\n')


# D) NUEVA COLUMNA fecha (concatenando ano,mes y dia)
# import datetime
import numpy as np

# eliminar la D de los valores de DIA (D01, etc)
emisionesReest['DIA']=emisionesReest['DIA'].str.strip('D')

# astype(str) --> convertir datos numéricos en str y concatenamos
fecha= (emisionesReest['DIA'].astype(str) + '-'+
        emisionesReest['MES'].astype(str)+'-'+
        emisionesReest['ANO'].astype(str)
        )
        

#convertir str anterior en tipo date y darle formato fecha( uso de %) y errors
# errors='coerce' --> en vez de NaN, not a time
emisionesReest['FECHA']=pd.to_datetime(fecha,format='%d-%m-%Y', errors='coerce')


# BORRAR FILAS CON FECHA NAN
# ~ (Operador NOT): Invierte el resultado booleano. 
# En lugar de seleccionar los valores nulos, selecciona los valores que no son nulos (las fechas válidas)
emisionesReest=emisionesReest[~np.isnat(emisionesReest['FECHA'])]

# emisionesReest=emisionesReest[np.isnat(emisionesReest['FECHA']=='False')]

print(f"\nFECHA: \n{emisionesReest['FECHA']}")

print(f"\nFINAL: \n{emisionesReest.head()}")

# ordenar por mes y dia
# print(f"\nFINAL: \n{emisionesRest.sort_values(by=['MES','DIA']).head()}")

# borrar columnas de año,mes y dia (quedaria fecha)
# emisionesRest=emisionesRest.drop(columns=['ANO','MES','DIA'])

print(f"\nFINAL: \n{emisionesReest.head()}")

# pd.set_option('display.width', 1000)

# Muestra las primeras 5 filas perfectamente alineadas
# print(f'FINAL TAB: \n{emisionesRest.head().to_string(index=False)}')

# %% Ej 5 emisiones continuacion

'''
5 Continuar con el ejercicio anterior para los siguientes apartados:
    
f) ordenar las columnas en este orden Estacion, Magnitud, Fecha
    
g) Función que reciba una estación, un contaminante y un rango de fechas y 
    devuelva una serie con las emisiones del contaminante dado en la estación y rango de fechas dado.
    
h) Mostrar un resumen descriptivo (min, MAX, media) para cada contaminante usando .groupby(...).VALOR.describe()
    
i) Mostrar resumen descriptivo para cada contaminante por distritos (estaciones) ampliando el código del apartado anterior
    
j) Función que reciba una estación y un contaminante y devuelva un resumen descriptivo de las emisiones del contaminante indicado en la estación indicada.
    
k) Función que devuelva emisiones medias mensuales de un contaminante y un año dados para todas las estaciones
    
l) Función que reciba estación y devuelva DataFrame con medias mensuales de los tipos de contaminantes
'''

# f) ORDENAR  las columnas en este orden Estacion, Magnitud, Fecha
# emisionesReest= emisionesReest.sort_values(by=['ESTACION','MAGNITUD','FECHA'],ascending=True)

# corregido
# ORDENAR LAS COLUMNAS (no los valores) en ese orden
columnas1=['ESTACION','MAGNITUD','FECHA']
columnas2=[col for col in emisionesReest.columns if col not in columnas1]

columnasFinal=columnas1+columnas2

emisionesReest=emisionesReest[columnasFinal]

print(emisionesReest.head())


# G)FUNCION

# def emisionFn(estacion,magnitud,fechaInicio,fechaFin):
#     filtro = ((emisionesReest['ESTACION'] == estacion) & 
#        (emisionesReest['MAGNITUD'] == magnitud) & 
#        (emisionesReest['FECHA'].between(pd.to_datetime(fechaInicio), pd.to_datetime(fechaFin))))


# corregido
def emisionFn(estacion,magnitud,fechaInicio,fechaFin):
    filtro = ((emisionesReest['ESTACION'] == estacion) & 
       (emisionesReest['MAGNITUD'] == magnitud) & 
       (emisionesReest['FECHA']>= fechaInicio) & (emisionesReest['FECHA']<= fechaFin))
    
    return emisionesReest.loc[filtro,'VALOR'].copy()
    
 
valor=emisionFn(4,1,'2016-01-01','2016-02-01')

print(f'RESULTADO funcion emisionFn: \n{valor}\n')

#%% H) RESUMEN DESCRIPTIVO (min,max,media) para cada contaminante usando .groupby(...).VALOR.describe()

# funciona tb .groupby(['MAGNITUD'])
# Solo mostrar estas operaciones --> [['min','max','mean']]

# tb valida
# resumenDescribe= emisionesReest.groupby(emisionesReest['MAGNITUD'])['VALOR'].describe()[['min','max','mean']]

resumenDescribe= emisionesReest.groupby(emisionesReest['MAGNITUD']).VALOR.describe()[['min','max','mean']]

print(f'RESUMEN DESCRIBE:\n {resumenDescribe}\n')



#%% I) Mostrar resumen descriptivo para cada contaminante por distritos (estaciones) ampliando el código del apartado anterior

resumenDescibeDistritos= emisionesReest.groupby(['MAGNITUD','ESTACION']).VALOR.describe()[['min','max','mean']]
print(f'RESUMEN DESCRIBE DISTRITOS:\n {resumenDescibeDistritos}\n')


# %% J) Función que reciba una estación y un contaminante y devuelva 
# un resumen descriptivo de las emisiones del contaminante indicado en la estación indicada

def emision2Fn (estacion,magnitud):
    filtro = (emisionesReest['ESTACION'] == estacion) & (emisionesReest['MAGNITUD'] == magnitud)
    
    resultado= emisionesReest.loc[filtro].VALOR.describe()
    return resultado

valor2= emision2Fn(4,1)

print(valor2)

# %% k) Función que devuelva emisiones medias mensuales de un contaminante y un año dados para todas las estaciones
def emisiones3Fn (magnitud, anio):
    filtro = (emisionesReest['MAGNITUD'] == magnitud) & (emisionesReest['ANO'] == anio)
    print(filtro)
    
    resultado = emisionesReest.loc[filtro].groupby(['ESTACION', 'MES']).VALOR.mean()
    
    return resultado

mediaMensual = emisiones3Fn(1, 2016)

print(f'MEDIA MENSUAL POR ESTACION Y MES:\n {mediaMensual}\n')

# %% l) Función que reciba estación y devuelva DataFrame con medias mensuales de los tipos de contaminantes

def emisiones4Fn(estacion):
    filtro = (emisionesReest['ESTACION'] == estacion)
    
    resultado = emisionesReest.loc[filtro].groupby(['MAGNITUD', 'MES']).VALOR.mean()
    
    return resultado

mediasMensualesEstacion = emisiones4Fn(4)

print(f'MEDIAS MENSUALES POR CONTAMINANTE Y MES (ESTACION 4):\n {mediasMensualesEstacion}\n')



# %% EJERCICIOS 26

'''
Desarrolla un modelo de regresión lineal múltiple para predecir el precio de las
 casas en California de acuerdo con el número de habitaciones que tiene la vivienda,
 el tiempo que ha estado ocupada y la distancia a los centros de trabajo de California.
 Estas son varias características que se tomarán en cuenta para diseñar nuestro modelo. 
 Para ello, utiliza el conjunto de datos fetch_california_housing() de los datasets de scikit-learn.

En este ejercicio vas a construir un modelo de regresión lineal para predecir
 el precio medio de viviendas en California utilizando el dataset California Housing
 disponible en sklearn. Sigue los siguientes pasos:

a)	 Importa las librerías necesarias:
•	pandas
•	fetch_california_housing desde sklearn.datasets
•	train_test_split desde sklearn.model_selection
•	LinearRegression desde sklearn.linear_model
•	mean_squared_error y r2_score desde sklearn.metrics
•	StandardScaler desde sklearn.preprocessing

b)	Carga el dataset California Housing.
 
c)	Crea un DataFrame de pandas con las variables predictoras y añade 
    una nueva columna llamada PRICE que contenga la variable objetivo 
    (usando el atributo .target de los datos cargados)
 
d)	Selecciona como variables independientes las siguientes características:
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.
 
e)	Separa el conjunto de datos en Variables predictoras (X) con las 
    caracteristicas del dataframe y Variable objetivo (y) con la columna de PRICE
 
f)	Divide los datos en conjunto de entrenamiento y prueba: 80% entrenamiento,
    20% prueba y usa random_state=42
    
'''

# a)
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# from sklearn.preprocessing import StandardScaler


# b)
# datosHousing = datasets.fetch_california_housing(as_frame=True)
datosHousing= datasets.fetch_california_housing()

print("\n ------DATASET------")
print(datosHousing)

print("\n ------Nombres de las caracteristicas COLUMNAS------")
print(datosHousing.feature_names)


print("\n ------Nombres del objetivos------")
print(datosHousing.target_names)

print("\n ------Nombres del frame------")
print(datosHousing.frame)

# print("\n ------Nombres del filename------")
# print(datosHousing.filename)

print("\n ------Descripción------")
print(datosHousing.DESCR)


# c)DATAFRAME

# hay que poner .data
# OPCION ALTERNATIVA --> dfHousing = datosHousing.frame (hay que poner (as_frame=True) )
# dfHousing = datosHousing.frame
dfHousing = pd.DataFrame(datosHousing.data,
                         columns=datosHousing.feature_names)

print("\n ------DATAFRAME HOUSING------")
print(dfHousing.head())


print("\n ------DATAFRAME HOUSING - PRICE (Columna añadida) OBJETIVO------")
# OBJETIVO
dfHousing['PRICE']=datosHousing.target
print(dfHousing.head())


#%% D VARIABLES INDEPENDIENTES

X = dfHousing[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]

# Y = dfHousing['PRICE']

print("\n ------VARIABLES INDEPENDIENTES (X)------")
print(X.head())


# %% E

# Variables predictoras (X): todas las características menos el precio
# X = dfHousing.drop('PRICE', axis=1)

# Variable objetivo (y): solo la columna del precio
Y = dfHousing['PRICE']

# Verificación de la separación
print(f"Dimensiones de X: {X.shape}") # Debería mostrar (20640, 8)
print(f"Dimensiones de y: {Y.shape}") # Debería mostrar (20640,)


# %% F

# División de los datos
# test_size=0.2 indica el 20% para prueba y, por defecto, el 80% para entrenamiento.
# random_state --> semilla (puede ser cualquier número)

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2, 
                                                    random_state=42)

# Verificación de los tamaños
print(f"Registros para entrenamiento: {len(X_train)}")
print(f"Registros para prueba: {len(X_test)}")

# %%
'''
g)	Aplica una normalización (escalado) de los datos utilizando StandardScaler:
    Ajusta el escalador con los datos de entrenamiento. Transforma tanto los datos
    de entrenamiento como los de prueba.

h)	Crea la variable modelo, instancia de LinearRegression() y usa su método fit()
    pasándole como parametros el X_train escalado y el y_train.

i)	Crea la variable ‘prediccionConjuntoPrueba’ y realiza predicciones sobre 
    el conjunto de prueba con el método .predict() usando de parametro el X_test escalado

j)	Calcula Error Cuadrático Medio (MSE) en una variable usando mean_squared_error(),
    con parametros y_test y la prediccion del conjunto de prueba. Muéstralo por pantalla

k)	 Calcula el Coeficiente de Determinación (R²) en una variable usando r2_score(),
    con parametros y_test y la prediccion del conjunto de prueba. Muéstralo por pantalla.
l)	Muestra por pantalla todos los valores (MedInc, HouseAge, AveRooms,etc) a los que
    se pueden acceder mediante el método modelo.coef_

'''

# G --<StandardScaler>--

print("\n -----STANDARDSCALER ----")
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

# SOLO 1 DATO POR FIT_TRANSFORM
X_train_scaler =scaler.fit_transform(X_train)

# sin fit el de X_TEST (solo transform)
X_test_scaler =scaler.transform(X_test)

# print(X_train_std)

# print(X_test_std)



# %% H - LINEAR REGRESSION
from sklearn import linear_model 

modelo =linear_model.LinearRegression()


modelo.fit(X_train_scaler,Y_train) 


w = modelo.coef_ 
b = modelo.intercept_ 
print(f"\nPrimer coeficiente (w1): {w[0]}") 
print(f"Intercepto (b): {b}")



#%% I 

prediccionTest = modelo.predict(X_test_scaler)

# %% J
mse = mean_squared_error(Y_test,prediccionTest)


print(f'\n MSE \n{mse}')


# %% K

coef_determinacion =r2_score(Y_test,prediccionTest)

print(f'\n coeficiente_determinacion \n{coef_determinacion}')



# %%L
for dato in modelo.coef_:
    print(dato)


# %% EJERCICIO 27

'''
27 Desarrolla un modelo de clasificación mediante Regresión Logística para predecir 
la especie de flor Iris utilizando únicamente el largo y el ancho del sépalo.
En este ejercicio vas a construir un modelo de clasificación supervisada usando
 el dataset load_iris() de scikit-learn. El objetivo es entrenar 
 un modelo de Regresión Logística, evaluarlo con métricas de clasificación y
 visualizar su frontera de decisión.
 
a) Importa las siguientes librerías:
• numpy como np
• pandas como pd
• load_iris desde sklearn.datasets
• train_test_split desde sklearn.model_selection
• LogisticRegression desde sklearn.linear_model
• StandardScaler desde sklearn.preprocessing
• accuracy_score, classification_report y confusion_matrix desde sklearn.metrics
• matplotlib.pyplot como plt
 
b) Carga el dataset Iris en una variable llamada datosIris utilizando:
datosIris = load_iris()
 
c) Crea un DataFrame llamado dfIris con:
• Las variables predictoras usando datosIris.data
• Los nombres de columnas usando datosIris.feature_names
Añade una nueva columna llamada target usando: dfIris['target'] = datosIris.target
 
d) Crea un array llamado caracteristicas con las siguientes 
    variables independientes:  'sepal length (cm)' y  'sepal width (cm)'
 
e) Separa el conjunto de datos en: variable X con dfIris[caracteristicas] y 
    variable y con dfIris['target']
 
f) Divide los datos en conjunto de entrenamiento y prueba utilizando:
    train_test_split(X, y, test_size=0.2, random_state=42)
    
'''
# A)
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

import matplotlib.pyplot as plt

# B)
datosIris=datasets.load_iris()



# C) CREAR DATASET
dfIris= pd.DataFrame(datosIris.data,
                     columns=datosIris.feature_names)

print(f'\n dfIris \n {dfIris.head()}')




# OBJETIVO nueva columna
dfIris['target']=datosIris.target

print(f'\n OBJETIVO: dfIris["target"] \n {dfIris["target"]}')



# D) variables independientes (array)
caracteristicas = ['sepal length (cm)','sepal width (cm)']

print(f'\n CARACTERISTICAS \n{caracteristicas}')

print(f'\n dfIris[caracteristicas] \n {dfIris[caracteristicas]}')


# E) VARIABLES X e Y (caracteriticas y objetivo)
X= dfIris[caracteristicas]

Y =dfIris['target']

print(f'\n X \n{X}')
print(f'\n Y \n{Y}')



# F) TRAIN TEST SPLIT
X_train,X_test,Y_train,Y_test= train_test_split(X,
                                                Y,
                                                test_size=0.2,
                                                random_state=42)



# %% 27 B

'''
g) Aplica una normalización utilizando StandardScaler:
    • Crea una instancia llamada scaler
    
    • Ajusta y transforma los datos de entrenamiento con: 
        escalaAjusteEntrenamiento = scaler.fit_transform(X_train)
        
    • Transforma los datos de prueba con: escalaAjustePrueba = scaler.transform(X_test)
 
h) Crea la variable ‘modelo’ como una instancia de LogisticRegression() 
    y entrena modelo usando modelo.fit(escalaAjusteEntrenamiento, y_train)
 
i) Realiza las predicciones sobre el conjunto de prueba utilizando:
    predicionesPrueba = modelo.predict(escalaAjustePrueba)
 
j) Evalúa la precisión del modelo calculando la variable accuracy usando:
    accuracy_score(y_test, predicionesPrueba) y mostrándola por pantalla.
 
k) Genera y muestra el informe de clasificación usando: 
    classification_report(y_test, predicionesPrueba) y la matriz de confusión
    almacenándola en una variable llamada matrizConfusion usando: 
        confusion_matrix(y_test, predicionesPrueba)
 
l) Visualización de la frontera de decisión (solo si el número de características es 2):
    • Crea una malla de puntos usando este código:
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
    • Predice las clases sobre la malla con este código
        Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
 
m)  Usa plt.contourf() para dibujar la superficie, plt.scatter() para representar
    los puntos de entrenamiento y añade etiquetas a los ejes
    y un título con plt.xlabel() y plt.title()
    
'''


# G) STANDARSCALER (normalizacion) (aprende media y desviacion tipica y despues lo transforma)

scaler= StandardScaler()

escalaAjusteEntrenamiento= scaler.fit_transform(X_train)

escalaAjustePrueba= scaler.transform(X_test)



# H) MODELO (crear y entrenar)

# modelo= linear_model.LogisticRegression()

modelo= LogisticRegression()

modelo.fit(escalaAjusteEntrenamiento,Y_train)



#I) PREDICCIONES

predicionesPrueba= modelo.predict(escalaAjustePrueba)

print(f'\n predicionesPrueba: \n {predicionesPrueba}')


    
# J) PRECISION modelo (ACCURACY)

precision=accuracy_score(Y_test, predicionesPrueba)

print(f'\n PRECISION accuracy_score: \n {precision}')




# K) INFORME CLASIFICACION

informe= classification_report(Y_test, predicionesPrueba)
print(f'\n INFORME CLASIFICACION: \n {informe}')




# K b) NATRIZ CONFUSION

matrizConfusion= confusion_matrix(Y_test, predicionesPrueba)

print(f'\n MATRIZ CONFUSION: \n {matrizConfusion}')


# L) FRONTERA de DECISION (sobre el escalado (standarScaler) del X_train)
x_min, x_max = escalaAjusteEntrenamiento[:, 0].min() - 1, escalaAjusteEntrenamiento[:, 0].max() + 1


y_min, y_max = escalaAjusteEntrenamiento[:, 1].min() - 1, escalaAjusteEntrenamiento[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))


# PREDICCION de CLASES sobre malla
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print(f'\n Z: \n {Z}')



# M) PLT (graficos)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

plt.scatter(escalaAjusteEntrenamiento[:, 0], 
            escalaAjusteEntrenamiento[:, 1], 
            c=Y_train, 
            edgecolors='k', 
            cmap='viridis')


plt.xlabel('Sepal length (estandarizado)')
plt.ylabel('Sepal width (estandarizado)')
plt.title('Fronteras de Decisión: Regresión Logística (Iris)')

plt.show()

