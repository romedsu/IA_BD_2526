# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
#import numpy as np


#%% seccion inicial
import numpy as np
a=np.array([1,2,3,4])
b=np.array([[1,2,3],[4,5,6],[7,8,9],[7,8,9],[7,8,9],[7,8,9]])

print(a)
print(a.shape)

print(b)
print(b.shape)



c=np.arange(1.0, 10.5, 0.5)
print(c)


zerosArray=np.zeros((2,3))
print("array de ceros: \n", zerosArray)

oneArray=np.ones((3,2))
print("array de unos: \n", oneArray)

identidad=np.identity(4)
print("array identidad: \n", identidad)

randomArray=np.random.random((2,3))
print("randomArray:\n", randomArray)

randomSample= np.random.ranf((4,3))
print("randomSample:\n", randomSample)

enterosRandom= np.random.randint(1,5,(3,3))
print("enterosRandom: \n", enterosRandom)

print("randomUniform (decimales en ese rango) \n", np.random.uniform(10, 50, (3,3)))


#%% seccion dos
import numpy as np
x=np.array([[1,2],[3,4]])
y=np.array([[1,2],[3,4]])

print(x)
print(y)

print("\nsuma:\n",x+y)
print("\nresta:\n",x-y)
print("\nmultiplicacion:\n",x*y)
print("\nproducto escalar:\n",np.dot(x,y))
print("\ndivision:\n",x/y)

#%% seccion matplotlib
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[10,20,24,30]

plt.plot(x,y,marker='o')

plt.title("ejemplo de grafico")
plt.xlabel("ejex")
plt.ylabel("ejey")
plt.xlim(-2,8)
plt.ylim(0,40)
plt.show()

fig,axs= plt.subplots(2, 2,figsize=(8,4))

axs[0,0].plot(x,y,marker='o')
axs[0,0].set_title("linea")
#limites axs[0,0]
axs[0,0].set_xlim(0,5)
axs[0,0].set_ylim(0,40)


axs[0,1].bar(x,y,color='y')
axs[0,1].set_title("barras")
#limites axs[0,1]
axs[0,1].set_xlim(0,5)
axs[0,1].set_ylim(0,40)

axs[1,0].scatter(x,y,color='orange')
axs[1,0].set_title("dispersion")

axs[1,1].hist(y,bins=4,color="purple")
axs[1,1].set_title("histograma")


#%% PANDAS
import pandas as pd

# ruta= r'C:\Users\Mañana\OneDrive - Consejería de Educación\Documentos\rm\SAA_MIA\u02\practica_numpy\archive\unsdg_2002_2021.csv'
ruta= r'C:\Users\romda\OneDrive - Consejería de Educación\Documentos\rm\SAA_MIA\u02\practica_numpy\archive\unsdg_2002_2021.csv'

df= pd.read_csv(ruta,sep=',')

#5 primeros registros
print("------< head() >-------")
print(df.head())
print("---------------\n")

print("------< info() >-------")
print(df.info())
print("---------------\n")

print("------< describe() >-------")
print(df.describe())
print("---------------\n")

print("------< columns() >-------")
print(df.columns)
print("---------------\n")

print("------< columns.values() >-------")
print(df.columns.values)
print("---------------\n")

print("------< index() >-------")
print(df.index)
print("---------------\n")

# print(df.info())
# print("---------------\n")

print("ordenado valores de las fechas")
print(df.sort_values("dt_year",ascending=True))
print("---------------\n")

print("usuario iloc[]")
print(df.iloc[1])
print("---------------\n")

print("usando loc[ para de_year CONDICION BOOLEANA")
df_2020=df.loc[df["dt_year"]==2020]
print(df_2020.head)
print("---------------\n")


# Eliminar filas duplicadas
print(df.drop_duplicates())
print(df.drop_duplicates(subset="dt_year"))

#no muestra si tiene campos nulos o vacios
#print(df.dropna())

print("---------------\n")
print(df.dropna(how='all'))
#susitye NaN (not a number) por 0
print(df.fillna(0))




#%%PANDAS 2
import pandas as pd

data= {
       "Name": ["Spongebob", "Patrick","Squidward"],
       "Age": [30,35,50]
       }
print(data)
print("---------------\n")

df= pd.DataFrame(data, index=["Employee1","Employee2","Employee3"])
print(df)

print("---------------\n")
print(df.loc["Employee2"])

print(df.iloc[1])

#Añadir columna
df["trabajo"] =["cocinero","nuevo","cajero"]
print(df)

#añadir fila
nuevaFila= pd.DataFrame([{"Name":"Sandy","Age":"28","Job":"Ingeniera"},
                         {"Name":"Cangrejo","Age":"60","Job":"Jefe"}],index=["Employee4","Employee5"])

df=pd.concat([df,nuevaFila])

print(df)

#%% SimpleImputer,StandarScaler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('archive/unsdg_2002_2021.csv')

dfObjetivo = df[["country", "level_of_development"]]
dfObjetivo =dfObjetivo.drop_duplicates()
print("---------------\n")

print("df normal head")
print(df.head())
print("---------------\n")

print("df Objetivo head")
print(dfObjetivo.head())
print("---------------\n")

print("-------< CARACTERISTICAS >--------\n")
dfCaracteristicas= df.drop(columns=["dt_year","dt_date", "region"])
print(dfCaracteristicas.columns)
print(dfCaracteristicas)


#AGRUPAR
print("-------< GROUPBY() >--------\n")
dfCaracteristicas=dfCaracteristicas.groupby("country").mean(numeric_only=True)
print(dfCaracteristicas)
print("---------------\n")


print("-------< SIMPLEIMPUTER() >--------\n")
imputador =SimpleImputer(missing_values=np.nan, strategy="mean")

# devuelve array (de numpy)
dfImputado = imputador.fit_transform(dfCaracteristicas)
print(dfImputado)

#convertilo a dataframe
dfImputado = pd.DataFrame(dfImputado,
                          index=dfCaracteristicas.index,
                          columns=dfCaracteristicas.columns)

print("----dfImputado---------\n")

print(dfImputado.head())
print("---------------\n")


print("-------< STANDARDSCALER() >--------\n")
#estanderizar los datos
escalador =StandardScaler()
dfEstandarizado =escalador.fit_transform(dfImputado)

#convertilo a dataframe
X = pd.DataFrame(dfEstandarizado,
                 index=dfCaracteristicas.index,
                 columns=dfCaracteristicas.columns)


print("----dfObjetivo (Y antes de set_index('country'))---------\n")
print(dfObjetivo.head())
print(dfObjetivo.iloc[1])

#prepralamos la variable objetivo Y
Y =dfObjetivo.set_index("country")

#resultados
print("\n X (caracterisitcas estandar) dataframe X STANDARIZADO")
print(X.head())

print("\n\n Y (nivel de desarrollo) dataframe Y")
print(Y.head())

print("\n\n dfObjetivo será como Y pero con otro index")
print(dfObjetivo)

print("---------------\n")

#train_test_split()
from sklearn.model_selection import train_test_split

X=X.loc[Y.index]
X_train,X_test,Y_train,Y_test = train_test_split(X,
                                                 Y,
                                                 test_size=0.30,
                                                 random_state=42
                                                 )


print("\n TAMAÑOS de los conjuntos")
print()
print("\n --- X_train ",X_train.shape)
print("\n --- X_test ",X_test.shape)
print("\n --- Y_train ",Y_train.shape)
print("\n --- Y_test  ",Y_test.shape)

# print("\n --- Y_TRAIN ",Y_train)
# print("\n --- Y_TEST  ",Y_test)
# print("\n --- X_TRAIN ",X_train)

print("---------------\n")


#%% CARGAR DATASET de SCIKIT-LEARN
from sklearn import datasets
import pandas as pd

#cargar datasets
datos_diabetes =datasets.load_diabetes()
#datos_boston = datasets.load_boston()
datos_iris=datasets.load_iris()
datos_cancer=datasets.load_breast_cancer()
datos_digits = datasets.load_digits()


#elegir datasets a explorar
datos=datos_iris
print(datos.keys())

print("\n ------Nombres de las caracteristicas------")
print(datos.feature_names)

print("\n ------Nombres del objetivos------")
print(datos.target_names)

print("\n ------Nombres del frame------")
print(datos.frame)

print("\n ------Nombres del filename------")
print(datos.filename)

print("\n ------Descripción------")
print(datos.DESCR)


print("\n------ Descripción datos digits------")
print(datos_digits.DESCR)


df_X = pd.DataFrame(datos.data,
                    columns=datos.feature_names)

print("\n -----DATAFRAME de DATOS----")
print(df_X.head())

#%% PRACTICA VIII - WALLMART

import pandas as pd
import random

dfDatos =pd.DataFrame()

def cargaDatos():
    dfDatos= pd.read_csv('wallmart/walmart_sales.csv')
    valoresPosiblesWeeklyRain =['Ninguna','Pocas',"Medias","Muchas"]
    
    weekly_Rains=[]
    
    #shape -> devuelve ->ejemplo(2,3) (filas,columnas)
    #shape[0] --> nº de filas
    #shape[1] --> nº de columnas   
    for i in range(dfDatos.shape[0]):
        weekly_Rains.append(valoresPosiblesWeeklyRain[random.randint(0,3)])
        
    dfDatos["Weekly_Rains"]= weekly_Rains
    
    #--
    valoresPosiblesWeeklyDiscounts =["Carnes","Pescados","Restos"]
    
    weekly_Discounts =[]
    
    for i in range(dfDatos.shape[0]):
        weekly_Discounts.append(valoresPosiblesWeeklyDiscounts[random.randint(0, 2)])
        
    dfDatos["Weekly_Discounts"]= weekly_Discounts
    
    #quedarse solo con un rango ¿?¿?¿
    # equivale a --> dfDatos = dfDatos[dfDatos["Store"] == 1]
    dfDatos = dfDatos[dfDatos.Store == 1]
    
    return dfDatos

dfDatos =cargaDatos()
print("\n\n Nº de filas")
print(dfDatos.shape[0])

print("\n\n Tipo de datos:")
print(type(dfDatos))

print("\n\n Dataframe dfDatos:" )
print(dfDatos)

#VERIFICAR SI NO HAY NULLs
print("\n\n Ver si hay algún null:")
print(dfDatos.isnull())

#SUMA DE LOS NULL)
print("\n\n Suma null:")
print(dfDatos.isnull().sum())


dfDatos =dfDatos.dropna(subset=['Store','Date','Weekly_Sales'])
# reinicia el index (con False guarda el vijeo index en una columna nueva)
# suele hacerse despues de borrado de elementos,filtrado, ordenamiento, etc
dfDatos=dfDatos.reset_index(drop=True)

print("\n\n Dataframe dfDatos tras dropna" )
print(dfDatos)

#imputaciones
dfDatos['Holiday_Flag']= dfDatos['Holiday_Flag'].fillna(0)
media_temp=dfDatos['Temperature'].mean()

mediana_fuel=dfDatos['Fuel_Price'].median()
dfDatos['Fuel_Price']=dfDatos['Fuel_Price'].fillna(mediana_fuel)

#moda
moda_cpi = dfDatos['CPI'].mode()[0]
dfDatos['CPI']=dfDatos['CPI'].fillna(moda_cpi)

q1_unemployment = dfDatos['Unemployment'].quantile(0.25)
dfDatos['Unemployment'].fillna(q1_unemployment)

print("\n\n Nulos tras imputación:")
print(dfDatos.isnull().sum())

print("\nMODA")
print(moda_cpi)

print("\nMEDIANA")
print(mediana_fuel)

print("\nMEDIA")
print(media_temp)


from sklearn.model_selection import train_test_split

df_X= pd.DataFrame(dfDatos,
                   columns=['Store','Date','Holiday_Flag','Temperature',
                            'Fuel_Price','CPI','Unemployment','Weekly_Rains',
                            'Weekly_Discounts'])

df_Y=pd.DataFrame(dfDatos,columns=['Weekly_Sales'])


df_X_train,df_X_test,df_Y_train,df_Y_test = train_test_split(
    df_X,df_Y,test_size=0.2,random_state=100)



print("\nCantidad de filas y columnas de X_train",df_X_train.shape)
print("\nCantidad de filas y columnas de X_train",df_X_test.shape)

print("\nCantidad de filas y columnas de Y_train",df_Y_train.shape)
print("\nCantidad de filas y columnas de Y_train",df_Y_test.shape)


print("\n\n Dataframe de df_X:")
print(df_X.head())

print("\n\n Dataframe de df_X_train:")
print(df_X_train.head())
    
print("\n\n Dataframe de df_X_test:")
print(df_X_test.head())


from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

df_datos=cargaDatos()
categorias_weekly_rains=[['Ninguna','Pocas','Medias','Muchas']]

# crear codificador
codificador_ordinal= OrdinalEncoder(categories=categorias_weekly_rains)

#ojo doble corchete
codificador_ordinal =codificador_ordinal.fit_transform(df_datos[['Weekly_Rains']])

df_nuevas_columnas_ordinal =pd.DataFrame(
    codificador_ordinal,
    columns=['Weekly_Rains_Cod'],
    index= df_datos.index)

df_datos= df_datos.join(df_nuevas_columnas_ordinal)

print("\n\n Datos Ordonal Encoder:")
print(df_datos.head())

print("\n ---------<->----------")



# --<OneHotEncoder>--
print("\n -----ONEHOTENCODER ----")
# de datos que no son numericos, pasarlos a numeros (previo a la normalizar)
from sklearn.preprocessing import OneHotEncoder
# cargamos oriignal
df_datos=cargaDatos()

# creamos objeto ONEHOTENCODER (convierte texto a bianrio 0|1)
codificador_oneHot =OneHotEncoder(sparse_output=False)

# transformamos colum categoria en matriz numerica, ojo doble [[]]
codificar_oneHot = codificador_oneHot.fit_transform(df_datos[['Weekly_Discounts']])

# creamos los nombre de las nuevas coliumnas OneHot
# categories[0] contiene lista de valores ['carnes','pescados','restos']
arr_nombre_nuevas_columnas = 'Weekly_Discounts_' + codificador_oneHot.categories_[0]

# convertimos la codificacion a DF
df_nuevas_columnas_oneHot=pd.DataFrame(codificar_oneHot,
                          columns=arr_nombre_nuevas_columnas,
                          index=df_datos.index)

df_datos=df_datos.join(df_nuevas_columnas_oneHot)

print("\n\n df_datos ONE HOT ENCODER")
print(df_datos.head())

print("\n ---------<->----------")




# --<MixMaxScaler>--
print("\n -----MIXMAXSCALER ----")
# Normalinar con MixMazScaler entre 0 y 1
# todos los valores van a estar entre 0 y 1
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df_datos=cargaDatos()
# seleccionamos solo columnas numericas a normalizar
columnas_a_normalizar=['Fuel_Price','CPI']

# creamos el escalador y lo ajustamos + tranformamos
minmax_scaler=MinMaxScaler()
arr_normalizado= minmax_scaler.fit_transform(df_datos[columnas_a_normalizar])

# convertimos de nuevo a DF
df_normalizado =pd.DataFrame(
    arr_normalizado,
    columns=['Fuel_Price_norm','CPI_norm'],
    index=df_datos.index
    )

print("MINIMO de Fuel_Price_norm:\n ",df_normalizado['Fuel_Price_norm'].min())

print("MAXIMO de CPI_norm:\n ",df_normalizado['CPI_norm'].max())

# añadimos las columnas al df_origen
df_datos =df_datos.join(df_normalizado)
df_datos[['Fuel_Price','Fuel_Price_norm']].hist(bins=30)
df_datos[['CPI','CPI_norm']].hist(bins=30)
plt.show()

print("\n ---------<->----------")



# --<StandardScaler>--
print("\n -----STANDARDSCALER ----")
from sklearn.preprocessing import StandardScaler
scaler_std= StandardScaler()

# calcula media(fit) y los aplica(transform)
arr_estandarizado=scaler_std.fit_transform(df_datos[['Fuel_Price','CPI']])

df_estandarizado=pd.DataFrame(
    arr_estandarizado,
    columns=['Fuel_Price_std','CPI_std'],
    index=df_datos.index
    )

print("MEDIA Fuel Price estanderizado :\n ",df_estandarizado['Fuel_Price_std'].mean())

print("Desviacion tipica Fuel Price  :\n ",df_estandarizado['Fuel_Price_std'].std())

df_datos[['Fuel_Price','CPI']].hist(bins=60)

plt.suptitle('ORIGINAL')
plt.show()
df_estandarizado[['Fuel_Price_std','CPI_std']].hist(bins=30)
plt.suptitle("ESTANDARIZADO")
plt.show()





