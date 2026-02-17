#EJERCICIO 2

millas=float(input("Introduce la cantidad de millas"))

km=float(input("Introduce la cantidad de kilómetros"))

millasConv=millas*1.61
millasConv=round(millasConv,2)

kmConv=km/1.61
kmConv=round(kmConv,2)

print(millas ," millas son " ,millasConv," kilómetros")

print(km, " kilómetros son ", kmConv," millas")
