from random import *

#1) Inicializar matriz con todos los valores a 0
    #31 filas (dias) | 24 columnas (horas)
valores=[[0.0 for j in range(24)]for i in range(31)]


#2)Rellenar con valores
valores=[[round(uniform(-10.0,50.0),2) for j in range(24)] for i in range(31)]

print(valores)

#3) Temperatura media mensual
total=0
for i in range(len(valores)):
        total+=valores[i][12]
else:
    total/=len(valores)
    total=round(total,2)
    print(f"La temperatura media a las 12 del medidiodia es: {total}ºC")


#4)Máxima del mes
maxi=0
for i in range(len(valores)):
        for j in range(len(valores[i])):
                
                ''' if max(valores[i]) > maxi:
                        maxi=max(valores[i])'''
                
                #print(len(valores[i]))
                #print(f"j={j}")
                
                if valores[i][j] > maxi:
                        maxi=valores[i][j]

else:
        
print(f"La temperetarua máxima del mes es: {maxi}ºC)




