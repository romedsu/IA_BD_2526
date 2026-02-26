lista=[1, 2, 4, 4, 1, 4, 2, 6, 2, 9]

nueva=[]

for i in range(len(lista)):   
   if lista[i] not in nueva:
       nueva.append(lista[i])
        
else:
    print(nueva)


#------<>---------
#OPCION POR TECLADO
from random import *

total=int(input("Introduce la cantidad total de valores"))

maxi=int(input("Introduce el nº máximo para la lista"))

lista2= [randint(0,maxi)for i in range(total)]
nueva2=[]

for i in lista2:
    if i not in nueva2:
        nueva2.append(i)
    
print(lista2)
print(nueva2)
nueva2.sort()
print(nueva2)




