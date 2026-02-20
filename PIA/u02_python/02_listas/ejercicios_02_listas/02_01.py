#EJERCICIO 1

lista1= [1,2,3,4,5,6,7,8,9,10]
lista2= []

for i in range(-1,(-len(lista1))-1,-1):
    lista2.append(lista1[i])

print(lista2)

#------

#COMPRENSIÓN
'''lista1= [1,2,3,4,5,6,7,8,9,10]


lista2=[i for i in lista1[-1::-1]]
print(lista2)'''

#-----

