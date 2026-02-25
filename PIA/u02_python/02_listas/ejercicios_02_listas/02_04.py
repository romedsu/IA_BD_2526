lista=[25,8,15,3,7,4]
mayor=0

#OPCION A
for i in range(len(lista)):
    if lista[i] > mayor:
        mayor= lista[i]
else:
    print(mayor)
    

#OPCION B
for i in lista:
    if i > mayor:
        mayor = i
else:
    print(mayor)
