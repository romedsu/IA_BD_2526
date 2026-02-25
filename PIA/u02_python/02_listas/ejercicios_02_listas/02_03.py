lista=[25,8,15,3,7,4]
lista2=lista[:]
aux=0

#OPCION --> ordenar 1º el mayor (desde izquierda)
for i in range(len(lista)):
    for j in range(len(lista)-1):
        if lista[j] > lista[j+1]:
            aux= lista[j]
            lista[j] = lista[j+1]
            lista[j+1] = aux
            print(f"i: {i} | j: {j} - {lista}")
        else:
            continue

print(lista)

#-----------<>-------------

#OPCION --> ordenar 1º el menor (desde derecha -1)
for i in range(len(lista2)):
    for j in range(-1,-len(lista2),-1):
        if lista2[j]< lista2[j-1]:
            aux=lista2[j]
            lista2[j]=lista2[j-1]
            lista2[j-1]=aux
print(lista2)


#-----------<>-------------

 


