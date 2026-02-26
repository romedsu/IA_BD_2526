
lista=[1, 2, 4, 4, 1, 4, 2, 6, 2, 9]
#lista=[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]


aux=0

for i in range(len(lista)-1):
    print(f"i={i}")
    borrar=[]
    for j in range(i+1,len(lista)):
        #print(f"j -->lenLista= {len(lista)}")
        print(f"i={i} | j={j}")
        if lista[i] == lista[j]:
            print(lista)
            print(lista[j])
            borrar.append(lista[j])
   
    for k in range(len(borrar)):
        print(f"k={k}")
        print(f"borrar={borrar}")
        lista.pop(k)
                
        print(lista)
        print("------")
             
            
print(lista)

'''print(len(lista))
for j in range(1,len(lista)):
    print(j)'''
    


        
 
           
            

