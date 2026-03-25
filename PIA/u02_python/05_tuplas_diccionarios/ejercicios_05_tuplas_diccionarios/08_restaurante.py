notasMenu={
    "arroz con bugre":[("ana",5),("antonio",2)],
    "bacalao":[("belarmino",3),("belen",2)]
    }

#print(notasMenu["arroz con bugre"][0][1])

def media(dic):
    dicNew={}
    valoresInforme=[]
    informe={}
    
    for key in dic.keys():
        total=0.0
        j=0
        for i in range(len(dic[key])):
            total+=dic[key][i][1]

            if dic[key][i][1] < 3:
                j+=1
                informe.update({key:j})
                
        else:
            total/=len(dic[key])
            dicNew[key]=total
    else:
        listaOrdenada=sorted(dicNew,key=dicNew.get)
        print(f"Platos con menos nota media: {listaOrdenada}")
        print(f"Notas < 3: {informe}")

    return dicNew



def agregarNotas(dic):
    plato=input("Plato  a valorar:\n")
    usuario=input("Usuario:\n")
    nota=int(input("Nota del plato:\n"))
   
    dic.update({plato :  (usuario,nota)})

    #OTRAS VERSIONES
    #dic[plato]=input("Usuario:\n"),int(input("Nota del plato:\n"))

    #dic.update({input("Plato  a valorar:\n") :  (input("Usuario:\n"),int(input("Nota del plato:\n")))})
    
    return dic

        
    
print(media(notasMenu))

print(agregarNotas(notasMenu))


