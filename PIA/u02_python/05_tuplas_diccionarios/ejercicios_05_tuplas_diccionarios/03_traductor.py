vocabulario={
    'perro':'dog',
    'gato':'cat',
    'oso':'bear',
    'pez':'fish'
    }

frase="pajaro pez perro oso topo gato"




def traductor(frase,vocabulario):
    listaFrase=frase.split()

    traduccion=[]
    traduccionCadena=''
    
    
    for i in range(len(listaFrase)):
        if listaFrase[i] in vocabulario:
            traduccion.append(vocabulario[listaFrase[i]])
        else:
            traduccion.append(listaFrase[i])
                
    else:
        traduccionCadena=' '.join(traduccion)
        print(traduccionCadena)
   
       
            
        

traductor(frase,vocabulario)


