'''
def promedio(nombre,dic):
    total=0
    for d in dic:
        if nombre in d:
            for key in d.keys():
                print(key)
                for a in d[key]:
                    print(f"{a[0]}: {a[1]}")
                    total+=a[1]
                media=round(total/len(d[key]),2)
                return media
    raise ValueError("Alumno no encontrado")
'''


def promedio(nombre,dic):
    total=0

    for d in dic:
        for key in d.keys():
            if nombre in d[key]:
                for a in d['notas']:
                    
                    #OPCION A
                    if type(a[1])!= float and type(a[1])!= int :
                        raise TypeError('Tipo de nota no válido')
                    '''
                    if a[1]>10 or a[1]<0:
                        raise ValueError('Nota no válida')
                    '''                            
                    print(a[1])
                    total+=a[1]
         
                    #OPCION B
                    assert a[1]>0 and a[1]<10, 'Nota debe ser un nº positivo y menor a 10'
                
                    
                media=round(total/len(d['notas']),2)
                return media
    raise ValueError("Alumno no encontrado")
    
                  
    





