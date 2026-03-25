productos={
    "teclado": [(48.5),(3),(1)],
    "raton": [(23.95),(3),(2)],
    "monitor": [(174.95),(4),(2)],
    "alfombrilla": [(6.0),(10),(1)],
    }

#print(productos["teclado"][0])

def valorInventario(dic):
    totalProducto={}
    totalInventario=0
    #total=0
    
    for key in dic.keys():
        #print(dic[key][0])
        total=round(dic[key][0] *dic[key][1],2)
        totalProducto[key]=total
        totalInventario+=total

    else:     
        totalProducto["totalInventario"]=totalInventario

        return totalProducto

def agregarProductos(dic):
    producto=input("Nombre del nuevo producto:")
    valores=[]
    
    '''
    #OPCION por VARIABLES
    pvp=float(input("Precio:"))
    stock=input("Stock:")
    stockMin=input("Stock Mínimo:")

    dic[producto]=pvp,stock,stockMin
    '''

    '''
    #OPCION por LISTA
    valores.append(float(input("Precio:")))
    valores.append(input("Stock::"))
    valores.append(input("Stock Mínimo:"))

    dic[producto]=valores
    '''

    #OPCION por DICCIONARIO
    dic[producto]=float(input("Precio:")),input("Stock:"),input("Stock Mínimo:")
        
    return dic

def controlAsistencia(dic):
    informe={}
    valoresInforme=[]
    for key in dic.keys():
        if dic[key][2] > dic[key][1]:
            valoresInforme.append(key)
            
    else:
        if valoresInforme:
            informe["Fuera de Stock Mínimo"]=valoresInforme
        else:
            informe["Fuera de Stock Mínimo"]="No hay productos"
        
        return informe
            
        
print(valorInventario(productos))

#print(agregarProductos(productos))

print(controlAsistencia(productos))





