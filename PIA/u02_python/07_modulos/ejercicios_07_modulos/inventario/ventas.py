import utilidades

def gestion(producto,venta,dic):
    for d in dic: 
        if producto == d['nombre']:
            if venta <= d['cantidad']:
                d['cantidad']-=venta
                print(d)

                #llamada a la fn en utilidades.py
                utilidades.total(producto,venta,dic)
                return dic
            
        else:    
            raise AttributeError('Producto NO DISPONIBLE')
           
    '''
    except ValueError as e:
        print('Producto NO DISPONIBLE')
        print(e)
    '''
        
            
 
               
            



