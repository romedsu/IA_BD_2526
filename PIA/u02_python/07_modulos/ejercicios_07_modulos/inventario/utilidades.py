def total(producto,cantidad,dic):
    total=0
    try:
        for d in dic:
            if producto == d['nombre']:
                total= d['precio']*cantidad
                print(f'PRECIO TOTAL: {total}')
                return total
    except:
        print("ERROR")
                
           


