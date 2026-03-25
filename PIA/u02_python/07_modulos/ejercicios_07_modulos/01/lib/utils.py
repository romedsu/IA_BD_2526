#!/usr/bin/env python 3

#variable privada --> las que empiezan por __
__counter=0

def suma(lista):
    global __counter

    if __name__=='__main__':
        __counter+=1

    total=0
    for n in lista:
        if type(n)!= int and type(n)!= float:
            raise TypeError ("ERROR: valores numéricos como parámetros")
        total+=n
    return total

    '''
    except TypeError:
        print("ERROR: valores numéricos como parámetros")
    '''

#print(suma([2,4,3]))

def producto(lista):
    global __counter

    if __name__=='__main__':
        __counter+=1
    try:
        total=1
        for n in lista:
            total*=n
        return total

    except TypeError:
        print("ERROR: valores numéricos como parámetros")


def contador():
    return __counter

if __name__=='__main__':
    test=[8,13,5]

    print(f"SUMA TEST: {suma(test)}")

    print(f"PRODUCTO TEST: {producto(test)}")

    
