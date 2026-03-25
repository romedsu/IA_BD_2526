#!/usr/bin/env python 3
# linea shabang, shebang, hashbang o poundbang 

import basics


#OPCION A
#CON from sys import path
#Modifica sys.path (lista de rutas modulos en python)
#módulo en carpeta diferente
#no necesita __init__.py
#con IMPORT --> asi se puede acceder a la global __counter (con utils.__counter)
'''
from sys import path
path.append('.\\lib')

import utils
'''

#OPCION B
#CON __init__.py (archivo vacio en la carpeta)
#se indica que lib es un paquete, en un directorio diferente
#(pero mismo nivel que el script que se ejecuta)
#con FROM --> asi se necesita una fn que devuelva el variable global __counter

from lib import utils


x='lolo'

numeros=['lolo',4,3,3]

#__counter=0

if __name__ == "__main__":
    try:
        print(f"FACTORIAL PROPIA MAIN: {basics.factorialPropia(x)}")

    except Exception as e:
        print(e)

    try:
        print(f"SUMA: {utils.suma(numeros)}")
    except Exception as e:
        print(e)

        print(f"PRODUCTO: {utils.producto(numeros)}")

        print(f"CONTADOR: {utils.contador()}")

    except Exception as e:
        print(e)

