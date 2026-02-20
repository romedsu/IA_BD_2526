from random import randint
from random import *

n1=randint(0,100)
n2=int(input("Introduce un nº entre el 0 y 100"))

i=1
while n1 != n2:
    i+=1
    if n2<n1:
        n2=int(input("Más alto, sube sin miedo"))
    else:
        n2=int(input("No tan alto, es menor"))
else:
    print(f"¡Buen trabajo! Lo has conseguido en {i} intentos")

        

      
