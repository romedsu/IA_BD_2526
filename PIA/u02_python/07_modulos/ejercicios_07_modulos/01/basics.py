#!/usr/bin/env python 3 
import math

x=5
#testFactorial=[(3,6),(5,120),(15,1307674368000)]
testFactorial=[('pepe',6),(5,120),(15,1307674368000)]

def factorialPropia(n):
    total=1

    if type(n) != int:
         raise TypeError("ERROR FACTORIAL PROPIA: Introduce un valor entero numérico")
     
    for i in range(2,n+1):
        total*=i
    return total
    
   
print(__name__)

if __name__ =='__main__':
    try:
        print(f"FACTORIAL PROPIA: {factorialPropia(x)}")  
    except TypeError as e:
        print(e)
    
    
    try:
        print(f"FACTORIAL MATH: {math.factorial(x)}")
    except TypeError:
        print("ERROR FACTORIAL MATH: Introduce un valor entero numérico")


    try:
        for i in range(len(testFactorial)):
            if type(testFactorial[i][0])!= int or type(testFactorial[i][1])!= int :
                raise TypeError("ERROR: Introduce un valor entero numérico")
            
            if factorialPropia(testFactorial[i][0])==testFactorial[i][1]:
                print("Correcto")
            else:
                print("Incorrecto")
    except TypeError as e:
        print(e)

        
#-- <> --
'''
def es_triangulo(a,b,c):
    if (a +b > c) and (a+c > b) and (b+c > a):
        return True
    else:
        return False
    

def es_rectangulo(a,b,c):
    if es_triangulo(a,b,c):
        if a**2 + b**2 == c**2:
            return True
        elif b**2 + c**2 == a**2:
            return True
        elif a**2 + c**2 == b**2:
            return True
        else:
            return False
    else:
        return False

#print(es_rectangulo(7,24,25))

print(math.hypot(7,24,25))
'''

'''
def factorialPropia(n):
    try:
        total=1
        for i in range(2,n+1):
            total*=i
        return total
    
    except TypeError:
        print("Introduce un valor entero numérico")

    except Exception as e:
         print("ERROR",e)
'''
