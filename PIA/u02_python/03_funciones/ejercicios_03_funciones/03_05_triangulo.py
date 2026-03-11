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

print(es_rectangulo(7,24,25))

    
