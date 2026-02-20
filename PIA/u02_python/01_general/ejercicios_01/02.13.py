c0=int(input("Introuce cualquier nº entero > a 0"))
i=0

while c0 != 1:
    i+=1
    if c0 %2 == 0:
        c0/=2
    else:
        c0=3*c0+1
    print(round(c0),end=" - ")    
else:
    print(f"Pasos: {i}")
