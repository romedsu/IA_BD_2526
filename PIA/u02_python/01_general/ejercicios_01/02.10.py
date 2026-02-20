n=int(input("Ingresa un nº entero"))
x=0

for i in range(1,n+1):
    x+=1/i
else:
    print(round(x,2))
