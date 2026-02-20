n=int(input("Introduce el nº total de bloques utilizados"))

i=1

while n>i:
     print(f"n={n} ; i={i}")
     n-=i
     if i>=n:
         continue
     i+=1
     
else:
    print(i)
