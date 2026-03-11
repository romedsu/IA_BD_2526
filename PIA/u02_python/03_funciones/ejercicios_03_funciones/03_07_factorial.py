def factorial(n):
    total=1
    for i in range(2,n+1):
        total*=i
    else:
        return total


#n=int(input("Introduce un nº entero para calcular su factorial"))
#print(factorial(n))

test_data=[1,2,3,4,5,6,7,8,9,10]
test_results=[1,2,6,24,120,720,5040,40320,362880,3628800]

for i in range(len(test_data)):
    if factorial(test_data[i]) == test_results[i]:
        print (True)
    else:
        print (False)

print("-----<>------")

def factorial_rec(n):
    if n==1:
        return n
    else:
        return n*(factorial_rec(n-1))

print(factorial_rec(5))
    
    
