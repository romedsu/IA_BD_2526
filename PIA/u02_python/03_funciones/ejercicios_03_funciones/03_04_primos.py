def primo(n):
    if n<=1:
        return False
    else:
        for i in range(2,n-1):
            if n%i == 0:
                return False
        else:
            return True
    

print(primo(7))

print("----")
primosLista =[2,3,5,7,11,13,17,19]
primosBooleanos =[True,True,True,True,True,True,True,True]

for i in range(len(primosLista)):
    if primo(primosLista[i]) == primosBooleanos[i]:
        print(True)
    else:
        print(False)
    
                
