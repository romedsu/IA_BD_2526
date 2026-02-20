palabra=input("Introduce una palabra")
palabraUp=palabra.upper()


for letra in palabraUp:
    if not(letra == "A" or letra == "E" or letra == "I" or letra=="O" or letra=="U"):
        print(letra,end="")
    else:
        continue
        
                    

