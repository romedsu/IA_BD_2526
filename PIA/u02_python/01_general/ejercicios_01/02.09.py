palabra=input("Introduce una palabra")

palabraUp=palabra.upper()
word_withouth_vowels=""

for letra in palabraUp:
    if not (letra == "A" or letra =="E" or letra == "I" or letra == "O" or letra == "U"):
        print(letra,end="")
    else:
        word_withouth_vowels+=letra
else:
    print("\nVocales devoradas:",word_withouth_vowels)
        
