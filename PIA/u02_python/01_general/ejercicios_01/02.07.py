palabra=input("Introduce una palabra")

while palabra != "chupacabra":
    palabra=input("No has acertado. Introduce otra palabra")
    if palabra == "chupacabra":
        print("Has dejado el bucle con éxito")
        break
