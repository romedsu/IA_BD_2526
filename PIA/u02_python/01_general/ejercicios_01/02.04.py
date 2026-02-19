anio=int(input("Introduce un año"))

if anio <= 1582:
    print("Fuera del período del calendario Gregoriano")
else:
    if anio %4 !=0:
        print("Año común")
    elif anio %100 !=0:
        print("Año bisiesto")
    elif anio %400 !=0:
        print("Año común")
    else:
        print("Año bisiesto")


