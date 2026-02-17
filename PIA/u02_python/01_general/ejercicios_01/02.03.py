# EJERCICIO 3

ingreso=float(input("Introduce el ingreso"))
impuesto=.0

if ingreso <= 85528:
    impuesto=(ingreso*1.18)-556.2
else:
    impuesto=14839.2+((ingreso-85528)*1.32)


print(f"El impuesto es {impuesto} €")
