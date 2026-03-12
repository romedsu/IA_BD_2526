diccionario={}

# --<LEER>--
def leer_diccionario(diccionario):
    for key,valor in diccionario.items():
        print(key,":",valor)

# --<MAYOR>--
def oldest(diccionario):
    edadMayor=0
    mayor={}
   
    for nombre,edad in diccionario.items():
        if edad > edadMayor:
            mayor.clear()
            mayor[nombre]=edad
            edadMayor=edad
          
    print(f"MAYOR: {mayor}")
    for key,valor in mayor.items():
        print(key,"con",valor,"años, es el usuario de mayor edad")
        

# --<MENOR>--
def youngest(diccionario):
    edadMenor=100
    menor={}

    for nombre,edad in diccionario.items():
        if edad < edadMenor:
            menor.clear()
            menor[nombre]=edad
            edadMenor=edad
            
    print(f"MENOR: {menor}")
    for key,valor in menor.items():
        print(f"{key} con {valor} años es el usuario con menor edad")
    
# -----<>-----


nombre=input("Introduce el nombre\n")
edad=int(input("Introduce la edad\n"))


while edad != 0:
    diccionario[nombre]=edad
    
    nombre=input("Introduce el nombre\n")
    edad=int(input("Introduce la edad\n"))

    
 
leer_diccionario(diccionario)
oldest(diccionario)
youngest(diccionario)

