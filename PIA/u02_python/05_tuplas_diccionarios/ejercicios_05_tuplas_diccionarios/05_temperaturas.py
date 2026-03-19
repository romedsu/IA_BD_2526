temperaturas={
    "lunes":[11.3,16.6],
    "martes":[10.7,15.2],
    "miercoles":[12.1,17.3],
    "jueves":[9.8,14.4],
    "viernes":[11.7,15.8],
    "sabado":[12.5,36.6],
    "domingo":[11.7,27.9]
    }

def analisis(temperaturas):
    tFrio=100
    dFrio=''

    tCalido=-100
    dCalido=''

    total=0
    media=0
    amplitud=[]
    tAmplitud=0

    for key,valor in temperaturas.items():
        if valor[0] < tFrio:
            tFrio = valor[0]
            dFrio=key

        if valor[1] > tCalido:
            tCalido= valor[1]
            dCalido = key

        if valor[1] - valor[0] > 15:
            amplitud.append(key)      

        total+=(valor[0]+valor[1])/2
   
    else:
        media=total/7
        print(f"Día más frío: {dFrio} con mínima de {tFrio}º")
        print(f"Día más cálido: {dCalido} con máxima de {tCalido}º")
        print(f"Temperatura media semana: {round(media,2)}º")

        if amplitud:
            print(f"Dias con amplitud térmica:",end=' ')
            for dia in amplitud:
                print(dia,end=', ')        
            
        else:
            print("No hay días con amplitud térmica")




analisis(temperaturas)
