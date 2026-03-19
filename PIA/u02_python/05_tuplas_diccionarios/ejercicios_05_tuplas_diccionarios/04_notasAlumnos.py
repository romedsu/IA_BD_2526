alumnos={
    "Ana":[('Matemáticas',5.5),('Lengua',6.7),('Historia',8.2),('Inglés',7.1)],
    "Belarmino":[('Matemáticas',8.9),('Lengua',6.8),('Historia',7.7),('Inglés',6.9)]
    }

def notaMedia(alumnos):
    alumnosMedia={}

    for key in alumnos.keys():
        #print(key)
        total=0
        media=0
        aMayor=''
        nMayor=0
        
        for i in range(4):
            #print(alumnos[key][0][1])
            total+=alumnos[key][i][1]

            if alumnos[key][i][1] > nMayor:
                nMayor= alumnos[key][i][1]
                aMayor=alumnos[key][i][0]        
        else:
            media=total/4
            print(f"{key}: {round(media,2)} nota media.\nNota más alta| {aMayor}:{nMayor}")
            alumnosMedia[key]=round(media,2),aMayor,nMayor

    return alumnosMedia
        
print(notaMedia(alumnos))



