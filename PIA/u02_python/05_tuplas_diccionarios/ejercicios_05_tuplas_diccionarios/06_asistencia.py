comportamiento=[ 'bueno', 'regular','malo']

estudiantes={
    "ana": [("matematicas",True,comportamiento[1]),
            ("lengua",True,comportamiento[0]),
            ("historia",False,comportamiento[2]),
            ("ingles",True,comportamiento[0]) 
            ],
    "belen": [("matematicas",True,comportamiento[2]),
            ("lengua",True,comportamiento[1]),
            ("historia",True,comportamiento[0]),
            ("ingles",False,comportamiento[2]) 
            ],
    "carlos": [("matematicas",False,comportamiento[2]),
            ("lengua",True,comportamiento[0]),
            ("historia",False,comportamiento[2]),
            ("ingles",True,comportamiento[1]) 
            ]
    }

print(estudiantes["ana"][0][0])


def asistencia(estudiantes):
    estudiantesAsistencia={}
    
    for key in estudiantes.keys():
        porcentaje=0
        comportamiento=0
        
        for i in range(len(estudiantes[key])):
            if estudiantes[key][i][1]:
                       porcentaje+=1

            if estudiantes[key][i][2] == "malo":
                comportamiento+=1
        else:
            porcentaje/=len(estudiantes[key])/100
            print(f"{key}|  media asistencia: {round(porcentaje)}%")
            #estudiantesAsistencia[key]=porcentaje

            comportamiento/=len(estudiantes[key])/100
            
            if comportamiento > 25:
                estudiantesAsistencia[key]=porcentaje,"malo"
            else:
                estudiantesAsistencia[key]=porcentaje,"aceptable"
                    

            
                
                

    return estudiantesAsistencia
        
        

        #key["asistencia"]
        
print(asistencia(estudiantes)      )
