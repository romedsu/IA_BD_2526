pedro={
    'lengua':[5.65,3.80,5.10],
    'matemáticas':[7.25,5.45],
    'física':[6.75,8.25],
    'inglés':[8.50,6.20,6.75]
    }

def notasAlumno(alumno):
    total=0
    print("---NOTAS ALUMNO---")
    for key,valor in alumno.items():
        print(f"{key}:{valor}")
        #print(alumno[key])
        for i in range(len(alumno[key])):
                       #print (alumno[key][i])
                       total+=alumno[key][i]
        else:
            media=total/len(alumno[key])
            print(f"{key} NOTA MEDIA: {round(media,2)}")
            print("--<>--\n")

notasAlumno(pedro)

'''
print(pedro)
print(pedro['lengua'])
print(pedro['lengua'][0])
'''

