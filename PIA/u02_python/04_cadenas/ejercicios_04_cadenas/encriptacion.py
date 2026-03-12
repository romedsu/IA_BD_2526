                #----<ENCRIPTACIÓN>----

#----<ENTRADA por TECLADO>----
'''
msg= input("Introduce un mensaje para encriptar:\n")
key= input("Introduce una clave:\n"
            "(Debe contener al menos 15 caraceteres,mayúscula,minúscula y número)\n")
'''

#------<>------

#----<VALIDAR CLAVE>----
#Recoge clave y valida los requisitos

def validate(key):
    minuscula=mayuscula=especial=digito= False
    
    if len(key) < 15:
        print("ERROR en CLAVE: Debe tener un mínimo de 15 caracteres")
        return False
    else:
        for i in range(len(key)):
            if key[i].islower():
                minuscula= True
            elif key[i].isupper():
                mayuscula= True
            elif key[i].isdigit():
                digito= True
            else:
                 especial= True

            #print (minuscula,mayuscula,especial,digito)
            
    if minuscula and mayuscula and especial and digito:
        return True
    else:
        if not minuscula:
            print("ERROR en CLAVE: Falta minúscula")
        if not mayuscula:
            print("ERROR en CLAVE: Falta mayúscula")
        if not especial:
            print("ERROR en CLAVE: Falta caracter especial")
        if not digito:
            print("ERROR en CLAVE: Falta número")
            
        return False

#------<>------
    

#----<CALCULAR CLAVE>----
#Calcula un entero a partir de la clave para encriptar el mensaje
    
def calcularKey(key):
    valorKey= 0
    
    if validate(key): 
        for i in range(len(key)):
            valorKey+= ord(key[i])
        
        else:
            valorKey%= ord(key[5])
            valorKey= round(valorKey/4)

        return valorKey
    
    else:
        print("Clave NO válida")

#------<>------
        


#----<TRANSFORMACION>----
'''
A través de un bucle de 4 pasos, transforma cada caracter del mensaje
en otro caracter distino (almacenada en una lista), a partir del valor de la clave.
Cada iteracción del bucle, un algoritmo diferente.
'''

def transformacion(valorKey,msgNew):
    valor= 0

    for j in range(4):
        if j== 0:
            for i in range(len(msgNew)):
                valor= ord(msgNew[i])+ valorKey
                msgNew[i]= chr(valor)
                
        elif j== 1:
            for i in range(len(msgNew)):
                valor= ord(msgNew[i])* valorKey
                msgNew[i]= chr(valor)
                
        elif j== 2:
            for i in range(len(msgNew)):
                valor= ord(msgNew[i])- valorKey +1
                msgNew[i]= chr(valor)
                
        elif j == 3:
            for i in range(len(msgNew)):
                valor= ord(msgNew[i])+5
                msgNew[i]= chr(valor)

    else:
        return msgNew

#------<>------

    
#----<PERMUTACIÓN>----
'''
A través de un bucle de 4 pasos, permuta la posición de la lista
por otra posición, a partir del valor de la clave.
Cada iteracción del bucle, un algoritmo diferente
Uso de lista auxiliar donde se va guardando el resultado
'''
def permutacion(valorKey,msgNew):
    msgAux=[]
   
    for j in range(4):
        valorKey2= 0
        
        if j== 0:
            for i in range(len(msgNew)):
                valorKey2=(valorKey+i)% len(msgNew)
                msgAux.append(msgNew[valorKey2])
                          
              
        elif j== 1:
            for i in range(len(msgNew)):
                valorKey2=((valorKey-3)+i)% len(msgNew)
                msgAux.append(msgNew[valorKey2])
           
                
        elif j == 2:
            for i in range(len(msgNew)):
                valorKey2=((valorKey+4)+i)% len(msgNew)
                msgAux.append(msgNew[valorKey2])
            
                
        else:
            for i in range(len(msgNew)):
                valorKey2=((valorKey-2)+i)% len(msgNew)
                msgAux.append(msgNew[valorKey2])
                            
        msgNew=msgAux[:]
        msgAux=[]
                
    else:
        return msgNew

#------<>------

#----<DESPERMUTACIÓN>----
'''
Devolvemos la posición previa a la permutación, en el orden inverso del bucle
de permutar.
Uso de una copia con lista auxuliar para realizar la despermutación
'''
def despermutar(valorKey,codeNew):
    codeAux= codeNew[:]

    for j in range(4):
        valorKey2=0

        if j== 0:
            for i in range(len(codeNew)):
                valorKey2=((valorKey-2)+i)% len(codeNew)
                codeNew[valorKey2]= codeAux[i]

        elif j== 1:
            for i in range(len(codeNew)):
                valorKey2=((valorKey+4)+i)% len(codeNew)
                codeNew[valorKey2]= codeAux[i]
                
        elif j== 2:
            for i in range(len(codeNew)):
                valorKey2=((valorKey-3)+i)% len(codeNew)
                codeNew[valorKey2]= codeAux[i]
                
                
        else:

            for i in range(len(codeNew)):
                valorKey2=(valorKey+i)% len(codeNew)
                codeNew[valorKey2]= codeAux[i]
                        
                
        codeAux= codeNew[:]
         
    else:
        return codeNew

#------<>------


#----<DESTRANFORMAR>----
'''
Realiza el paso contrario a la transformación en el orden inverso del bucle.
Cada algoritmo es el inverso a la transformación
'''
def destransformar(valorKey,codeNew):
    valor=0

    for j in range(4):
        if j== 0:
            for i in range(len(codeNew)):
                valor=ord(codeNew[i])-5
                codeNew[i]=chr(valor)
                
        if j== 1:
            for i in range(len(codeNew)):
                valor=ord(codeNew[i]) +valorKey-1
                codeNew[i]=chr(valor)

        if j==2:
            for i in range(len(codeNew)):
                valor=ord(codeNew[i]) /valorKey
                codeNew[i]=chr(round(valor))

        if j== 3:
            for i in range(len(codeNew)):
                valor=ord(codeNew[i]) -valorKey
                codeNew[i]= chr(valor)
                

    else:
        return codeNew
            
#------<>------


#----<ENCRIPTACION>----
'''
Función principal  de encriptación con llamadas internas
a tranformación y permutación
'''
def encrytion(msg,key):
    valorKey=0
    msgNew=[]
    code=''

    valorKey= calcularKey(key)

    #crea lista a partir de la cadena del mensaje
    msgNew=list(msg)

    msgNew=transformacion(valorKey,msgNew)

    msgNew=permutacion(valorKey,msgNew)
    #print("FINAL ENCRIPTADO",msgNew)

    #crea cadena a partir de la lista encriptada
    code=''.join(msgNew)
          
    return code
    
#------<>-----


#----<DECODIFICAR>-----
'''
Función principal  de desencriptación con llamadas internas
a destranformación y despermutación
'''
def decode(code,key):
    valorKey=0
    codeNew=[]
    decode=''

    valorKey=calcularKey(key)

    codeNew=code[:]
    #crea lista a partir de la cadena encriptada
    codeNew=list(codeNew)
    
    codeNew=despermutar(valorKey,codeNew)

    codeNew=destransformar(valorKey,codeNew)

    #crea cadena a partir de la lista desencriptada
    decode=''.join(codeNew)
    return decode
                           

#----<>----

#----<LLAMADAS>----
#llamadas a funciones principales (con input por teclado)
'''
if validate(key):
    code=encrytion(msg,key)
    print(f"ENCRIPTADO: {code}")

    desencriptado=decode(code,key)
    print(f"DESENCRIPTADO: {desencriptado}")
'''
        
#----<>----

#----<CASOS DE PRUEBA>----

testKey=['0987654321?BarrasVerdes','colchonetas2--Negras',
          'cuadroAcolor!555Amarillo','Sabiduria_Impopular33',
          'Granada!Significa2cosaso3_','BarakaldoNoEsUn@PuebloAndaluz22']

testMsg=['hola Patata','Camaron De LaMancha',
          'CabalgandoPorAlaska','33revolucionesPor_Minuto',
          '9876543210kylian-Mbappe','ramonRamirezSeHaMarchadoysus2Hijos']

testCode=''
testDecode=''

for i in range(len(testMsg)):
    print(f"\t -- CASO PRUEBA {i} --")
    
    print(f"Mensaje Original[{i}]: {testMsg[i]}")
   
    testCode=encrytion(testMsg[i],testKey[i])
    print(f"Mensaje Encriptado[{i}]: {testCode}")

    testDecode=decode(testCode,testKey[i])

    if testMsg[i]== testDecode:
        print("\n\t -- Encriptado correcto --")
        print(f"Mensaje Desencriptado[{i}]: {testDecode}")
    
    else:
        print("El mensaje original y el encriptado NO coinciden")

    print("\n------<>-----\n")
               
     
               

               



                     


