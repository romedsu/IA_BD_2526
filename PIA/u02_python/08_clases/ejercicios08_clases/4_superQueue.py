class QueueError(IndexError):
    pass


#PADRE
class Queue:
    def __init__(self):
        self.__pila=[]

    def __str__(self):
        self.__str=f"OBJETO: {self.__pila}"

        return self.__str
        
        

    def put(self,elemento):
        self.elemento= elemento
        self.__pila.append(self.elemento)

        return self.__pila


    def get(self):
        if not self.__pila:
            raise QueueError("Objeto vacío")
        else:
            self.__primero=self.__pila[0]
            self.__pila.pop()
            
            return self.__primero

#HERENCIA
class SuperQueue(Queue):
    
    def vacia(self):
        print(f"hijo: {self._Queue__pila}")


        #como llamar al objeto solo (__str__)
        print(f"hijo: {self}")

        print(self.__dict__)

        #llamar al objeto a través del nombre de la clase padre
        print(len(self._Queue__pila))
        
        
        if len(self._Queue__pila)==0:
            return True
        else:
            return False
        
              
        
cola=SuperQueue()

#cola.put('eo')

print(cola.__dict__)

print(cola.vacia())

#print(cola)

