class QueueError(IndexError):
    pass


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
        


cola=Queue()


cola.put("hola")

print(cola.put("caracola"))

cola.put("adios")

print(cola.__dict__)



try:
    print(cola.get())
except QueueError as e:
    print(f"ERROR: {e}")


print(cola.__dict__)

print(cola)


    
