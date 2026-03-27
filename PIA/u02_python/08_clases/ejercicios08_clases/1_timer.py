class Timer:
    def __init__(self,hr=0,mt=0,sec=0):
        self.__hora= hr
        self.__minutos= mt
        self.__segundos= sec

    
    def __str__(self):
        #print(self.hora)
        self.completa=str(self.__hora)+':'+str(self.__minutos)+':'+str(self.__segundos)
        return self.completa
       
        
    def next_second(self):
        if self.__segundos<58:
            self.__segundos +=1
        else:
            self.__segundos=0
            
            if self.__minutos<58:
                self.__minutos +=1
            else:
                self.__minutos =0

                if self.__hora <22:
                    self.__hora +=1
                else:
                    self.__hora=0

    def prev_second(self):
        if self.__segundos >0:
            self.__segundos -=1
        else:
            self.__segundos =59

            if self.__minutos >0:
                self.__minutos -=1
            else:
                self.__minutos =59

                if self.__hora > 0:
                    self.__hora-=1
                else:
                    self.__hora=23
            
        

timer=Timer(23,59,59)

print(timer._Timer__hora)

print(timer)

timer.next_second()
print(timer)

'''
timer=Timer(5,32,59)
print(timer)
timer.next_second()
print(timer)
'''

'''
timer=Timer(23,59,59)
print(timer)
'''
timer.prev_second()
print(timer)


'''
  if self.__hora <22:
            self.__hora +=1
        else:
            self.__hora=0

        if self.__minutos<58:
            self.__minutos +=1
        else:
            self.__minutos =0

        if self.__segundos<58:
            self.__segundos +=1
        else:
            self.__segundos=0
'''


