def es_bisiesto(year):
    if year < 1582:
        return None;
    elif year %4 !=0:
        return False
    elif year %100 !=0:
        return True
    elif year %400 !=0:
        return False
    else:
        return True

def dias_del_mes(year,month):
    global dias
    diasCopy= dias[:]

    if year <=0 or month > 12:
        return None
    elif es_bisiesto(year):
        diasCopy[1]=29
    return diasCopy[month-1]


def numero_dia(year,month,day):
   
    if day >31 or day <1:
        return None
    elif day > dias_del_mes(year,month):
        return None
    else:
        total=0
        for i in range(1,month):
            total+=dias_del_mes(year,i)
        else:
            total+=day
        return total
        
dias=[31,28,31,30,31,30,31,31,30,31,30,31]

print(numero_dia(2000,2,30))
