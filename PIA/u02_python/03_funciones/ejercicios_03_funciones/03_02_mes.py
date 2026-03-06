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
    dias=[31,28,31,30,31,30,31,31,30,31,30,31]

    if year <=0 or month > 12:
        return None
    elif es_bisiesto(year):
        dias[1]=29
    return dias[month-1]
        

print(dias_del_mes(1991,2))


test_years = [1900, 2000, 2016, 1987,1992] 
test_months = [2, 2, 1, 11,2] 
test_results = [28, 29, 31, 30,29]

for i in range(len(test_results)):
    if dias_del_mes(test_years[i],test_months[i]) == test_results[i]:
        print (True)
    else:
        print (False)
