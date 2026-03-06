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


test_data = [1900, 2000, 2016, 1987, 1992]   
test_results = [False, True, True, False, True]

print(es_bisiesto(2012));
print("----");


for i in range(len(test_data)):
    if es_bisiesto(test_data[i]) == test_results[i]:
        print (True)
    else:
        print (False)

