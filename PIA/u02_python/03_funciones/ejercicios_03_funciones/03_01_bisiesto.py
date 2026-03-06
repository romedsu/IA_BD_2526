def bisiesto(data):
    #global test_data, test_results

    test_results = [True, True, True, False, True, False,False,True]
    print(len(data))
    

    for i in range(len(data)):
        if data[i] %4 != 0:
            return not(test_results[i])
        elif data[i] %100 != 0:
            return test_results[i]
        elif data[i] %400 != 0:
            return not test_results[i]
        else:
            return test_results[i]

test_data = [1993, 2000, 2016, 1987, 1992, 1987, 2001, 2012]
test_results = [False, True, True, False, True, False,False,True]


print(bisiesto(test_data))


'''
 if data[i] %4 != 0:
            return False
        elif data[i] %100 != 0:
            return True
        elif data[i] %400 != 0:
            return False
        else:
            return True

'''

'''

    for i in range(len(data)):
        if data[i] %4 == results[i]:
            
            print (False)
        else:
            print (True)
'''
