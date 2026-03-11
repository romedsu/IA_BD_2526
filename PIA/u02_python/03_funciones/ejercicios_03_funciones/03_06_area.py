def es_triangulo(a,b,c):
    if (a +b > c) and (a+c > b) and (b+c > a):
        return True
    else:
        return False
    
def heron(a,b,c):
        s=(a+b+c)/2
        area= (s*(s-a)*(s-b)*(s-c))**0.5
        return area

def area_triangulo(a,b,c):
    if es_triangulo(a,b,c):
        print(f"El área del triángulo es {heron(a,b,c)}")
        return True
    else:
        print("No es un triángulo")
        return False

    
print(area_triangulo(3, 4, 5))

test_data = [[6, 7, 8], [5, 5, 6], [10, 8, 7], [9, 9, 5], [12, 13, 5], [7, 10, 12], [15, 14, 13], [8, 15, 17]]
results_data=[True, True, True, True, True, True, True, True]


for i in range(len(test_data)):
    if area_triangulo(test_data[i][0],test_data[i][1],test_data[i][2]) == results_data[i]:
        print(True)
    else:
        print(False)        
