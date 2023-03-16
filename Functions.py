#Using print
def functionname1(name):
    print("Function Test",name) 
functionname1('Kalpana')

#1. Using return - Use only return for functions as the packages uses return only
def functionname2(a,b):
    return a+b 
w=functionname2(3,1)
print(w)  

#2. Using return
def functionname3(a):
    return a*3
print(functionname3(3))

#3. Using pass - it will not return anything
def functionname4(a):
    pass

#4. Using list and for loop
def functionname4(a):
    for i in a:
        print(i)
functionname4([1,2,3,4,5])
#5. Keyword arguments, orbitary arguments (multiple values passing, it will be stored in tuple)
#*a is orbitary argument
def functionname5(*a):
    print(a)
functionname5(1,2,3,4)
#**a is keyword argument - pass key and value - This will be stored in Dictionary format
def functionname6(**a):
    print(a)
functionname6(height=5.6, weight=60)




#orbitary parameters and Keyword parameters
#Orbitary argument *, can give n number of parameters stored in Tuple
#def orbitary(s):
#return s
#orbitary(1)




