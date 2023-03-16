#for i in range()
# Can pass anything like lists, dictionary,sets,tuple,variable etc
for i in 'Kalpana':
    print(i)
    
#1. using lists
l1=[1,2,3,4,5]
for i in l1:
    print(i)

#2. lists
sum=0
for i in l1:
    sum=sum+i
    print(sum)

# Range - start, stop, stepsize
for i in range(0,20,5):
    print(i)

# for with if using strings
mobiles=['iphone','oneplus','realme']
for i in mobiles:
    if i=='iphone':
        print('This is iphone')
    else:
        print('This is not iphone')
    
# while - Make false condition
c=0
while c<3:
    c+=1
    print("c is",c)
else:
    print('completed')
  