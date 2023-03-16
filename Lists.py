# List is a datastructure
# defined in Square brackets [] - different type of datatypes can be stored"
# indexing will be from 0 onwards
# Any data structure will be Variable.function
#Empty List
List1=[]
print(List1)

#List having elements************************************************* 

List2=[1,2.2,"bc",True]
print(len(List2))
print(List2)
#Append - it will add at the back
List2.append(1)
print(List2)
#Extend - give list instead of elements in Extend which will be added at the end
List2.extend([2,4,6,7])
print(List2)
#Insert - inserts at particular location
List2.insert(4,"insert test")
print(List2)
#count- counts the element how many times it is repeating , it takes into consideration true as well
print(List2.count(1))
#Len - Total count
print(len(List2))
#Pop - same as Delete
List2.pop(1)
print(List2)
#Reverse - Reverses the List
List2.reverse()
print(List2)
#Sort - For numeric elements, it gives in ascending order 
List3=[10,4,5,6,7]
List3.sort()
print(List3)
#Find out the type
print(type(List2))
#For loop
for i in List3:
    print(i)

#******************************************************************
#List with in Lists - Say for eg. I need 4 here, as 4 is in list, it starts from 0 in that list
#******************************************************************
List4=[1,1,2.2,[4,6,7,98],7]
print(List4[3][0])
print(List4[1])


