#Tuple is datastructure - Elements will not change
#Defined with ()
#Empty Tuple
T=()
print(type(T))

#************************************************************************
#Difference bween Tuple and Lists 
# We have more functionalities in Lists compared to Tuple
# Elements will change in Lists, will not change in Tuple
#Elements************************************************************************
T1=(1,2,"abc",True)
print(T1)

#Retrieve the data using indexing
print(T1[1])

#Repetition - how many times you want to see the data
print(T1*3)

#concatenation for two tuples
a=(1,2,3,5)
b=(3,4,5,6)
print(a+b)

#Membership - used with in - used to check the condition - is 2 in a? if yes, true,otherwise no
print(4 in a)

#iterations using for loop
for i in a:
    print(i)

#Functions Sum,Min,Max
print(min(a))
print(max(a))
print(sum(a))
