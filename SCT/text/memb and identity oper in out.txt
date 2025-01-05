list1=[0,2,4,6,8]
list2=[1,3,5,7,9]
#Initially we assign 0 to not overlapping
check=0
for item in list1:
    if item in list2:
#Overlapping true so check is assigned 1
        check=1
if check==1:
    print("overlapping")
else:
    print("not overlapping")
