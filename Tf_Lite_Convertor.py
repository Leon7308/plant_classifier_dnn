from itertools import permutations
import math 

# def get_count(d):
#     c=0
#     for i in d:
#         c+=1 
#     return c 

n=int(input())
l=list(map(int,input().split()))

cc=[]

# d1=permutations(l,n-1)
# d2=permutations(l,n)
# cc.append(get_count(d1))
# cc.append(get_count(d2))

s1=math.factorial(n)//math.factorial(n-(n))
s2=math.factorial(n)//math.factorial(n-(n-1))

cc.append(s1)
cc.append(s2)

if(n%2==0):
    t=sum(cc)+2
else:
    t=sum(cc)-1 

print("%.6f"%(t/cc[-1]))
