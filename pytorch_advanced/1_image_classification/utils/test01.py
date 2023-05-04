# python 列表 可变测试

a = 5 
b = ['q','w','e']
c = 7
d = [1,2,3]
print(id(b),type(b),b)
print(id(d),type(d),d)

ll = [a,b,c,d]
print(id(ll),type(ll),ll)

d.append(4)
print(id(d),type(d),d)

print(id(ll),type(ll),ll)


b[1] = 'xyz'
print(id(b),type(b),b)
print(id(ll),type(ll),ll)

a = 9
print(id(ll),type(ll),ll)

d[1] = 99
print(id(ll),type(ll),ll)