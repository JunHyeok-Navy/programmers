a, b = map(int, input().strip().split(' '))
column = []
d = ''
for i in range(0, a):
    column.append('*')
    d += column[i]
for i in range(0, b):
    print(d)
