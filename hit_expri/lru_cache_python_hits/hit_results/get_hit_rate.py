import os

hit_lst = []

file_name =input()
with open(os.getcwd()+'/'+file_name) as f:
    for line in f:
        if line.startswith('CacheInfo'):
            hit_lst.append(line.split('=')[1].split(',')[0])

for i in range(len(hit_lst)):
    if i>0:
        print('{:.2f}'.format((int(hit_lst[i])-int(hit_lst[i-1]))/25000*100),end=' ')
print()