f1 = open('class.txt')
txt = f1.read()
#print(txt)
f1.close()
ls = txt.split(',')
f2 = open('classes.txt','w')
for i in ls:
    f2.write(i.split(':')[-1]+'\n')
f2.close()