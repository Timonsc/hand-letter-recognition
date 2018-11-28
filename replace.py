f1 = open('textfile.txt', 'r')
f2 = open('abc.txt', 'w')
for line in f1:
    f2.write(line.replace(' ', ''))
f1.close()
f2.close()
