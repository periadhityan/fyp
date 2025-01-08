f = open("xsstored.txt", "r")
lines = [line.strip() for line in f.readlines()]

set1 = lines[0:100]
set2 = lines[100:200]
set3 = lines[200:300]
set4 = lines[300:400]
set5 = lines[1000:1100]
set6 = lines[1100:1200]
set7 = lines[1200:1300]
set8 = lines[1300:1400]
set9 = lines[2000:2100]
set10 = lines[2100:2200]
set11 = lines[2200:2300]
set12 = lines[2300:2400]
set13 = lines[3200:3300]
set14 = lines[3300:3400]
set15 = lines[3400:3500]
set16 = lines[3500:3600]
set17 = lines[4200:4300]
set18 = lines[4300:4400]
set19 = lines[4400:4500]
set20 = lines[4500:4600]

sets = [set1,
set2,
set3,
set4,
set5,
set6,
set7,
set8,
set9, set10,
set11,
set12,
set13,
set14,
set15,
set16,
set17,
set18,
set19,
set20 ]

print(len(sets[0]))

for i in range(len(sets)):
    with(open(f"mal{i+1}.txt", 'a')) as output:
        for line in sets[i]:
            output.write(f"{line}\n")