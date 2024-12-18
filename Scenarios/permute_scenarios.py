#Script to build a text file with every possible permutation of benign and malicious scenarios for DVWA
"""
Malicious:

xssstored
xssreflected
xssdom
commandinjection
sqlinjection
bruteforce

Benign:

message
submit
query
ping
databaseentry
login
"""

import itertools

malicious = ["xssstored", "xssreflected", "xssdom", "commandinjection", "sqlinjection", "bruteforce"]
benign = ["message", "submit", "query", "ping", "databaseentry", "login"]

def benign_permutations():
    all_perms = []
    for i in range(len(benign)):
        perms = list(itertools.permutations(benign, i+1))
        all_perms.extend(perms)

    with open('benign_scenarios.txt', 'w') as f:
        for line in all_perms:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")
    return all_perms

def malicious_permutations():
    mal = benign
    mal.append(malicious[0])


    #for i in range(len(mal)):
    perms = list(itertools.permutations(mal, 7))
    perms = [x for x in perms if malicious[0] in x]

    with open('mal1.txt', 'w') as f:
        for line in perms[:400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal2.txt', 'w') as f:
        for line in perms[1000:1400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal3.txt', 'w') as f:
        for line in perms[3200:3600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal4.txt', 'w') as f:
        for line in perms[4200:4600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('malextra1.txt', 'w') as f:
        for line in perms[2000:2400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")
    
    mal.remove(malicious[0])
    mal.append(malicious[1])

    #for i in range(len(mal)):
    perms = list(itertools.permutations(mal, 7))
    perms = [x for x in perms if malicious[1] in x]

    with open('mal5.txt', 'w') as f:
        for line in perms[:400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal6.txt', 'w') as f:
        for line in perms[1000:1400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal7.txt', 'w') as f:
        for line in perms[3200:3600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal8.txt', 'w') as f:
        for line in perms[4200:4600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('malextra2.txt', 'w') as f:
        for line in perms[2000:2400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    mal.remove(malicious[1])
    mal.append(malicious[2])

    #for i in range(len(mal)):
    perms = list(itertools.permutations(mal, 7))
    perms = [x for x in perms if malicious[2] in x]

    with open('mal9.txt', 'w') as f:
        for line in perms[:400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal10.txt', 'w') as f:
        for line in perms[1000:1400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal11.txt', 'w') as f:
        for line in perms[3200:3600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal12.txt', 'w') as f:
        for line in perms[4200:4600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('malextra3.txt', 'w') as f:
        for line in perms[2000:2400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    mal.remove(malicious[2])
    mal.append(malicious[3])

    #for i in range(len(mal)):
    perms = list(itertools.permutations(mal, 7))
    perms = [x for x in perms if malicious[3] in x]

    with open('mal13.txt', 'w') as f:
        for line in perms[:400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal14.txt', 'w') as f:
        for line in perms[1000:1400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal15.txt', 'w') as f:
        for line in perms[3200:3600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal16.txt', 'w') as f:
        for line in perms[4200:4600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('malextra4.txt', 'w') as f:
        for line in perms[2000:2400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")


    mal.remove(malicious[3])
    mal.append(malicious[4])

    #for i in range(len(mal)):
    perms = list(itertools.permutations(mal, 7))
    perms = [x for x in perms if malicious[4] in x]

    with open('mal17.txt', 'w') as f:
        for line in perms[:400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal18.txt', 'w') as f:
        for line in perms[1000:1400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal19.txt', 'w') as f:
        for line in perms[3200:3600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal20.txt', 'w') as f:
        for line in perms[4200:4600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('malextra5.txt', 'w') as f:
        for line in perms[2000:2400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    mal.remove(malicious[4])
    mal.append(malicious[5])

    #for i in range(len(mal)):
    perms = list(itertools.permutations(mal, 7))
    perms = [x for x in perms if malicious[5] in x]

    with open('mal21.txt', 'w') as f:
        for line in perms[:400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal22.txt', 'w') as f:
        for line in perms[1000:1400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal23.txt', 'w') as f:
        for line in perms[3200:3600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('mal24.txt', 'w') as f:
        for line in perms[4200:4600]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

    with open('malextra6.txt', 'w') as f:
        for line in perms[2000:2400]:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")

def main():
    #benign_permutations()
    malicious_permutations()

if __name__ == "__main__":
    main()
