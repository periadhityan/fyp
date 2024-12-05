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
    mal1 = benign
    mal1.append(malicious[0])
    print(mal1)

    mal_perms = []

    for i in range(len(mal1)):
        perms = list(itertools.permutations(mal1, i+1))
        perms = [x for x in perms if malicious[0] in x]
        mal_perms.extend(perms)
    
    mal1.remove(malicious[0])
    mal1.append(malicious[1])

    for i in range(len(mal1)):
        perms = list(itertools.permutations(mal1, i+1))
        perms = [x for x in perms if malicious[1] in x]
        mal_perms.extend(perms)

    mal1.remove(malicious[1])
    mal1.append(malicious[2])

    for i in range(len(mal1)):
        perms = list(itertools.permutations(mal1, i+1))
        perms = [x for x in perms if malicious[2] in x]
        mal_perms.extend(perms)

    mal1.remove(malicious[2])
    mal1.append(malicious[3])

    for i in range(len(mal1)):
        perms = list(itertools.permutations(mal1, i+1))
        perms = [x for x in perms if malicious[3] in x]
        mal_perms.extend(perms)

    mal1.remove(malicious[3])
    mal1.append(malicious[4])

    for i in range(len(mal1)):
        perms = list(itertools.permutations(mal1, i+1))
        perms = [x for x in perms if malicious[4] in x]
        mal_perms.extend(perms)

    mal1.remove(malicious[4])
    mal1.append(malicious[5])

    for i in range(len(mal1)):
        perms = list(itertools.permutations(mal1, i+1))
        perms = [x for x in perms if malicious[5] in x]
        mal_perms.extend(perms)
    
    with open('malicious_scenarios.txt', 'w') as f:
        for line in mal_perms:
            for action in line:
                f.write(action)
                if(action != line[-1]):
                    f.write(",")
            f.write("\n")


malicious_permutations()


