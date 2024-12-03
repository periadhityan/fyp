#Function to automate the running of benign and attack scenarios
"""
Attack Scenarios
xssstored
xssreflected
commandinjection
sqlinjection
bruteforce
customattack

Benign Scenarios
message
submit
query
ping
databaseentry
login
custombehaviour
"""

import subprocess

def main():
    
    subprocess.run(['cd /home/$USER/flurry'])
    subprocess.run(['conda activate env_flurry'])
    subprocess.run(['python webserver.py'])
    f = open("scenarios.txt", "r")

    for x in f:
        subprocess.run([{}].format(x))
    



if __name__ == "__main__":
    main()

