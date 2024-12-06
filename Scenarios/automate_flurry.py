import subprocess

def flurry_webserver(input_scenario):
    p = subprocess.Popen(['python', 'webserver.py'], stdin=subprocess.PIPE,
                                                    stdout=subprocess.PIPE,
                                                    encoding='útf8')
    p.communicate(input=input_scenario)


def main():
    suffix = "\n1\n1\nf\nc"
    with open("benign_scenarios.txt", 'r') as f:
        for line in f:
            input = line+suffix
            
if __name__== "__main__":
    main()