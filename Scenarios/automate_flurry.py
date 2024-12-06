import subprocess

def flurry_webserver(input_scenario):
    p = subprocess.Popen(['python', 'webserver.py'], stdin=subprocess.PIPE,
                                                    stdout=subprocess.PIPE,
                                                    encoding='utf8')
    p.communicate(input=input_scenario)


def main():
    suffix = "\n1\n1\nf\nc"
    with open("mini_sample.txt", 'r') as f:
        for line in f:
            input = line+suffix
            print(input)

if __name__== "__main__":
    main()