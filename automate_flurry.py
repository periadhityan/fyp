import subprocess

def flurry_webserver(scenario, sudo):
    p = subprocess.Popen(['python', 'webserver.py'])
    output, error = p.communicate(input=scenario)