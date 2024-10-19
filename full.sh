if [ $USER == "root" ]
then
 echo "Don't run this script as root. Run it as a normal user, and it'll use
sudo internally as needed"
 echo "This is done to stop the permissions from getting messed up on
downloaded files"
 exit
fi
echo "Adding NOPASSWD property to sudoers for ${USER}"
sudo echo "${USER} ALL = NOPASSWD: ALL" | sudo EDITOR='tee -a' visudo
# Camflow already installed in previous steps
# Add Camflow installation steps if necessary
# Set Camflow kernel as default
# Allow whole system provenance capture
sudo camflow -a true
# Update Camflow Configurations
sudo sh -c "echo '[provenance]
;unique identifier for the machine, use hostid if set to 0
machine_id=0
;enable provenance capture
enabled=true
;record provenance of all kernel object
;all=true
node_filter=directory
node_filter=inode_unknown
node_filter=char
node_filter=envp
; propagate_node_filter=directory
; relation_filter=sh_read
; relation_filter=sh_write
; propagate_relation_filter=write
[compression]
; enable node compression
node=true
edge=true
duplicate=true
[file]
;set opaque file
opaque=/usr/bin/bash
;set tracked file
;track=/home/thomas/test.o
;propagate=/home/thomas/test.o
[ipv4−egress]
;propagate=0.0.0.0/0:80
;propagate=0.0.0.0/0:404
;record exchanged with local server
;record=127.0.0.1/32:80
[ipv4−ingress]
;propagate=0.0.0.0/0:80
;propagate=0.0.0.0/0:404
;record exchanged with local server
;record=127.0.0.1/32:80
[user]
;track=vagrant
;propagate=vagrant
;opaque=vagrant
[group]
;track=vagrant
;propagate=vagrant
;opaque=vagrant
[secctx]
;track=system_u:object_r:bin_t:s0
;propagate=system_u:object_r:bin_t:s0
;opaque=system_u:object_r:bin_t:s0' > /etc/camflow.ini"
sudo sh -c "echo '[general]
;output=null
output=mqtt
;output=unix_socket
;output=fifo
;output=log
format=w3c
;format=spade_json
[log]
;path=/tmp/audit.log
[mqtt]
address=localhost:1883
username=camflow
password=camflow
; message delivered: 0 at most once, 1 at least once, 2 exactly once
qos=0
; topic, provided prefix + machine_id (e.g. camflow/provenance/1234)
topic=camflow/provenance/
[unix]
address=/tmp/camflowd.sock
[fifo]
path=/tmp/camflowd-pipe' > /etc/camflowd.ini"
# Restart to render the changes made in the configuration file
sudo systemctl restart CamFlowd.service
sudo dnf -y update
echo "Installing libnsl..."
sudo dnf -y install libnsl
echo "Installing wget..."
sudo dnf -y install wget
# XAMPP needs to be downloaded here, but it's not in a repository
wget -O xampp-installer
"https://sourceforge.net/projects/xampp/files/XAMPP%20Linux/8.2.0/xampp-linuxx64-8.2.0-0-installer.run/download"
chmod +x xampp-installer
sudo ./xampp-installer
#read continue
echo "adding xampp to path..."
sudo sed -i "s/Defaults secure_path =
\/usr\/local\/sbin:\/usr\/local\/bin:\/usr\/sbin:\/usr\/bin:\/sbin:\/bin:\/var\
/lib\/snapd\/snap\/bin/Defaults secure_path =
\/usr\/local\/sbin:\/usr\/local\/bin:\/usr\/sbin:\/usr\/bin:\/sbin:\/bin:\/var\
/lib\/snapd\/snap\/bin:\/opt\/lampp/" /etc/sudoers
echo "export PATH=\${PATH}:/opt/lampp" >> ~/.bashrc
echo "setting up cron job..."
sudo dnf -y install cronie
echo "@reboot sudo /opt/lampp/xampp start" | EDITOR='tee -a' crontab -e
# DVWA needs XAMPP to run
echo "Setting up the DVWA..."
cd /opt/lampp/htdocs
sudo git clone https://github.com/digininja/DVWA.git
sudo cp /opt/lampp/htdocs/DVWA/config/config.inc.php.dist
/opt/lampp/htdocs/DVWA/config/config.inc.php
sudo cp -r /opt/lampp/htdocs/DVWA/* /opt/lampp/htdocs
sudo sh -c "sed \"s/\[ 'db_user' \] = 'dvwa'/\[ 'db_user' \] =
'root'/;s/\[ 'db_password' \] = 'p@ssw0rd'/\[ 'db_password' \] = ''/\"
/opt/lampp/htdocs/config/config.inc.php.dist >
/opt/lampp/htdocs/config/config.inc.php"
# Create flurry directory
cd ~
mkdir flurry
# Install Flurry
echo "Cloning Flurry Repo..."
cd ~/flurry
git clone https://github.com/mayakapoor/flurry.git
# Hard reset to latest working commit, remove when main is stable
cd ~/flurry/flurry
# Latest working commit 42a548d441447defd4b6b3ef8841f6c5c353ed42
git reset --hard 42a548d441447defd4b6b3ef8841f6c5c353ed42
# Install flake
cd ~/flurry
git clone https://github.com/mayakapoor/flake.git
sudo sh -c "echo '[DEFAULT]
OUTPUT_DIR = /home/perlyn/flurry/output
INPUT_DIR =
[DATABASE]
DB_FILE = /home/perlyn/flurry/data/flake.db
SAVE_TO_DISK = yes
[FILTER] EDGE_GRANULARITY = coarse NODE_GRANULARITY = fine
[MQTT]
MQTT_USERNAME = camflow
MQTT_PASSWORD = camflow
MQTT_HOST = localhost
MQTT_PORT = 1883
MQTT_TOPIC = camflow/provenance/#' > ~/flurry/flake/src/flurryflake/flake.ini"
# add /opt/lampp to secure path in sudoers
# Install chrome and chrome driver
sudo dnf -y config-manager --set-enabled google-chrome
sudo dnf -y install google-chrome
sudo dnf -y install google-chrome-stable
GC_VERSION=$(google-chrome --version)
GC_VERSION_NUM=$(echo "$GC_VERSION" | cut -b 15- | cut -d "." -f 1)
GCD_VERSION=$(wget -qOhttps://chromedriver.storage.googleapis.com/LATEST_RELEASE_${GC_VERSION_NUM})
wget -O chromedriver_linux64.zip
"https://chromedriver.storage.googleapis.com/${GCD_VERSION}/chromedriver_linux6
4.zip"
unzip chromedriver_linux64.zip
sudo mv chromedriver /bin
# Make xampp point to the DVWA
sudo sed -i "s/dashboard/login.php/" /opt/lampp/htdocs/index.php
# Install conda and copy environment
echo "Installing Conda..."
sudo dnf -y install conda
conda init bash
touch condasetup.sh
echo "cd ~/flurry
conda env create -f environment.yml
sed -i \"s/from collections import Mapping/from collections.abc import
Mapping/\" ~/.conda/envs/flurryenv/lib/python3.10/sitepackages/dgl/dataloading/base.py
echo \"finished, press enter to exit...\"
read continue" > condasetup.sh
chmod +x condasetup.sh
gnome-terminal -- ./condasetup.sh
# Setting up mosquitto
echo "installing mosquitto..."
sudo dnf -y install mosquitto
sudo systemctl enable mosquitto.service
cd ~/flurry
mkdir mosquitto
cd mosquitto
echo 'camflow:camflow' > password.txt
mosquitto_passwd -U password.txt
echo 'allow_anonymous false' > mosquitto.conf
echo 'password_file /home/perlyn/flurry/mosquitto/password.txt' >>
mosquitto.conf
echo "installing hydra..."
sudo dnf -y install hydra
echo "installing hping3..."
sudo dnf -y install hping3
echo "installing sysdig..."
sudo dnf -y install sysdig
# Installing python dependencies
echo "installing python dependencies..."
sudo dnf -y install python3-pip
pip3 install termcolor paho-mqtt dgl torch orjson selenium sqlite3
sudo dnf -y install graphviz graphviz-devel
sudo yum -y groupinstall 'Development Tools'
# Fixes python.h error in pygraphviz
sudo yum -y install python3-devel
pip3 install pygraphviz
echo "INSTALLATION COMPLETE"
echo "IMPORTANT: The system will now reboot. Wait until all other terminal
windows have finished before continuing, then choose the Camflow kernel in
GRUB"
echo "Ready? [y/N] "
read answer
if [ "${answer^^}" = "Y" ]
then
 sudo reboot now
else
 echo "Reboot as soon as possible"
fi