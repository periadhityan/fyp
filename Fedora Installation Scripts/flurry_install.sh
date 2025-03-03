#!/bin/bash

echo "Navigating to home folder"
cd /home/$USER

echo "Cloning flurry repository for intallation"
git clone https://github.com/mayakapoor/flurry

echo "Resetting to previous working commit"


echo "Navigating into flurry directory"
cd flurry
git init
git reset --hard 42a548d441447defd4b6b3ef8841f6c5c353ed42

echo "Cloning flake repository"
git clone https://github.com/mayakapoor/flake

echo "Configuring flake.ini"
cd /home/$USER/flurry/flake/src/flurryflake
sh -c "echo '
[DEFAULT]
OUTPUT_DIR=/home/$USER/flurry/output
INPUT_DIR=

[DATABASE] 
DB_FILE=/home/$USER/flurry/data/flake.db
SAVE_TO_DISK=yes

[MQTT]
MQTT_USERNAME=camflow
MQTT_PASSWORD=camflow
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC=camflow/provenance#
CAMFLOW_TOPIC=provenance/camflow#' > flake.ini"

echo "Configuring config.py"
cd /home/$USER/flurry/flake/src/flurryflake
rm config.py

sh -c "echo '
import configparser
import os
import sys

configp = configparser.ConfigParser()
config_path = 'home/$USER/flurry/flake/src/flurryflake/flake.ini'
configp.read(config_path)

# check if the path is to the valid file
if not os.path.isfile(config_path):
 print(config_path)
 sys.exit()
 
def initFromConfig(param):
 for section in configp.sections():
 if configp.has_option(section, param):
  return configp[section][param]
 sys.exit()' > config.py"
 
cd /home/$USER/flurry/flake/src/flurryflake/filters
sh -c "echo '
import configparser
import os
import sys

configp = configparser.ConfigParser()
config_path = 'home/$USER/flurry/flake/src/flurryflake/flake.ini'
configp.read(config_path)

# check if the path is to the valid file
if not os.path.isfile(config_path):
 print(config_path)
 sys.exit()
 
def initFromConfig(param):
 for section in configp.sections():
 if configp.has_option(section, param):
  return configp[section][param]
 sys.exit()' > config.py"

echo  "Configuring camflowd.ini"
cd /etc
sudo sh -c "echo '
[general]
;output=null
output=mqtt
;output-unix_socket
;output=fifo
;output=log

format=w3c
;format=spade_json

[log]
path-/tmp/audit.log

[mqtt]
address=localhost:1883
username=camflow
password=camflow

; message delivered: 0 at most once, at least once, 2 exactly once
qos=0
;topic, provided prefix + machine_id (e.g. camflow/provenance/1234)
topic=camflow/provenance

[unix]
address=/tmp/camflowd.sock

[fifo]
path/tmp/camflowd-pipe' > camflowd.ini"

echo  "Configuring camflow.ini"
sudo sh -c "echo '
[provenance]
;unique identifier for the machine, use hostid if set to 0
machine_id=0
;enable provenance capture
enabled=true
;record provenance of all kernel object
;all=false
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
;opaque=system_u:object_r:bin_t:s0' > camflow.ini"
