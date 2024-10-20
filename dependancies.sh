conda activate flurryenvironment

sudo dnf -y install python-pip3

pip3 install selenium
pip3 install termcolor
pip3 install paho-mqtt
pip3 install dgl
pip3 install torch
pip3 install orjson

sudo dnf install graphviz graphviz-devel
sudo yum groupinstall 'Development Tools'
pip3 install pygraphviz

sudo dnf -y install hydra
sudo dnf -y install hping3
