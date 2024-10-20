sudo dnf -y install conda
conda --init bash

sudo dnf config-manager --set-enabled google-chrome
sudo dnf install google-chrome
sudo dnf -y install google-chrome-stable

cd chromdriver
sudo mv chromedriver /usr/bin/chromedriver

sudo chown root:root /usr/bin/chromedriver
sudo chmod 0755 /usr/bin/chromedriver
