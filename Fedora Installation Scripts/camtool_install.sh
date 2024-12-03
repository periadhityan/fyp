#!/bin/bash

set -e

echo "Installing ruby....."
sudo dnf -y install ruby >> camtool_install_results.log
echo "Installing camtool...."
gem install camtool >> camtool_install_results.log

echo "Installing nano...."
sudo dnf install nano >> camtool_install_results.log

sudo sed -i 's|;format=w3c|format=w3c|g' /etc/camflowd.ini
sudo sed -i 's|format=spade_json|;format=spade_json|g' /etc/camflowd.ini

sudo systemctl restart camflowd.service
echo "Success!"



