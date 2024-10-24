echo "Installing Mosquitto MQTT"

sudo dnf -y install mosquitto
sudo systemctl enable mosquitto.service

cd /home/$USER/flurry
mkdir mosquitto
cd mosquitto
echo 'camflow:camflow'>password.txt

mosquitto_passwd -U password.txt

echo 'allow_anonymous false' > mosquitto.conf
echo -n 'password_file /home/' >> mosquitto.conf
echo -n $USER >> mosquitto.conf
echo -n '/flurry/mosquitto/password.txt' >> mosquitto.conf
