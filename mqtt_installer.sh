echo "Installing Mosquitto MQTT"

sudo dnf -y install mosquitto
suso systemctl enable mosquitto.service

cd /home/$USER/flurry
mkdir mosquitto
cd mosquitto
echo 'camflow:camflow'>password.txt

mosquitto_passwd -U password.txt

echo 'allow_anonymous false' > mosquitto.conf
echo 'password_file /home/$USER/flurry/mosquitto/password.txt' >> mosquitto.conf
