echo "Installing dependancied"
sudo dnf -y update
sudo dnf -y install libnsl
echo "Installing XAMPP"

chmod +x xampp-installer.run

echo "Complete XAMPP installation through GUI"
sudo ./xampp-installer.run
