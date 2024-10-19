#/bin/bash

echo "Downloading camflow rpm files now....."
curl -1sLf 'https://dl.cloudsmith.io/public/camflow/camflow/cfg/setup/bash.rpm.sh' | sudo -E bash

if [ $? -eq 0 ]
then
	echo "Package Downloaded Successfully"
	echo "Installing package now......"
	sudo dnf -y install camflow >> package_install_results.log
	if [ $? -eq 0 ]
	then
		echo "Package installed successfully"
		echo "Activating camconfd.service and camflowd.service"
		sudo systemctl enable camconfd.serivce
		sudo systemctl enable camflowd.service
		
		echo "Rebooting System now"
		sudo reboot now
	else
		echo "Failed to install package" >> package_install_results.log
	fi 
else
	echo "Package Download Failed. Please visit camflow.org to check if there are any updates to the package file locations"
fi

echo "Success!"
