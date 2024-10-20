echo "Cloning DVWA repo into lampp/htdocs"
cd /opt/lampp/htdocs
sudo git clone https://github.com/digininja/DVWA

echo "Setting up configuration files"

sudo cp /opt/lampp/htdocs/DVWA/config/config.inc.php.dist /opt/lampp/htdocs/DVWA/config/config.inc.php
sudo cp -r /opt/lampp/htdowc/DVWA/* /opt/lampp/htdocs

sudo sh -c "sed \"s/\[ 'db_user' \] = 'dvwa'/\[ 'db_user' \] =
'root'/;s/\[ 'db_password' \] = 'p@ssw0rd'/\[ 'db_password' \] = ''/\"
/opt/lampp/htdocs/config/config.inc.php.dist >
/opt/lampp/htdocs/config/config.inc.php"

sudo sed -i "s/dashboard/login/php" /opt/lampp/htdocs/index.php

