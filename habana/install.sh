wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.18.0/habanalabs-installer.sh
chmod +x habanalabs-installer.sh
./habanalabs-installer.sh install --type base
sudo apt install -y habanalabs-container-runtime
