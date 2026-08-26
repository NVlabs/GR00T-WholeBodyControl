#!/bin/bash

# Script to restore official Ubuntu repositories
# Removes Tsinghua University mirrors and restores official sources

echo "Creating backup of current sources.list..."
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup.$(date +%Y%m%d_%H%M%S)

echo "Restoring official Ubuntu repositories for Ubuntu 22.04 (Jammy) ARM64..."

# Create new sources.list with official repositories
sudo tee /etc/apt/sources.list > /dev/null << 'SOURCES'
# Official Ubuntu repositories for ARM64 (ports.ubuntu.com)
deb http://ports.ubuntu.com/ubuntu-ports/ jammy main restricted
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main restricted
deb http://ports.ubuntu.com/ubuntu-ports/ jammy universe
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-updates universe
deb http://ports.ubuntu.com/ubuntu-ports/ jammy multiverse  
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-updates multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-backports main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security main restricted
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security universe
deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security multiverse

# Source repositories (uncomment if you need source packages)
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy universe
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-updates universe
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-updates multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-backports main restricted universe multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-security main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-security universe
# deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy-security multiverse
SOURCES

echo "Updated sources.list with official Ubuntu repositories"
echo "Updating package lists..."
sudo apt update

echo "Done! Ubuntu repositories restored to official sources."
echo "Backup saved as: /etc/apt/sources.list.backup.$(date +%Y%m%d_%H%M%S)"
