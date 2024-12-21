#!/bin/bash

# Updating system packages
sudo yum update -y

sudo yum install docker -y

# starting Docker service
sudo systemctl start docker