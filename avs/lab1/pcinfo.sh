#!/bin/bash
date
whoami
hostname
lscpu | grep "Имя модели:" | cut -d'@' -f1
lscpu | grep "Архитектура:"
lscpu | grep "Имя модели:" | cut -d'@' -f2