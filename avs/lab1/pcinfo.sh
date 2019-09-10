#!/bin/bash
echo -ne "Дата:\t\t\t"
date

echo -ne "Имя учётной записи:\t"
whoami

echo -ne "Доменное имя пк:\t"
hostname

echo "Процессор:"

echo -ne "\t* "
lscpu | grep "Имя модели:" | cut -d'@' -f1
echo -ne "\t* "
lscpu | grep "Архитектура:"
echo -ne "\t* Тактовая частота:   "
lscpu | grep "Имя модели:" | cut -d'@' -f2
echo -ne "\t* Количество ядер:     "
lscpu | grep -m1 "CPU(s):" | rev | cut -d' ' -f1 | rev
echo -ne "\t* Количество потоков\n\t  на одно ядро:\t       "
lscpu | grep -m1 "Thread(s) per core:" | rev | cut -d' ' -f1 | rev

echo "Оперативная память:"
echo -ne "\t* Всего:\t       "
free -h | grep "Mem:" | tr -s ' ' | cut -d' ' -f2
echo -ne "\t* Доступно:\t       "
free -h | grep "Mem:" | tr -s ' ' | cut -d' ' -f7

echo "Жёсткий диск:"
echo -ne "\t* Всего:\t       "
df -h | grep "/dev/sda1" | tr -s ' ' | cut -d' ' -f2
echo -ne "\t* Доступно:\t       "
df -h | grep "/dev/sda1" | tr -s ' ' | cut -d' ' -f4
echo -ne "\t* Смонтировано в\n\t  корневую директорию: "
df -h | grep "/dev/sda1" | tr -s ' ' | cut -d' ' -f1
echo -ne "\t* SWAP всего:\t       "
free -h | grep "Swap:" | tr -s ' ' | cut -d' ' -f2
echo -ne "\t* SWAP доступно:       "
free -h | grep "Swap:" | tr -s ' ' | cut -d' ' -f4

echo "Сетевые интерфейсы:"
echo -ne "\t* Количество сетевых\n\t  интерфейсов:\t       "
ls /sys/class/net/ | wc -l
for var in /sys/class/net/*
do

done
