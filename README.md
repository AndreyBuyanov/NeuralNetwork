NeuralNetwork
===========
![License](https://img.shields.io/badge/Code%20License-MIT-blue.svg)
---------------
Лабораторная работа №3 по дисциплине "Интеллектуальные системы".

Сборка и запуск
---------------
cmake -S . -B build && cmake --build build --config RelWithDebInfo

Запуск в докере
---------------
docker build -t intelligent-systems/neural-network .
docker run intelligent-systems/neural-network
