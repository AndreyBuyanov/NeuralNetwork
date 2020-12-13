#include <iostream>
#include <random>

#include "NeuralNetwork.hpp"
#include "NeuralNetworkTrainer.hpp"

#if defined(WIN32)
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <Windows.h>
#endif

/**
 * Решение задачи XOR.
 */

// Входные данные
std::vector<NN::Vector> X = {
    { 0.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 1.0, 1.0 }
};

// Выходные данные
std::vector<NN::Vector> Y = {
    { 0.0 },
    { 1.0 },
    { 1.0 },
    { 0.0 }
};

// Максимальное количество эпох
const std::size_t epochs = 1000000;
// Скорость обучения
const double learningRate = 0.5;
//
const double momentum = 0.5;
// Минимальная ошибка
const double epsilon = 1e-5;

// TODO: Добавить возможность задавать параметры сети из командной строки
int main (int argc, char *argv[]){
    // Костыль для винды
#if defined(WIN32)
    SetConsoleOutputCP(65001);
#endif
    // Нейронная сеть
    NN::NeuralNetwork nn(2, {                   // 2 входа
        {2, NN::ActivationFunction::Sigmoid, 1.0 },  // Скрытый слой: 2 нейрона, функция активации - сигмоида
        {1, NN::ActivationFunction::Sigmoid, 1.0 }   // Выходной слой: 1 нейрон, функция активации - сигмоида
    });
    // Обучатель нейронной сети
    NN::NeuralNetworkTrainer nnTrainer(nn, learningRate, momentum);
    // Генератор случайных чисел
    std::random_device rd;
    // Движок генерации случайных чисел
    std::mt19937 rng(rd());
    rng.seed(1);
    // Инициализируем веса нейронной сети
    nnTrainer.Init(-0.5, 0.5, rng);
    // Начальная эпоха
    std::size_t epoch = 1;
    // Текущая ошибка сети
    double error = 0.0;
    // Нормальное распределение от 0 до 3 (количество входных\выходных данных)
    std::uniform_int_distribution<std::size_t> ds(0, X.size() - 1);
    // Запускаем цикл обучения
    do {
        // Генерируем индекс
        auto index = ds(rng);
        // Выбираем случайную пару входных и выходных данных и отправляем в обучатель
        error = nnTrainer.Train(X[index], Y[index]);
        if (epoch % 1000 == 0){
            // Выводим ошибку. Чтобы не забивать консоль сообщениями
            // выводим только когда эпоха кратна 1000
            std::cout << "Epoch: " << epoch << ", Error: " << error << std::endl;
        }
        epoch++;
    // Выполняем обучение до тех пор, пока не будет достигнуто максимальное количество эпох,
    // либо пока не получим достаточную точность нейронной сети
    } while (epoch <= epochs && error > epsilon);
    // Выводим ошибку
    std::cout << "Epoch: " << epoch << ", Error: " << error << std::endl;
    // Проверяем обученную нейронную сеть,
    // последовательно подавая в сеть пары входных данных
    // и выводя результат
    for (int i = 0; i < X.size(); i++) {
        // Делаем прямой проход по сети
        NN::Vector output = nn.Forward(X[i]);
        // Выводим результат
        std::cout << "X: " << X[i][0] << " " << X[i][1] << ", Output: " << output[0] << std::endl;
    }
    return 0;
}
