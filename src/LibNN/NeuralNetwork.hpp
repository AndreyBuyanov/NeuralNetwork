#pragma once

#include "Matrix.hpp"
#include "ActivationFunctions.hpp"

namespace NN
{

class NeuralNetworkTrainer;

/**
 * Структура, описывающая слой нейронной сети
 */
struct LayerConfig
{
    // Количество нейронов
    std::size_t neurons;
    // Тип функции активации
    ActivationFunction fn;
};

/**
 * Класс, реализующий нейронную сеть.
 */
// TODO: Добавить нейрон смещения к слоям
// TODO: Добавить возможность сохранения\загрузки весов из файла
// TODO: Реализовать другие типы нейронных сетей: свёрточные, рекурентные и др
class NeuralNetwork
{
public:
    /**
     * Конструктор.
     *
     * \param inputs Количество входов нейронной сети
     * \param layers Массив с конфигурацией слоёв
     */
    NeuralNetwork(const std::size_t inputs, const std::vector<LayerConfig>& layers):
        m_weights(layers.size()),   // Количество матриц весов соответстует количеству слоёв
        m_layers(layers)            // Сохраняем конфигурацию
    {
        // Веса первого слоя - это матрица, имеющая количество строк,
        // равное количеству нейронов первого слоя,
        // и количество столбцов, равное количеству входов
        m_weights[0] = Matrix(layers[0].neurons, inputs);
        // Проходим по остальным слоям
        for (std::size_t i = 1; i < layers.size(); i++) {
            // Веса текущего слоя - это матрица, имеющая количество строк,
            // равное количеству нейронов текущего слоя,
            // и количество столбцов, равное количеству нейронов предыдущего слоя
            m_weights[i] = Matrix(layers[i].neurons, layers[i - 1].neurons);
        }
    }
    /**
     * Получение количества слоёв нейронной сети.
     *
     * \return Количество слоёв нейронной сети
     */
    std::size_t LayersCount() const
    {
        return m_weights.size();
    }
    /**
     * Прямой проход по нейронной сети
     *
     * \param input Вектор входных данных
     * \return Вектор выходных данных
     */
    Vector Forward(const Vector& input) const
    {
        // TODO: Реализовать одним циклом
        // Проходим по первому слою, подав на него входной вектор
        // В output получим выход первого слоя,
        // и это будет входом следующего слоя
        auto output = Forward(input, 0);
        // Проходим по остальным слоям
        for (std::size_t i = 1; i < m_weights.size(); i++) {
            // Сейчас в output хранится выходной вектор предыдущего слоя
            // Подадим output на вход текущего слоя
            output = Forward(output, i);
            // Теперь в output выход текущего слоя
        }
        // Прошли по всем слоям, значит в output выход последнего слоя.
        // Возвращаем результат
        return output;
    }
private:
    // Массив с матрицами весов каждого слоя
    std::vector<Matrix> m_weights;
    // Массив с конфигурациями слоёв
    std::vector<LayerConfig> m_layers;

    /**
     * Прямой проход по слою нейронной сети
     * 
     * \param input Вектор входных данных
     * \param layer Номер слоя
     * \return Вектор выходных данных
     */
    Vector Forward(const Vector& input, const std::size_t layer) const
    {
        // Умножаем матрицу весов на вектор входных данных
        // К получившемуся вектору применим функцию активации
        // Вернём получившийся вектор
        return (m_weights[layer] * input).ApplyFunction(GetFunction(m_layers[layer].fn));
    }

    friend class NeuralNetworkTrainer;
};

}
