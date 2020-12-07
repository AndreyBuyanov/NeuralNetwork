#pragma once

#include <random>

#include "NeuralNetwork.hpp"

namespace NN
{

/**
 * Класс, реализующий "обучатель" нейронной сети.
 */
// TODO: Реализовать разные оптимизаторы: Momentum, NAG, Adam и др.
// TODO: Реализовать обучение с помощью эволюционных алгоритмов
class NeuralNetworkTrainer
{
public:
    /**
     * Конструктор.
     *
     * \param nn Нейронная сеть для обучения
     * \param learningRate Скорость обучения
     */
    NeuralNetworkTrainer(NeuralNetwork& nn, const double learningRate):
        m_nn(nn),                       // Сохраняем ссылку на нейронную сеть
        m_learningRate(learningRate)    // Сохраняем скорость обучения
    {
        // Количество элементов в массиве выходных значений
        // должно соответствовать количеству слоёв
        m_outputs.resize(m_nn.LayersCount());
        // Количество элементов в массиве градиентов
        // должно соответствовать количеству слоёв
        m_gradients.resize(m_nn.LayersCount());
        // Инициализируем нейронную сеть случайными значениями
        Init();
    }
    /**
     * Обучение нейронной сети
     *
     * \param input Вектор входных данных
     * \param output Вектор желаемых выходных данных
     * \return Ошибка
     */
    // TODO: Добавить возможность подавать сразу массивы входных и выходных данных
    double Train(const Vector& input, const Vector& output)
    {
        // Делаем прямой проход по сети,
        // попутно запоминая выходные значения каждого слоя
        // TODO: Реализовать прямой проход одним циклом
        // Проходим по первому слою, подавая на вход вектор входных данных
        m_outputs[0] = m_nn.Forward(input, 0);
        // Проходим по остальным слоям
        for (std::size_t i = 1; i < m_nn.LayersCount(); i++) {
            m_outputs[i] = m_nn.Forward(m_outputs[i - 1], i);
        }

        // Для удобства запомним индекс последнего слоя
        const std::size_t lastLayerIndex = m_nn.LayersCount() - 1;
        // TODO: Реализовать обратный проход одним циклом
        // Посчитаем ошибку на выходе сети.
        // Ошибка на выходе - это разность между выходом сети и желаемым выходом
        const Vector outputError = m_outputs[lastLayerIndex] - output;
        // Посчитаем градиенты на последнем слое.
        // Вектор градиентов последнего слоя - это произведение
        // вектора ошибок последнего слоя и вектора производных
        // от выходного вектора последнего слоя
        m_gradients[lastLayerIndex] = outputError
            * (m_outputs[lastLayerIndex].ApplyFunction(GetFunctionDerivative(m_nn.m_layers[lastLayerIndex].fn)));
        // Корректируем веса на последнем слое
        // Строка матрицы весов - это веса отдельного нейрона
        // Проходим по весам каждого нейрона последнего слоя
        for (std::size_t i = 0; i < m_nn.m_weights[lastLayerIndex].Rows(); i++) {
            // Вектор величин корректировки - это произведение вектора выходов предыдущего слоя,
            // градиента текущего нейрона и скорости обучения.
            // Уменьшаем вектор весов текущего нейрона на вектор с величинами корректировки
            m_nn.m_weights[lastLayerIndex][i] -= m_outputs[lastLayerIndex - 1]
                * m_gradients[lastLayerIndex][i] * m_learningRate;
        }

        // Проходим по другим слоям, от большего к меньшему, те двигаемся обратно,
        // от выходного слоя к входному (до первого слоя)
        for (std::size_t currentLayerIndex = lastLayerIndex - 1; currentLayerIndex >= 1; currentLayerIndex--) {
            // Посчитаем вектор ошибок текущего слоя - это произведение
            // транспонированной матрицы весов следующего слоя
            // и вектора градиентов следующего слоя
            auto currentLayerError = (m_nn.m_weights[currentLayerIndex + 1].Transpose())
                * m_gradients[currentLayerIndex + 1];
            // Посчитаем градиенты на текущем слое.
            // Вектор градиентов текущего слоя - это произведение
            // вектора ошибок текущего слоя и вектора производных
            // от выходного вектора текущего слоя
            m_gradients[currentLayerIndex] = currentLayerError
                * (m_outputs[currentLayerIndex].ApplyFunction(GetFunctionDerivative(m_nn.m_layers[currentLayerIndex].fn)));
            // Проходим по весам каждого нейрона текущего слоя
            for (std::size_t i = 0; i < m_nn.m_weights[currentLayerIndex].Rows(); i++) {
                // Вектор величин корректировки - это произведение вектора выходов предыдущего слоя,
                // градиента текущего нейрона и скорости обучения.
                // Уменьшаем вектор весов текущего нейрона на вектор с величинами корректировки
                m_nn.m_weights[currentLayerIndex][i] -= m_outputs[currentLayerIndex - 1]
                    * m_gradients[currentLayerIndex][i] * m_learningRate;
            }
        }

        // Посчитаем вектор ошибок первого слоя - это произведение
        // транспонированной матрицы весов следующего слоя
        // и вектора градиентов следующего слоя
        auto inputLayerError = (m_nn.m_weights[1].Transpose())
            * m_gradients[1];
        // Посчитаем градиенты на первом слое.
        // Вектор градиентов первого слоя - это произведение
        // вектора ошибок первого слоя и вектора производных
        // от выходного вектора первого слоя
        m_gradients[0] = inputLayerError
            * (m_outputs[0].ApplyFunction(GetFunctionDerivative(m_nn.m_layers[0].fn)));
        // Проходим по весам каждого нейрона первого слоя
        for (std::size_t i = 0; i < m_nn.m_weights[0].Rows(); i++) {
            // Вектор величин корректировки - это произведение вектора входных данных,
            // градиента текущего нейрона и скорости обучения.
            // Уменьшаем вектор весов текущего нейрона на вектор с величинами корректировки
            m_nn.m_weights[0][i] -= input
                * m_gradients[0][i] * m_learningRate;
        }
        // Обратный проход завершён
        // Посчитаем общую ошибку. Это будет среднеквадратичная ошибка.
        double error = 0.0;
        // Проходим по вектору выходных ошибок
        for (std::size_t i = 0; i < outputError.Size(); i++) {
            // Аккумулируем половину квадрата текущего элемента
            error += (outputError[i] * outputError[i]);
        }
        // Возвращаем общую ошибку
        return error / outputError.Size();
    }
private:
    // Ссылка на нейронную сеть
    NeuralNetwork& m_nn;
    // Скорость обучения
    double m_learningRate;
    // Массив векторов, содержащий выходные данные слоёв
    std::vector<Vector> m_outputs;
    // Массив векторов, содержащий градиенты слоёв
    std::vector<Vector> m_gradients;

    /**
     * Инициализация весов нейронной сети.
     *
     * \return 
     */
    void Init()
    {
        // Генератор случайных чисел
        std::random_device rd;
        // Движок для генерации случайных чисел
        std::mt19937 rng(rd());
        // Равномерное распределение в диапазоне от -0.5 до 0.5
        std::uniform_real_distribution<double> ds(-0.5, 0.5);
        // Проходим по матрицам весов каждого слоя
        for (std::size_t layer = 0; layer < m_nn.LayersCount(); layer++) {
            // Проходим по строкам матрицы весов текущего слоя
            for (std::size_t row = 0; row < m_nn.m_weights[layer].Rows(); row++) {
                // Проходим по столбцам матрицы весов текущего слоя
                for (std::size_t col = 0; col < m_nn.m_weights[layer].Cols(); col++) {
                    // Инициализируем текущий вес случайным значением
                    m_nn.m_weights[layer][row][col] = ds(rng);
                }
            }
        }
    }
};

}
