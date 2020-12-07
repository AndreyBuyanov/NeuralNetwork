#pragma once

#include <functional>

namespace NN
{

namespace detail{

    // TODO: Реализовать другие функции активации
    /**
     * Сигмоидальная функция активации.
     *
     * \param input Входное значение
     * \return Значение сигмоидальной функции
     */
    static double Sigmoid(const double input)
    {
        return 1.0 / (1.0 + std::exp(-input));
    }
    /**
     * Производная сигмоидальной функции активации.
     *
     * \param input Входное значение
     * \return Значение производной сигмоидальной функции
     */
    static double SigmoidDerivative(const double input)
    {
        return input * (1.0 - input);
    }

}

/**
 * Тип функции активации.
 */
// TODO: Добавить другие функции активации
enum class ActivationFunction
{
    Sigmoid     // Сигмоида
};

/**
 * Получение функции активации по типу.
 *
 * \param fn Тип функции активации
 * \return Функция активации
 */
// TODO: MSVC - избавиться от варнинга C4715
std::function<double(double)> GetFunction(const ActivationFunction fn)
{
    switch (fn) {
    case ActivationFunction::Sigmoid:
        return detail::Sigmoid;
    }
}

/**
 * Получение производной функции активации по типу.
 *
 * \param fn Тип функции активации
 * \return Производная функции активации
 */
 // TODO: MSVC - избавиться от варнинга C4715
std::function<double(double)> GetFunctionDerivative(const ActivationFunction fn)
{
    switch (fn) {
    case ActivationFunction::Sigmoid:
        return detail::SigmoidDerivative;
    }
}

}
