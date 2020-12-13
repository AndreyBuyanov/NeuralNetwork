#pragma once

#include <vector>
#include <functional>

namespace NN
{

/**
 * Класс, реализующий операции с векторами.
 */
class Vector
{
public:
    Vector() = default;
    /**
     * Конструктор.
     *
     * \param data Список инициализации
     */
    Vector(std::initializer_list<double> data):
        m_vector(data) {}
    /**
     * Конструктор.
     *
     * \param size Размер вектора
     */
    Vector(const std::size_t size):
        m_vector(size) {}
    /**
     * Доступ к элементам вектора.
     *
     * \param index Индекс
     * \return Константная ссылка на элемент вектора
     */
    const double& operator [] (const std::size_t index) const
    {
        return m_vector[index];
    }
    /**
     * Доступ к элементам вектора.
     *
     * \param index Индекс
     * \return Ссылка на элемент вектора
     */
    double& operator [] (const std::size_t index)
    {
        return m_vector[index];
    }
    /**
     *
     * \param value
     * \return
     */
    Vector& operator = (const double value)
    {
        for (std::size_t index = 0; index < Size(); ++index) {
            m_vector[index] = value;
        }
        return (*this);
    }
    /**
     * Получение размера вектора.
     *
     * \return Размер вектора
     */
    std::size_t Size() const noexcept
    {
        return m_vector.size();
    }
    /**
     * Скалярное произведение векторов.
     * Пример:
     * [a1, a2, a3] ^ [b1, b2, b3] = a1*b1 + a2*b2 + a3*b3
     *
     * \param v1 Первый вектор
     * \param v2 Второй вектор
     * \return Скалярное произведение
     */
    friend double operator ^ (const Vector& v1, const Vector& v2) noexcept(false)
    {
        // Вектора должны быть одинакового размера
        if (v1.Size() != v2.Size()) {
            throw std::out_of_range("Vectors must be the same size");
        }
        // Результат
        double result = 0.0;
        // Проходим по элементам векторов
        for (std::size_t index = 0; index < v1.Size(); ++index) {
            // Суммируем покомпонентное произведение векторов
            result += v1[index] * v2[index];
        }
        // Возвращаем результат
        return result;
    }
    /**
     * Покомпонентное произведение векторов.
     * Пример:
     * [a1, a2, a3] * [b1, b2, b3] = [a1*b1, a2*b2, a3*b3]
     * 
     * \param v1 Первый вектор
     * \param v2 Второй вектор
     * \return Покомпонентное произведение
     */
    friend Vector operator * (const Vector& v1, const Vector& v2) noexcept(false)
    {
        // Вектора должны быть одинакового размера
        if (v1.Size() != v2.Size()) {
            throw std::out_of_range("Vectors must be the same size");
        }
        // Результат
        Vector result(v1.Size());
        // Проходим по элементам векторов
        for (std::size_t index = 0; index < v1.Size(); ++index) {
            // Сохраняем в текущем элементе результата покомпонентное произведение векторов
            result[index] = v1[index] * v2[index];
        }
        // Возвращаем результат
        return result;
    }
    /**
     * Умножение вектора на число.
     * Пример:
     * [a1, a2, a3] * b = [a1*b, a2*b, a3*b]
     *
     * \param v Вектор
     * \param value Число
     * \return Вектор, умноженный на число
     */
    friend Vector operator * (const Vector& v, const double value) noexcept(false)
    {
        Vector result(v.Size());
        for (std::size_t index = 0; index < v.Size(); ++index) {
            result[index] = v[index] * value;
        }
        return result;
    }
    /**
     * Умножение числа на вектор.
     * Пример:
     * a * [b1, b2, b3] = [a*b1, a*b2, a*b3]
     *
     * \param value Число
     * \param v Вектор
     * \return Вектор, умноженный на число
     */
    friend Vector operator * (const double value, const Vector& v) noexcept(false)
    {
        return v * value;
    }
    /**
     * Сложение векторов.
     * Пример:
     * [a1, a2, a3] + [b1, b2, b3] = [a1+b1, a2+b2, a3+b3]
     *
     * \param v1 Первый вектор
     * \param v2 Второй вектор
     * \return Сумма векторов
     */
    friend Vector operator + (const Vector& v1, const Vector& v2) noexcept(false)
    {
        // Вектора должны быть одинакового размера
        if (v1.Size() != v2.Size()) {
            throw std::out_of_range("Vectors must be the same size");
        }
        // Результат
        Vector result(v1.Size());
        // Проходим по элементам векторов
        for (std::size_t index = 0; index < v1.Size(); ++index) {
            // Сохраняем в текущем элементе результата сумму векторов
            result[index] = v1[index] + v2[index];
        }
        // Возвращаем результат
        return result;
    }
    /**
     * Вычитание векторов.
     * Пример:
     * [a1, a2, a3] - [b1, b2, b3] = [a1-b1, a2-b2, a3-b3]
     * 
     * \param v1 Первый вектор
     * \param v2 Второй вектор
     * \return Разность векторов
     */
    friend Vector operator - (const Vector& v1, const Vector& v2) noexcept(false)
    {
        // Вектора должны быть одинакового размера
        if (v1.Size() != v2.Size()) {
            throw std::out_of_range("Vectors must be the same size");
        }
        // Результат
        Vector result(v1.Size());
        // Проходим по элементам векторов
        for (std::size_t index = 0; index < v1.Size(); ++index) {
            // Сохраняем в текущем элементе результата разность векторов
            result[index] = v1[index] - v2[index];
        }
        // Возвращаем результат
        return result;
    }
    /**
     * Вычитание векторов совмещённое с присваиванием.
     * Пример:
     * [a1, a2, a3] =- [b1, b2, b3] то же самое, что и
     * [a1, a2, a3] = [a1, a2, a3] - [b1, b2, b3]
     *
     * \param vector Входной вектор
     * \return Ссылка на текущий вектор
     */
    Vector& operator -= (const Vector& vector)
    {
        (*this) = (*this) - vector;
        return (*this);
    }
    /**
     * Применение функции к каждому элемента вектора.
     *
     * \param fn Функция
     * \return Вектор, элементы которого содержат элементы текущего вектора
     * с применённым к ним функцией
     */
    Vector ApplyFunction(const std::function<double(double)>& fn) const
    {
        // Результат
        Vector result(Size());
        // Проходим по элементам векторов
        for (std::size_t index = 0; index < Size(); ++index) {
            // Сохраняем в текущем элементе результата
            // значение текущего элемента с применённым к нему функцией
            result[index] = fn(m_vector[index]);
        }
        // Возвращаем результат
        return result;
    }
private:
    std::vector<double> m_vector;
};

}
