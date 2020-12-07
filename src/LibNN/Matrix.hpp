#pragma once

#include "Vector.hpp"

namespace NN
{

/**
 * Класс, реализующий операции с матрицами.
 * Матрица представляет собой массив векторов одинакового размера.
 * Пример:
 * [ v1 ]   [[a11, a12, a13]]
 * [ v2 ] = [[a21, a22, a23]]
 * [ v3 ]   [[a31, a32, a33]]
 */
class Matrix
{
public:
    Matrix() = default;
    /**
     * Конструктор.
     *
     * \param rows Количество сторок
     * \param cols Количество столбцов
     */
    Matrix(const std::size_t rows, const std::size_t cols) :
        m_matrix(rows)
    {
        // Проходим по строкам
        for (std::size_t i = 0; i < m_matrix.size(); i++) {
            // и создаём вектор, который будет являтся строкой
            m_matrix[i] = Vector(cols);
        }
    }
    /**
     * Доступ к строке матрицы.
     *
     * \param index Индекс
     * \return Константная ссылка на строку матрицы
     */
    const Vector& operator [] (const std::size_t index) const
    {
        return m_matrix[index];
    }
    /**
     * Доступ к строке матрицы.
     *
     * \param index Индекс
     * \return Ссылка на строку матрицы
     */
    Vector& operator [] (const std::size_t index)
    {
        return m_matrix[index];
    }
    /**
     * Получение количества строк.
     *
     * \return Размер вектора
     */
    std::size_t Rows() const
    {
        return m_matrix.size();
    }
    /**
     * Получение количества столбцов.
     *
     * \return Размер вектора
     */
    std::size_t Cols() const
    {
        return m_matrix[0].Size();
    }
    /**
     * Умножение матрицы на вектор ("вектор-столбец").
     * Умножение происходит по правилам матричного умножения.
     * Пример:
     * [[a11, a12]]          [a11*b1 + a12*b2]
     * [[a21, a22]] * [b1] = [a21*b1 + a22*b2]
     * [[a31, a32]]   [b2]   [a31*b1 + a32*b2]
     * 
     * A(3x2) * B(2x1) = C(3x1)
     *
     * \param matrix Матрица
     * \param vector Вектор
     * \return Вектор
     */
    friend Vector operator * (const Matrix& matrix, const Vector& vector) noexcept(false)
    {
        // Количество столбцов матрицы должно быть равно размеру вектора
        if (matrix.Cols() != vector.Size()) {
            throw std::out_of_range("Number of columns of matrix must be equal to the size of vector");
        }
        // Результат
        Vector result(matrix.Rows());
        // Проходим по строкам
        for (std::size_t i = 0; i < matrix.Rows(); i++) {
            // Значение текущего элемента резульата -
            // это скалярное произведение текщей строки матрицы
            // на входной вектор
            result[i] = matrix[i] ^ vector;
        }
        // Возвращаем результат
        return result;
    }
    /**
     * Вычитание вектора из матрицы.
     * Пример:
     * [[a11, a12]]   [b1]   [a11-b1, a12-b1]
     * [[a21, a22]] - [b2] = [a21-b2, a22-b2]
     * [[a31, a32]]   [b3]   [a31-b3, a32-b3]
     *
     * \param matrix Матрица
     * \param vector Вектор
     * \return Матрица
     */
    friend Matrix operator - (const Matrix& matrix, const Vector& vector) noexcept(false)
    {
        // Количество строк матрицы должно быть равно размеру вектора
        if (matrix.Rows() != vector.Size()) {
            throw std::out_of_range("Number of rows of matrix must be equal to the size of vector");
        }
        // Результат
        Matrix result(matrix.Rows(), matrix.Cols());
        // Проходим по строкам
        for (std::size_t i = 0; i < matrix.Rows(); i++) {
            // Проходим по столбцам
            for (std::size_t j = 0; j < matrix.Cols(); j++) {
                result[i][j] = matrix[i][j] - vector[i];
            }
        }
        // Возвращаем результат
        return result;
    }
    /**
     * Получение транспонированной матрицы из текущей.
     * Пример:
     * [[a11, a12]]    [[a11, a21, a31]]
     * [[a21, a22]] -> [[a12, a22, a32]]
     * [[a31, a32]]
     *
     * \return Транспонированная матрица
     */
    Matrix Transpose() const
    {
        // Результат
        Matrix result(Cols(), Rows());
        // Проходим по строкам
        for (std::size_t row = 0; row < Rows(); row++) {
            // Проходим по столбцам
            for (std::size_t col = 0; col < Cols(); col++) {
                // Присваиваем элементу результата с позицией:
                // номер строки == номер текущего столбца
                // номер столбца == номер текущей строки
                // текущий элемент матрицы
                result[col][row] = m_matrix[row][col];
            }
        }
        // Возвращаем результат
        return result;
    }
private:
    std::vector<Vector> m_matrix;
};

}
