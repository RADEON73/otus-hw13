#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;

/**
 * @brief Загружает модель логистической регрессии из файла
 * @param filename Путь к файлу с моделью
 * @return Вектор векторов коэффициентов модели (по одному вектору на каждый класс)
 * @note Формат файла: каждая строка содержит коэффициенты для одного класса,
 *       разделенные пробелами (первое значение - intercept)
 */
vector<vector<double>> load_model(const string& filename) {
    vector<vector<double>> model;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ' ')) {
            row.push_back(stod(value));
        }
        model.push_back(row);
    }

    return model;
}

/**
 * @brief Вычисляет вероятность принадлежности к классу
 * @param model_row Вектор коэффициентов для конкретного класса
 * @param features Вектор признаков (пикселей изображения)
 * @return Вероятность принадлежности к классу (значение сигмоидной функции)
 */
double predict(const vector<double>& model_row, const vector<int>& features) {
    double sum = model_row[0]; // intercept
    for (size_t i = 1; i < model_row.size(); ++i) {
        sum += model_row[i] * features[i - 1];
    }
    return 1.0 / (1.0 + exp(-sum)); // sigmoid
}

/**
 * @brief Обрабатывает тестовые данные и вычисляет точность модели
 * @param file Поток с тестовыми данными
 * @param model Загруженная модель логистической регрессии
 * @param[out] correct Счетчик правильных предсказаний
 * @param[out] total Общее количество обработанных примеров
 * @note Формат CSV: первое значение в строке - метка класса, остальные - признаки
 */
void process_test_data(ifstream& file, const vector<vector<double>>& model,
    int& correct, int& total) {
    string line;
    // Пропускаем заголовок
    getline(file, line);

    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        string value;

        // Читаем метку класса
        getline(ss, value, ',');
        int true_label = stoi(value);

        // Читаем признаки
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }

        // Вычисляем вероятности для каждого класса
        vector<double> scores;
        for (const auto& model_row : model) {
            scores.push_back(predict(model_row, row));
        }

        // Определяем предсказанный класс
        int pred_label = distance(scores.begin(),
            max_element(scores.begin(), scores.end()));

        // Обновляем счетчики
        if (pred_label == true_label) {
            correct++;
        }
        total++;
    }
}

/**
 * @brief Главная функция программы
 * @param argc Количество аргументов командной строки
 * @param argv Массив аргументов
 * @return Код завершения программы
 * @note Программа ожидает два аргумента:
 *       1. Путь к тестовым данным в формате CSV
 *       2. Путь к файлу с обученной моделью
 */
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " test.csv model.txt" << endl;
        return 1;
    }

    // Загрузка модели
    auto model = load_model(argv[2]);

    // Открытие тестового файла
    ifstream test_file(argv[1]);
    if (!test_file.is_open()) {
        cerr << "Error opening test file" << endl;
        return 1;
    }

    int correct = 0;
    int total = 0;

    // Обработка данных и вычисление точности
    process_test_data(test_file, model, correct, total);

    // Вывод результата
    double accuracy = static_cast<double>(correct) / total;
    cout << "Accuracy: " << accuracy << endl;

    return 0;
}