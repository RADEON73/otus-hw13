#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;

/**
 * @brief ��������� ������ ������������� ��������� �� �����
 * @param filename ���� � ����� � �������
 * @return ������ �������� ������������� ������ (�� ������ ������� �� ������ �����)
 * @note ������ �����: ������ ������ �������� ������������ ��� ������ ������,
 *       ����������� ��������� (������ �������� - intercept)
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
 * @brief ��������� ����������� �������������� � ������
 * @param model_row ������ ������������� ��� ����������� ������
 * @param features ������ ��������� (�������� �����������)
 * @return ����������� �������������� � ������ (�������� ���������� �������)
 */
double predict(const vector<double>& model_row, const vector<int>& features) {
    double sum = model_row[0]; // intercept
    for (size_t i = 1; i < model_row.size(); ++i) {
        sum += model_row[i] * features[i - 1];
    }
    return 1.0 / (1.0 + exp(-sum)); // sigmoid
}

/**
 * @brief ������������ �������� ������ � ��������� �������� ������
 * @param file ����� � ��������� �������
 * @param model ����������� ������ ������������� ���������
 * @param[out] correct ������� ���������� ������������
 * @param[out] total ����� ���������� ������������ ��������
 * @note ������ CSV: ������ �������� � ������ - ����� ������, ��������� - ��������
 */
void process_test_data(ifstream& file, const vector<vector<double>>& model,
    int& correct, int& total) {
    string line;
    // ���������� ���������
    getline(file, line);

    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        string value;

        // ������ ����� ������
        getline(ss, value, ',');
        int true_label = stoi(value);

        // ������ ��������
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }

        // ��������� ����������� ��� ������� ������
        vector<double> scores;
        for (const auto& model_row : model) {
            scores.push_back(predict(model_row, row));
        }

        // ���������� ������������� �����
        int pred_label = distance(scores.begin(),
            max_element(scores.begin(), scores.end()));

        // ��������� ��������
        if (pred_label == true_label) {
            correct++;
        }
        total++;
    }
}

/**
 * @brief ������� ������� ���������
 * @param argc ���������� ���������� ��������� ������
 * @param argv ������ ����������
 * @return ��� ���������� ���������
 * @note ��������� ������� ��� ���������:
 *       1. ���� � �������� ������ � ������� CSV
 *       2. ���� � ����� � ��������� �������
 */
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " test.csv model.txt" << endl;
        return 1;
    }

    // �������� ������
    auto model = load_model(argv[2]);

    // �������� ��������� �����
    ifstream test_file(argv[1]);
    if (!test_file.is_open()) {
        cerr << "Error opening test file" << endl;
        return 1;
    }

    int correct = 0;
    int total = 0;

    // ��������� ������ � ���������� ��������
    process_test_data(test_file, model, correct, total);

    // ����� ����������
    double accuracy = static_cast<double>(correct) / total;
    cout << "Accuracy: " << accuracy << endl;

    return 0;
}