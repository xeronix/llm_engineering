#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(int iterations, double param1, double param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    using namespace std::chrono;
    
    auto start = high_resolution_clock::now();
    double result = calculate(100000000, 4.0, 1.0) * 4.0;
    auto end = high_resolution_clock::now();
    
    duration<double> elapsed = end - start;
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << std::endl;
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}
