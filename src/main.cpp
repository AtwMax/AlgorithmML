#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>  // For system calls
#include "OptimizationMethods.h"
#include "utils.h"  // Include utils.h for computeGradient

// Rosenbrock Cost Function
double rosenbrockCostFunction(const std::vector<double>& params) {
    double cost = 0.0;
    for (size_t i = 0; i < params.size() - 1; ++i) {
        double term1 = params[i + 1] - params[i] * params[i];
        double term2 = 1 - params[i];
        cost += 100 * term1 * term1 + term2 * term2;
    }
    return cost;
}

// Function to write data to a file for plotting
void writeDataForPlotting(const std::vector<std::vector<double>>& allCosts) {
    std::ofstream file("costs.dat");  // Open a file to save the costs
    size_t maxIterations = 0;
    for (const auto& costs : allCosts) {
        maxIterations = std::max(maxIterations, costs.size());
    }

    for (size_t i = 0; i < maxIterations; ++i) {
        file << i;
        for (const auto& costs : allCosts) {
            file << " " << (i < costs.size() ? costs[i] : 0);  // Avoid out-of-bounds access
        }
        file << "\n";
    }
    file.close();
}

// Function to call the Python script to plot the data
void plotWithPython() {
    // Call the Python script for plotting
    std::system("python src/plot_data.py");
}

int main() {
    // Initial parameters
    std::vector<double> params = {-1.2, 1.0}; // Example starting point
    double learningRate = 0.001;
    int iterations = 1000;

    // Vectors to store cost for each iteration
    std::vector<std::vector<double>> allCosts(3);

    // Using Stochastic Gradient Descent (SGD)
    std::cout << "Running Stochastic Gradient Descent..." << std::endl;
    std::vector<double> paramsSGD = params;  // Copy parameters for SGD
    OptimizationMethods::stochasticGradientDescent(rosenbrockCostFunction, paramsSGD, learningRate, iterations, allCosts[0]);

    // Using Adam Optimizer
    std::cout << "Running Adam Optimizer..." << std::endl;
    std::vector<double> paramsAdam = params;  // Copy parameters for Adam
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    OptimizationMethods::adamOptimizer(rosenbrockCostFunction, paramsAdam, learningRate, iterations, beta1, beta2, epsilon, allCosts[1]);

    // Using Momentum Optimizer
    std::cout << "Running Momentum Optimizer..." << std::endl;
    std::vector<double> paramsMomentum = params;  // Copy parameters for Momentum
    double beta = 0.9;  // Momentum factor
    OptimizationMethods::momentumOptimizer(rosenbrockCostFunction, paramsMomentum, learningRate, iterations, beta, allCosts[2]);

    // Write data to a file for plotting
    writeDataForPlotting(allCosts);

    // Call the Python script to plot the data
    plotWithPython();

    return 0;
}
