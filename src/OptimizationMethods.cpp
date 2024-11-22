#include "OptimizationMethods.h"
#include "utils.h"  // For computeGradient
#include <cmath>    // For pow, sqrt
#include <iostream> // For std::cout

// Stochastic Gradient Descent
void OptimizationMethods::stochasticGradientDescent(double (*costFunction)(const std::vector<double>&),
                                                    std::vector<double>& parameters, double learningRate, 
                                                    int iterations, std::vector<double>& costHistory) {
    for (int i = 0; i < iterations; ++i) {
        std::vector<double> gradient = computeGradient(costFunction, parameters);

        for (size_t j = 0; j < parameters.size(); ++j) {
            parameters[j] -= learningRate * gradient[j];
        }

        double cost = costFunction(parameters);  // Compute the current cost
        costHistory.push_back(cost);  // Track the cost history

        if (i % 100 == 0) {
            std::cout << "Iteration " << i << ", Cost: " << cost << std::endl;
        }
    }
}

// Adam Optimizer
void OptimizationMethods::adamOptimizer(double (*costFunction)(const std::vector<double>&), 
                                        std::vector<double>& parameters, double learningRate, 
                                        int iterations, double beta1, double beta2, 
                                        double epsilon, std::vector<double>& costHistory) {
    std::vector<double> m(parameters.size(), 0.0); // First moment vector
    std::vector<double> v(parameters.size(), 0.0); // Second moment vector

    for (int t = 1; t <= iterations; ++t) {
        std::vector<double> gradient = computeGradient(costFunction, parameters);

        for (size_t j = 0; j < parameters.size(); ++j) {
            m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
            v[j] = beta2 * v[j] + (1 - beta2) * gradient[j] * gradient[j];

            double m_hat = m[j] / (1 - std::pow(beta1, t));
            double v_hat = v[j] / (1 - std::pow(beta2, t));

            parameters[j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        double cost = costFunction(parameters);
        costHistory.push_back(cost);

        if (t % 100 == 0) {
            std::cout << "Iteration " << t << ", Cost: " << cost << std::endl;
        }
    }
}

// Momentum Optimizer
void OptimizationMethods::momentumOptimizer(double (*costFunction)(const std::vector<double>&),
                                            std::vector<double>& parameters, double learningRate,
                                            int iterations, double beta, std::vector<double>& costHistory) {
    std::vector<double> velocity(parameters.size(), 0.0);

    for (int i = 0; i < iterations; ++i) {
        std::vector<double> gradient = computeGradient(costFunction, parameters);

        for (size_t j = 0; j < parameters.size(); ++j) {
            velocity[j] = beta * velocity[j] + (1 - beta) * gradient[j];
            parameters[j] -= learningRate * velocity[j];
        }

        double cost = costFunction(parameters);
        costHistory.push_back(cost);

        if (i % 100 == 0) {
            std::cout << "Iteration " << i << ", Cost: " << cost << std::endl;
        }
    }
}

// Nesterov Accelerated Gradient (NAG)
void OptimizationMethods::nesterovAcceleratedGradient(double (*costFunction)(const std::vector<double>&),
                                                      std::vector<double>& parameters, double learningRate,
                                                      int iterations, double beta, std::vector<double>& costHistory) {
    std::vector<double> velocity(parameters.size(), 0.0);

    for (int i = 0; i < iterations; ++i) {
        // Lookahead step
        std::vector<double> lookaheadParameters = parameters;
        for (size_t j = 0; j < parameters.size(); ++j) {
            lookaheadParameters[j] -= beta * velocity[j];
        }

        std::vector<double> gradient = computeGradient(costFunction, lookaheadParameters);

        for (size_t j = 0; j < parameters.size(); ++j) {
            velocity[j] = beta * velocity[j] + (1 - beta) * gradient[j];
            parameters[j] -= learningRate * velocity[j];
        }

        double cost = costFunction(parameters);
        costHistory.push_back(cost);

        if (i % 100 == 0) {
            std::cout << "Iteration " << i << ", Cost: " << cost << std::endl;
        }
    }
}
