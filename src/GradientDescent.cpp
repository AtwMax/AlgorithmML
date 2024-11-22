#include "GradientDescent.h"
#include "utils.h"  // Include utils.h for computeGradient
#include <iostream>
#include <vector>

// Gradient Descent implementation
void GradientDescent::gradientDescent(double (*costFunction)(const std::vector<double>&),
                                      std::vector<double>& parameters, double learningRate, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // Compute the gradient
        std::vector<double> gradient = computeGradient(costFunction, parameters);

        // Update parameters using the gradient
        for (size_t j = 0; j < parameters.size(); ++j) {
            parameters[j] -= learningRate * gradient[j];
        }

        // Output the progress
        if (i % 100 == 0) {
            std::cout << "Iteration " << i << ", Cost: " << costFunction(parameters) << std::endl;
        }
    }
}
