#include "utils.h"
#include <vector>

std::vector<double> computeGradient(double (*costFunction)(const std::vector<double>&),
                                     const std::vector<double>& params, double epsilon) {
    std::vector<double> gradient(params.size(), 0.0);
    double originalCost = costFunction(params);

    for (size_t i = 0; i < params.size(); ++i) {
        // Create a copy of parameters and perturb the i-th parameter
        std::vector<double> perturbedParams = params;
        perturbedParams[i] += epsilon;

        // Calculate the cost with the perturbed parameter
        double perturbedCost = costFunction(perturbedParams);

        // Compute the gradient as (f(x + epsilon) - f(x)) / epsilon
        gradient[i] = (perturbedCost - originalCost) / epsilon;
    }

    return gradient;
}
