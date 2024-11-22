#ifndef UTILS_H
#define UTILS_H

#include <vector>

// Function to compute the gradient using finite differences
std::vector<double> computeGradient(double (*costFunction)(const std::vector<double>&),
                                     const std::vector<double>& params, double epsilon = 1e-5);

#endif // UTILS_H

