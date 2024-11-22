#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <vector>
#include "utils.h"  // Include utils.h for computeGradient

class GradientDescent {
public:
    static void gradientDescent(double (*costFunction)(const std::vector<double>&),
                                 std::vector<double>& parameters, double learningRate, int iterations);
};

#endif // GRADIENT_DESCENT_H
