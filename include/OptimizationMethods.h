#ifndef OPTIMIZATION_METHODS_H
#define OPTIMIZATION_METHODS_H

#include <vector>

class OptimizationMethods {
public:
    // Stochastic Gradient Descent
    static void stochasticGradientDescent(double (*costFunction)(const std::vector<double>&),
                                          std::vector<double>& parameters, double learningRate, 
                                          int iterations, std::vector<double>& costHistory);

    // Adam Optimizer
    static void adamOptimizer(double (*costFunction)(const std::vector<double>&), 
                              std::vector<double>& parameters, double learningRate, 
                              int iterations, double beta1, double beta2, 
                              double epsilon, std::vector<double>& costHistory);

    // Momentum Optimizer
    static void momentumOptimizer(double (*costFunction)(const std::vector<double>&),
                                  std::vector<double>& parameters, double learningRate,
                                  int iterations, double beta, std::vector<double>& costHistory);

    // Nesterov Accelerated Gradient (NAG)
    static void nesterovAcceleratedGradient(double (*costFunction)(const std::vector<double>&),
                                            std::vector<double>& parameters, double learningRate,
                                            int iterations, double beta, std::vector<double>& costHistory);
};

#endif
