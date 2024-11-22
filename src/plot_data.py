import matplotlib.pyplot as plt

def plot_costs(filename="costs.dat"):
    # Read the data from the file
    iterations = []
    sgd_costs = []
    adam_costs = []
    momentum_costs = []

    with open(filename, "r") as f:
        for line in f:
            data = line.split()
            iterations.append(int(data[0]))
            sgd_costs.append(float(data[1]))
            adam_costs.append(float(data[2]))
            momentum_costs.append(float(data[3]))

    # Plot the data
    plt.plot(iterations, sgd_costs, label="SGD")
    plt.plot(iterations, adam_costs, label="Adam")
    plt.plot(iterations, momentum_costs, label="Momentum")

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Optimization Method Comparison')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_costs()