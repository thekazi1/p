import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, num_neurons, epsilon=1e-5):
        self.num_neurons = num_neurons
        self.epsilon = epsilon  # Convergence tolerance
        self.weights = np.zeros((num_neurons, num_neurons))  # Initialize weights as zero
    
    def train(self, patterns):
        """
        Train the network to store the given patterns.
        Patterns should be a list of binary patterns (as arrays).
        """
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)  # Outer product of the pattern with itself
        
        # Set diagonal to 0 (no self-connections)
        np.fill_diagonal(self.weights, 0)
    
    def recall(self, noisy_input, max_iterations=100):
        """
        Recall a stored pattern from a noisy input using asynchronous updates.
        noisy_input: The noisy or incomplete input pattern.
        max_iterations: How many iterations to update the neurons.
        """
        state = np.array(noisy_input)  # Convert input to an array
        energy_history = []  # Track energy over time for visualization
        
        for _ in range(max_iterations):
            # Pick a random neuron to update
            i = np.random.randint(self.num_neurons)
            
            # Calculate the sum of inputs to neuron i
            input_sum = np.dot(self.weights[i], state)
            
            # Update the state of neuron i (1 if sum > 0, -1 if sum <= 0)
            state[i] = 1 if input_sum > 0 else -1
            
            # Calculate the current energy
            energy = self.energy(state)
            energy_history.append(energy)
            
            # Check for convergence (if energy change is less than epsilon)
            if len(energy_history) > 1 and abs(energy_history[-1] - energy_history[-2]) < self.epsilon:
                print("Network has converged!")
                break
        
        return state, energy_history  # Return the final recalled pattern and energy history
    
    def energy(self, state):
        """
        Calculate the energy of a given state. The energy should decrease as the network stabilizes.
        """
        return -0.5 * np.sum(np.dot(state, self.weights) * state)
    
# Helper function to convert patterns to binary (-1, 1)
def binarize(pattern):
    return np.where(pattern > 0, 1, -1)

# Example usage
if __name__ == "__main__":
    # Define some binary patterns (as arrays of 1s and -1s)
    pattern1 = np.array([1, 1, -1, -1])
    pattern2 = np.array([1, -1, 1, -1])
    
    # Create a Hopfield network with 4 neurons (one for each element in the patterns)
    hopfield = HopfieldNetwork(num_neurons=4, epsilon=1e-5)
    
    # Train the network with the patterns
    hopfield.train([pattern1, pattern2])
    
    # Recall a pattern from a noisy input (we add noise by flipping a bit)
    noisy_input = np.array([1, 1, 1, -1])  # A noisy version of pattern1
    print("Noisy input:", noisy_input)
    
    # Recall the original pattern from the noisy input
    recalled_pattern, energy_history = hopfield.recall(noisy_input)
    print("Recalled pattern:", recalled_pattern)
    
    # Visualize the energy changes using Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot(energy_history, label="Energy", color='blue')
    plt.title("Energy Decrease During Recall")
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()
    
    # Visualize the weight matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(hopfield.weights, annot=True, cmap="coolwarm", linewidths=0.5, cbar=True)
    plt.title("Weight Matrix")
    plt.show()
