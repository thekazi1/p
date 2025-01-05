import numpy as np

class ART1:
    def __init__(self, input_size, rho=0.9, alpha=0.1):
        """
        Initialize the ART1 network.
        
        :param input_size: Number of input features (input neurons)
        :param rho: Vigilance parameter (defines match threshold)
        :param alpha: Learning rate (controls weight update)
        """
        self.input_size = input_size  # Number of input features
        self.rho = rho  # Vigilance parameter
        self.alpha = alpha  # Learning rate
        self.weights = np.array([])  # Initialize weight matrix

    def train(self, data, max_epochs=10):
        """
        Train the ART1 model on the provided data.
        
        :param data: Training data (binary patterns)
        :param max_epochs: Number of training epochs
        """
        for epoch in range(max_epochs):
            for input_vector in data:
                input_vector = input_vector / np.linalg.norm(input_vector)  # Normalize input

                # Initialize weights if empty
                if self.weights.size == 0:
                    self.weights = np.zeros((1, self.input_size))

                # Compute match scores between input and weights
                match_scores = np.dot(self.weights, input_vector.T)
                best_match = np.argmax(match_scores)

                # Check if the match is above the vigilance threshold
                if match_scores[best_match] / np.linalg.norm(input_vector) >= self.rho:
                    # If match, update the weights
                    self.weights[best_match] += self.alpha * (input_vector - self.weights[best_match])
                else:
                    # If no match, create a new category (new row)
                    self.weights = np.vstack([self.weights, input_vector])

    def predict(self, input_vector):
        """
        Predict the category for a new input vector.
        
        :param input_vector: New input vector to predict category for
        :return: The index of the best matching category
        """
        input_vector = input_vector / np.linalg.norm(input_vector)  # Normalize input
        match_scores = np.dot(self.weights, input_vector.T)  # Compute match score
        return np.argmax(match_scores)  # Return index of best match

# Example usage
if __name__ == "__main__":
    # Example binary data (patterns)
    data = np.array([[1, 0, 0],  # Category 1
                     [0, 1, 0],  # Category 2
                     [0, 0, 1],  # Category 3
                     [1, 1, 0],  # Category 4
                     [1, 0, 1],  # Category 5
                     [0, 1, 1]]) # Category 6

    # Split data into training and testing sets
    train_data = data[:4]  # First 4 examples for training
    test_data = data[4:]   # Last 2 examples for testing

    # Initialize ART1 model
    art1 = ART1(input_size=3, rho=0.9, alpha=0.1)
    
    # Train the ART1 model
    art1.train(train_data, max_epochs=10)

    # Test the ART1 model on the test data
    print("Testing the model on test data:")
    for test_input in test_data:
        predicted_category = art1.predict(test_input)
        print(f"Input: {test_input}, Predicted category: {predicted_category}")
