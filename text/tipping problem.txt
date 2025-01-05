import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define the fuzzy sets for inputs and output
def fuzzy_tip_calculation():
    # Create a range of values for Bill Amount and Service Quality
    bill_range = np.arange(0, 501, 1)  # Bill amounts from $0 to $500
    service_range = np.arange(0, 11, 1)  # Service Quality from 0 (Bad) to 10 (Good)
    tip_range = np.arange(0, 26, 1)  # Tip percentage from 0% to 25%

    # Fuzzification of Bill Amount (Low, Medium, High)
    bill_low = fuzz.trimf(bill_range, [0, 0, 150])
    bill_medium = fuzz.trimf(bill_range, [100, 250, 400])
    bill_high = fuzz.trimf(bill_range, [300, 500, 500])

    # Fuzzification of Service Quality (Bad, Average, Good)
    service_bad = fuzz.trimf(service_range, [0, 0, 5])
    service_average = fuzz.trimf(service_range, [3, 5, 7])
    service_good = fuzz.trimf(service_range, [6, 10, 10])

    # Fuzzification of Tip Percentage (Low, Medium, High)
    tip_low = fuzz.trimf(tip_range, [0, 0, 10])
    tip_medium = fuzz.trimf(tip_range, [5, 12, 18])
    tip_high = fuzz.trimf(tip_range, [15, 25, 25])

    # Visualize the fuzzy sets
    plt.figure(figsize=(10, 7))
    plt.subplot(311)
    plt.plot(bill_range, bill_low, label="Low", color='r')
    plt.plot(bill_range, bill_medium, label="Medium", color='g')
    plt.plot(bill_range, bill_high, label="High", color='b')
    plt.title("Bill Amount")
    plt.legend()

    plt.subplot(312)
    plt.plot(service_range, service_bad, label="Bad", color='r')
    plt.plot(service_range, service_average, label="Average", color='g')
    plt.plot(service_range, service_good, label="Good", color='b')
    plt.title("Service Quality")
    plt.legend()

    plt.subplot(313)
    plt.plot(tip_range, tip_low, label="Low", color='r')
    plt.plot(tip_range, tip_medium, label="Medium", color='g')
    plt.plot(tip_range, tip_high, label="High", color='b')
    plt.title("Tip Percentage")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Define inputs (example)
    bill_amount = 200  # Input bill amount
    service_quality = 8  # Input service quality

    # Fuzzify the inputs (using membership functions)
    bill_low_level = fuzz.interp_membership(bill_range, bill_low, bill_amount)
    bill_medium_level = fuzz.interp_membership(bill_range, bill_medium, bill_amount)
    bill_high_level = fuzz.interp_membership(bill_range, bill_high, bill_amount)

    service_bad_level = fuzz.interp_membership(service_range, service_bad, service_quality)
    service_average_level = fuzz.interp_membership(service_range, service_average, service_quality)
    service_good_level = fuzz.interp_membership(service_range, service_good, service_quality)

    # Apply fuzzy rules to combine inputs into tip output
    # Rule 1: If Bill is Low and Service is Bad, then Tip is Low
    tip_low_level = np.fmin(bill_low_level, service_bad_level)

    # Rule 2: If Bill is Low and Service is Average, then Tip is Low or Medium
    tip_medium_level = np.fmin(bill_low_level, service_average_level)

    # Rule 3: If Bill is Medium and Service is Average, then Tip is Medium
    tip_medium_level2 = np.fmin(bill_medium_level, service_average_level)

    # Rule 4: If Bill is Medium and Service is Good, then Tip is Medium or High
    tip_high_level = np.fmin(bill_medium_level, service_good_level)

    # Rule 5: If Bill is High and Service is Good, then Tip is High
    tip_high_level2 = np.fmin(bill_high_level, service_good_level)

    # Aggregate the outputs (combine the fuzzy output sets)
    aggregated_tip = np.fmax(tip_low_level, np.fmax(tip_medium_level, np.fmax(tip_medium_level2, np.fmax(tip_high_level, tip_high_level2))))

    # Defuzzify (convert fuzzy result to crisp value)
    tip_percentage = fuzz.defuzz(tip_range, aggregated_tip, 'centroid')

    print(f"Suggested Tip Percentage: {tip_percentage:.2f}%")

# Run the fuzzy tip calculation
fuzzy_tip_calculation()
