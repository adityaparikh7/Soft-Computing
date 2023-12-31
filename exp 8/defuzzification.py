# import numpy as np
# import matplotlib.pyplot as plt

# # Define the fuzzy output as a membership function
# x = np.linspace(0, 10, 100)
# membership_function = 1 / (1 + np.exp(-(x - 5)))

# # Defuzzification methods
# def max_membership(membership_function, x):
#     return x[membership_function.argmax()]

# def centroid(membership_function, x):
#     return np.trapz(x * membership_function, x) / np.trapz(membership_function, x)

# def center_of_sums(membership_function, x):
#     return np.sum(x * membership_function) / np.sum(membership_function)

# def mean_of_maxima(membership_function, x):
#     maxima = x[membership_function == membership_function.max()]
#     return maxima.mean()

# def weighted_average(membership_function, x):
#     return np.sum(x * membership_function) / np.sum(membership_function)

# def center_of_largest_area(membership_function, x):
#     area = np.trapz(membership_function, x)
#     return np.trapz(x * membership_function, x) / (2 * area)

# # Calculate defuzzified values
# max_membership_value = max_membership(membership_function, x)
# centroid_value = centroid(membership_function, x)
# center_of_sums_value = center_of_sums(membership_function, x)
# mean_of_maxima_value = mean_of_maxima(membership_function, x)
# weighted_average_value = weighted_average(membership_function, x)
# center_of_largest_area_value = center_of_largest_area(membership_function, x)

# # Plot the membership function and defuzzification points
# plt.plot(x, membership_function, label='Membership Function', color='blue')
# plt.axvline(x=max_membership_value, linestyle='--', color='red', label='Max Membership')
# plt.axvline(x=centroid_value, linestyle='--', color='green', label='Centroid')
# plt.axvline(x=center_of_sums_value, linestyle='--', color='purple', label='Center of Sums')
# plt.axvline(x=mean_of_maxima_value, linestyle='--', color='orange', label='Mean of Maxima')
# plt.axvline(x=weighted_average_value, linestyle='--', color='pink', label='Weighted Average')
# plt.axvline(x=center_of_largest_area_value, linestyle='--', color='brown', label='Center of Largest Area')
# plt.legend()
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Fuzzy Output Defuzzification')
# plt.show()

# # Display the defuzzified values
# print("Max Membership: {:.2f}".format(max_membership_value))
# print("Centroid: {:.2f}".format(centroid_value))
# print("Center of Sums: {:.2f}".format(center_of_sums_value))
# print("Mean of Maxima: {:.2f}".format(mean_of_maxima_value))
# print("Weighted Average: {:.2f}".format(weighted_average_value))
# print("Center of Largest Area: {:.2f}".format(center_of_largest_area_value))


# import numpy as np
# import matplotlib.pyplot as plt

# # Define the fuzzy output as a membership function
# x = np.linspace(0, 10, 100)
# membership_function = 1 / (1 + np.exp(-(x - 5)))

# def max_membership(membership_function, x):
#     return x[membership_function.argmax()]

# def centroid(membership_function, x):
#     return np.trapz(x * membership_function, x) / np.trapz(membership_function, x)

# def center_of_sums(membership_function, x):
#     return np.sum(x * membership_function) / np.sum(membership_function)

# def mean_of_maxima(membership_function, x):
#     maxima = x[membership_function == membership_function.max()]
#     return maxima.mean()

# def weighted_average(membership_function, x):
#     return np.sum(x * membership_function) / np.sum(membership_function)

# def center_of_largest_area(membership_function, x):
#     area = np.trapz(membership_function, x)
#     return np.trapz(x * membership_function, x) / (2 * area)



# while True:
#     print("\nSelect a defuzzification method:")
#     print("1. Max Membership")
#     print("2. Centroid")
#     print("3. Center of Sums")
#     print("4. Mean of Maxima")
#     print("5. Weighted Average")
#     print("6. Center of Largest Area")
#     print("7. Lambda Cut")
#     print("0. Exit")

#     choice = input("Enter your choice: ")

#     if choice == '0':
#         break
#     elif choice == '1':
#         result = max_membership(membership_function, x)
#     elif choice == '2':
#         result = centroid(membership_function, x)
#     elif choice == '3':
#         result = center_of_sums(membership_function, x)
#     elif choice == '4':
#         result = mean_of_maxima(membership_function, x)
#     elif choice == '5':
#         result = weighted_average(membership_function, x)
#     elif choice == '6':
#         result = center_of_largest_area(membership_function, x)
#     else:
#         print("Invalid choice. Please select a valid option.")
#         continue

#     # Plot the membership function and the defuzzified point
#     plt.plot(x, membership_function, label='Membership Function', color='blue')
#     plt.axvline(x=result, linestyle='--', color='red', label='Defuzzified Value')
#     plt.legend()
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title('Fuzzy Output Defuzzification')
#     plt.show()

#     print(f"Defuzzified Value: {result:.2f}")


import numpy as np
import matplotlib.pyplot as plt

# Define the fuzzy output as a membership function
x = np.linspace(0, 10, 100)
membership_function = 1 / (1 + np.exp(-(x - 5)))

def lambda_cut(membership_function, x, cut):
    cut_indices = np.where(membership_function >= cut)
    if len(cut_indices[0]) > 0:
        return np.mean(x[cut_indices])
    return None

def weighted_average(membership_function, x):
    return np.sum(x * membership_function) / np.sum(membership_function)

def height_of_maxima(membership_function, x):
    return membership_function.max()

def first_of_maxima(membership_function, x):
    maxima = x[membership_function == membership_function.max()]
    return maxima[0] if maxima.size > 0 else None

def last_of_maxima(membership_function, x):
    maxima = x[membership_function == membership_function.max()]
    return maxima[-1] if maxima.size > 0 else None

def mean_of_maxima(membership_function, x):
    maxima = x[membership_function == membership_function.max()]
    return maxima.mean() if maxima.size > 0 else None

def center_of_centroid(membership_function, x):
    centroid = np.trapz(x * membership_function, x) / np.trapz(membership_function, x)
    return centroid

def center_of_sum(membership_function, x):
    return np.sum(x * membership_function) / np.sum(membership_function)

while True:
    print("\nSelect a defuzzification method:")
    print("1. Lambda Cut")
    print("2. Weighted Average")
    print("3. Height of Maxima")
    print("4. First of Maxima")
    print("5. Last of Maxima")
    print("6. Mean of Maxima")
    print("7. Center of Centroid")
    print("8. Center of Sum")
    print("0. Exit")

    choice = input("Enter your choice: ")

    if choice == '0':
        break
    elif choice == '1':
        cut = float(input("Enter the lambda cut value: "))
        result = lambda_cut(membership_function, x, cut)
        if result is None:
            print("No values found within the lambda cut.")
            continue
    elif choice == '2':
        result = weighted_average(membership_function, x)
    elif choice == '3':
        result = height_of_maxima(membership_function, x)
    elif choice == '4':
        result = first_of_maxima(membership_function, x)
        if result is None:
            print("No maxima found.")
            continue
    elif choice == '5':
        result = last_of_maxima(membership_function, x)
        if result is None:
            print("No maxima found.")
            continue
    elif choice == '6':
        result = mean_of_maxima(membership_function, x)
        if result is None:
            print("No maxima found.")
            continue
    elif choice == '7':
        result = center_of_centroid(membership_function, x)
    elif choice == '8':
        result = center_of_sum(membership_function, x)
    else:
        print("Invalid choice. Please select a valid option.")
        continue

    # Plot the membership function and the defuzzified point
    plt.plot(x, membership_function, label='Membership Function', color='blue')
    plt.axhline(y=result, linestyle='--', color='red', label='Defuzzified Value')
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Fuzzy Output Defuzzification')
    plt.show()

    print(f"Defuzzified Value: {result:.2f}")
