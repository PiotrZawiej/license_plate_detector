from grade import calculate_final_grade
from accuracy import accuracy_n_time

accuracy, time = accuracy_n_time()

print(calculate_final_grade(accuracy, time))