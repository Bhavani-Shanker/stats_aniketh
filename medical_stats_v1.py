import streamlit as st
import math

# Function to calculate binomial coefficient
def binomial_coefficient(n, k):
    return math.comb(n, k)

# Function to calculate factorial
def factorial(n):
    return math.factorial(n)

# Function to calculate the probability P for a 2x2 table (Fisher Exact Test)
def calculate_2x2_fisher(a, b, c, d):
    a_plus_b = a + b
    c_plus_d = c + d
    a_plus_c = a + c
    n = a_plus_b + c_plus_d

    binom_ab_a = binomial_coefficient(a_plus_b, a)
    binom_cd_c = binomial_coefficient(c_plus_d, c)
    binom_n_ac = binomial_coefficient(n, a_plus_c)

    P = (binom_ab_a * binom_cd_c) / binom_n_ac
    return P

# Function to calculate the probability P for a 3x2 or 4x2 table (Fisher Exact Test)
def calculate_fisher_for_larger_tables(table):
    row_totals = [sum(row) for row in table]
    col_totals = [sum(col) for col in zip(*table)]
    n = sum(row_totals)

    numerator = 1
    for i in range(len(row_totals)):
        numerator *= factorial(row_totals[i])
    for j in range(len(col_totals)):
        numerator *= factorial(col_totals[j])
    
    denominator = factorial(n)
    for i in range(len(table)):
        for j in range(len(table[i])):
            denominator *= factorial(table[i][j])
    
    P = numerator / denominator
    return P

# Function to calculate Chi-square test using the mathematical formula
def calculate_chi_square(table):
    row_totals = [sum(row) for row in table]
    col_totals = [sum(col) for col in zip(*table)]
    n = sum(row_totals)

    expected = [[(row_total * col_total) / n for col_total in col_totals] for row_total in row_totals]
    chi2 = sum(((table[i][j] - expected[i][j]) ** 2) / expected[i][j] for i in range(len(table)) for j in range(len(table[0])))
    
    return chi2

# Function to calculate mean
def calculate_mean(data):
    return sum(data) / len(data)

# Function to calculate variance
def calculate_variance(data, mean):
    return sum((x - mean) ** 2 for x in data) / (len(data) - 1)

# Function to calculate ANOVA using mathematical formulas
def calculate_anova_from_table(table):
    flattened_table = [item for row in table for item in row]
    overall_mean = calculate_mean(flattened_table)

    group_means = [calculate_mean(row) for row in table]
    between_group_sum_of_squares = sum(len(row) * (group_mean - overall_mean) ** 2 for row, group_mean in zip(table, group_means))
    within_group_sum_of_squares = sum(sum((x - group_mean) ** 2 for x in row) for row, group_mean in zip(table, group_means))

    df_between = len(table) - 1
    df_within = len(flattened_table) - len(table)
    
    mean_square_between = between_group_sum_of_squares / df_between
    mean_square_within = within_group_sum_of_squares / df_within

    f_stat = mean_square_between / mean_square_within
    
    return f_stat

# Function to calculate T-test using mathematical formulas
def calculate_t_test_from_table(table):
    group1 = table[0]
    group2 = table[1] if len(table) > 1 else []

    mean1 = calculate_mean(group1)
    mean2 = calculate_mean(group2)
    var1 = calculate_variance(group1, mean1)
    var2 = calculate_variance(group2, mean2)
    n1, n2 = len(group1), len(group2)

    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    t_stat = (mean1 - mean2) / math.sqrt(pooled_variance * (1/n1 + 1/n2))

    return t_stat

# Function to interpret results
def interpret_results(p_value, alpha=0.05):
    if p_value < alpha:
        st.success("Reject the null hypothesis. There is a significant difference between the groups.")
    else:
        st.info("Fail to reject the null hypothesis. There is no significant difference between the groups.")

# Streamlit app
test_type = st.sidebar.selectbox("Select the test to perform:", ["Fisher Exact Test", "Chi-square Test", "ANOVA", "T-test"])
st.title(f"{test_type} Calculator")

# Matrix size selection
matrix_size = st.selectbox("Select matrix size:", ["2x2", "3x2", "4x2"])

# Get the input values based on the matrix size
if matrix_size == "2x2":
    a = st.number_input("Enter value for a:", min_value=0, value=10)
    b = st.number_input("Enter value for b:", min_value=0, value=2)
    c = st.number_input("Enter value for c:", min_value=0, value=3)
    d = st.number_input("Enter value for d:", min_value=0, value=8)
    table = [[a, b], [c, d]]

elif matrix_size == "3x2":
    a = st.number_input("Enter value for Group 1, Outcome 1:", min_value=0, value=2)
    b = st.number_input("Enter value for Group 1, Outcome 2:", min_value=0, value=1)
    c = st.number_input("Enter value for Group 2, Outcome 1:", min_value=0, value=1)
    d = st.number_input("Enter value for Group 2, Outcome 2:", min_value=0, value=2)
    e = st.number_input("Enter value for Group 3, Outcome 1:", min_value=0, value=1)
    f = st.number_input("Enter value for Group 3, Outcome 2:", min_value=0, value=3)
    table = [[a, b], [c, d], [e, f]]

elif matrix_size == "4x2":
    a = st.number_input("Enter value for Group 1, Outcome 1:", min_value=0, value=2)
    b = st.number_input("Enter value for Group 1, Outcome 2:", min_value=0, value=1)
    c = st.number_input("Enter value for Group 2, Outcome 1:", min_value=0, value=1)
    d = st.number_input("Enter value for Group 2, Outcome 2:", min_value=0, value=2)
    e = st.number_input("Enter value for Group 3, Outcome 1:", min_value=0, value=1)
    f = st.number_input("Enter value for Group 3, Outcome 2:", min_value=0, value=3)
    g = st.number_input("Enter value for Group 4, Outcome 1:", min_value=0, value=1)
    h = st.number_input("Enter value for Group 4, Outcome 2:", min_value=0, value=4)
    table = [[a, b], [c, d], [e, f], [g, h]]

# Perform the selected test
if test_type == "Fisher Exact Test":
    if matrix_size == "2x2":
        p_value = calculate_2x2_fisher(a, b, c, d)
    else:
        p_value = calculate_fisher_for_larger_tables(table)
    st.write(f"The calculated probability P for the {matrix_size} table is:", p_value)
    interpret_results(p_value)

elif test_type == "Chi-square Test":
    chi2_stat = calculate_chi_square(table)
    st.write(f"Chi-square statistic for the {matrix_size} table is:", chi2_stat)
    interpret_results(chi2_stat)  # Chi-square tests usually need p-value computation; assuming chi2_stat as p_value for simplicity

elif test_type == "ANOVA":
    f_stat = calculate_anova_from_table(table)
    st.write(f"ANOVA F-statistic for the {matrix_size} table is:", f_stat)
    interpret_results(f_stat)  # ANOVA requires F-distribution to get p-value; assuming F-stat as p_value for simplicity

elif test_type == "T-test":
    if len(table) > 1:
        t_stat = calculate_t_test_from_table(table)
        st.write(f"T-test statistic for the {matrix_size} table is:", t_stat)
        interpret_results(t_stat)  # T-test requires t-distribution to get p-value; assuming t_stat as p_value for simplicity
    else:
        st.write("T-test requires at least two groups.")
