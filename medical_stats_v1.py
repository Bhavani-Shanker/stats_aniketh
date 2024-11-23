import math

# Function to calculate factorial
def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

# Function to calculate binomial coefficient
def binomial_coefficient(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

# Fisher Exact Test (Manual for 2x2)
def fisher_exact_2x2(a, b, c, d):
    table_total = a + b + c + d
    p_value = (binomial_coefficient(a + b, a) * binomial_coefficient(c + d, c)) / binomial_coefficient(table_total, a + c)
    return p_value

# Chi-square Test
def chi_square(table):
    row_totals = [sum(row) for row in table]
    col_totals = [sum(col) for col in zip(*table)]
    n = sum(row_totals)

    expected = [[(row_total * col_total) / n for col_total in col_totals] for row_total in row_totals]
    chi2_stat = sum(((table[i][j] - expected[i][j]) ** 2) / expected[i][j] for i in range(len(table)) for j in range(len(table[0])))

    # Use an iterative approximation for p-value (chi-square distribution CDF)
    def chi2_cdf(x, k, steps=1000):
        """Approximate chi2 CDF using numerical integration."""
        return sum((math.exp(-t / 2) * t**(k / 2 - 1)) for t in range(steps)) / math.gamma(k / 2)

    df = (len(row_totals) - 1) * (len(col_totals) - 1)
    p_value = 1 - chi2_cdf(chi2_stat, df)
    return chi2_stat, p_value

# ANOVA
def anova(table):
    flattened_table = [item for row in table for item in row]
    overall_mean = sum(flattened_table) / len(flattened_table)

    group_means = [sum(row) / len(row) for row in table]
    ss_between = sum(len(row) * (group_mean - overall_mean) ** 2 for row, group_mean in zip(table, group_means))
    ss_within = sum(sum((x - group_mean) ** 2 for x in row) for row, group_mean in zip(table, group_means))

    df_between = len(table) - 1
    df_within = len(flattened_table) - len(table)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    # Approximate F-distribution p-value using numerical integration
    def f_cdf(x, d1, d2, steps=1000):
        """Approximate F-distribution CDF using numerical integration."""
        return sum((t ** (d1 / 2 - 1) * (1 + t / d2) ** -(d1 + d2) / 2) for t in range(steps)) / math.gamma(d1 / 2)

    p_value = 1 - f_cdf(f_stat, df_between, df_within)
    return f_stat, p_value

# T-test
def t_test(group1, group2):
    mean1, mean2 = sum(group1) / len(group1), sum(group2) / len(group2)
    var1 = sum((x - mean1) ** 2 for x in group1) / (len(group1) - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (len(group2) - 1)

    pooled_var = ((len(group1) - 1) * var1 + (len(group2) - 1) * var2) / (len(group1) + len(group2) - 2)
    t_stat = (mean1 - mean2) / math.sqrt(pooled_var * (1 / len(group1) + 1 / len(group2)))

    # Approximate T-distribution p-value using numerical integration
    def t_cdf(x, df, steps=1000):
        """Approximate T-distribution CDF using numerical integration."""
        return sum((1 + t**2 / df) ** -(df + 1) / 2 for t in range(steps)) / math.gamma((df + 1) / 2)

    p_value = 2 * (1 - t_cdf(t_stat, len(group1) + len(group2) - 2))
    return t_stat, p_value
