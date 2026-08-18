from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import ttest_ind


# Fisher Exact Test
def fisher_exact_2x2(a, b, c, d):
    table = [[a, b],
             [c, d]]

    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    return odds_ratio, p_value


# Chi-square Test
def chi_square(table):
    chi2_stat, p_value, df, expected = chi2_contingency(table)

    return chi2_stat, p_value, df, expected


# One-way ANOVA
def anova(table):
    f_stat, p_value = f_oneway(*table)

    return f_stat, p_value


# Independent two-sample t-test
def t_test(group1, group2):
    t_stat, p_value = ttest_ind(
        group1,
        group2,
        equal_var=True
    )

    return t_stat, p_value
