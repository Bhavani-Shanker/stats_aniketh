import streamlit as st
import pandas as pd
from scipy.stats import (
    fisher_exact,
    chi2_contingency,
    f_oneway,
    ttest_ind
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Statistical Test Calculator",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Statistical Test Calculator")
st.markdown(
    "Perform **Fisher Exact, Chi-square, One-way ANOVA, "
    "and Independent Samples t-tests** using SciPy."
)

st.divider()


# ---------------------------------------------------------
# Fisher Exact Test
# ---------------------------------------------------------

def fisher_exact_2x2(a, b, c, d):
    table = [
        [a, b],
        [c, d]
    ]

    odds_ratio, p_value = fisher_exact(
        table,
        alternative="two-sided"
    )

    return odds_ratio, p_value


# ---------------------------------------------------------
# Chi-square Test
# ---------------------------------------------------------

def chi_square(table):
    chi2_stat, p_value, df, expected = chi2_contingency(table)

    return chi2_stat, p_value, df, expected


# ---------------------------------------------------------
# One-way ANOVA
# ---------------------------------------------------------

def anova(groups):
    f_stat, p_value = f_oneway(*groups)

    return f_stat, p_value


# ---------------------------------------------------------
# Independent Samples t-test
# ---------------------------------------------------------

def t_test(group1, group2, equal_var=True):
    t_stat, p_value = ttest_ind(
        group1,
        group2,
        equal_var=equal_var
    )

    return t_stat, p_value


# ---------------------------------------------------------
# Helper Function
# ---------------------------------------------------------

def interpretation(p_value, alpha):
    if p_value < alpha:
        return (
            f"**Significant result:** p-value ({p_value:.6f}) "
            f"< α ({alpha}). Reject the null hypothesis."
        )
    else:
        return (
            f"**Not significant:** p-value ({p_value:.6f}) "
            f"≥ α ({alpha}). Fail to reject the null hypothesis."
        )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("⚙️ Settings")

alpha = st.sidebar.selectbox(
    "Significance Level (α)",
    [0.01, 0.05, 0.10],
    index=1
)

st.sidebar.info(
    "A p-value below α indicates a statistically significant result."
)


# ---------------------------------------------------------
# Test Selection
# ---------------------------------------------------------

test = st.selectbox(
    "Select Statistical Test",
    [
        "Fisher Exact Test",
        "Chi-square Test",
        "One-way ANOVA",
        "Independent Samples t-test"
    ]
)

st.divider()


# =========================================================
# FISHER EXACT TEST
# =========================================================

if test == "Fisher Exact Test":

    st.header("🔬 Fisher Exact Test")

    st.write(
        "Used to determine whether there is a significant "
        "association between two categorical variables in a 2×2 table."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Group 1")

        a = st.number_input(
            "Group 1 — Success",
            min_value=0,
            value=3,
            step=1
        )

        b = st.number_input(
            "Group 1 — Failure",
            min_value=0,
            value=13,
            step=1
        )

    with col2:
        st.subheader("Group 2")

        c = st.number_input(
            "Group 2 — Success",
            min_value=0,
            value=2,
            step=1
        )

        d = st.number_input(
            "Group 2 — Failure",
            min_value=0,
            value=9,
            step=1
        )

    table = pd.DataFrame(
        [[a, b], [c, d]],
        columns=["Success", "Failure"],
        index=["Group 1", "Group 2"]
    )

    st.subheader("2 × 2 Contingency Table")

    st.dataframe(
        table,
        use_container_width=True
    )

    if st.button("Calculate Fisher Exact Test", type="primary"):

        odds_ratio, p_value = fisher_exact_2x2(
            a, b, c, d
        )

        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Odds Ratio",
                f"{odds_ratio:.6f}"
            )

        with col2:
            st.metric(
                "P-value",
                f"{p_value:.6f}"
            )

        st.markdown(
            interpretation(p_value, alpha)
        )


# =========================================================
# CHI-SQUARE TEST
# =========================================================

elif test == "Chi-square Test":

    st.header("📐 Chi-square Test")

    st.write(
        "Tests whether there is a statistically significant "
        "association between categorical variables."
    )

    rows = st.number_input(
        "Number of Rows",
        min_value=2,
        max_value=10,
        value=2,
        step=1
    )

    cols = st.number_input(
        "Number of Columns",
        min_value=2,
        max_value=10,
        value=2,
        step=1
    )

    st.subheader("Enter Observed Frequencies")

    data = []

    for i in range(rows):

        row = []

        columns = st.columns(cols)

        for j, column in enumerate(columns):

            value = column.number_input(
                f"R{i+1} C{j+1}",
                min_value=0,
                value=1,
                step=1,
                key=f"chi_{i}_{j}"
            )

            row.append(value)

        data.append(row)

    table = pd.DataFrame(
        data,
        index=[f"Row {i+1}" for i in range(rows)],
        columns=[f"Column {j+1}" for j in range(cols)]
    )

    st.dataframe(
        table,
        use_container_width=True
    )

    if st.button("Calculate Chi-square Test", type="primary"):

        chi2_stat, p_value, df, expected = chi_square(
            data
        )

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Chi-square Statistic",
                f"{chi2_stat:.6f}"
            )

        with col2:
            st.metric(
                "Degrees of Freedom",
                f"{df}"
            )

        with col3:
            st.metric(
                "P-value",
                f"{p_value:.6f}"
            )

        st.subheader("Expected Frequencies")

        expected_df = pd.DataFrame(
            expected,
            index=table.index,
            columns=table.columns
        )

        st.dataframe(
            expected_df.style.format("{:.4f}"),
            use_container_width=True
        )

        st.markdown(
            interpretation(p_value, alpha)
        )


# =========================================================
# ONE-WAY ANOVA
# =========================================================

elif test == "One-way ANOVA":

    st.header("📈 One-way ANOVA")

    st.write(
        "Tests whether the means of three or more independent "
        "groups are statistically different."
    )

    num_groups = st.number_input(
        "Number of Groups",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

    st.subheader("Enter Group Data")

    groups = []

    for i in range(num_groups):

        text = st.text_area(
            f"Group {i+1}",
            value="10, 12, 11, 13, 12",
            key=f"group_{i}"
        )

        try:
            values = [
                float(x.strip())
                for x in text.split(",")
                if x.strip()
            ]

            groups.append(values)

        except ValueError:

            st.error(
                f"Invalid numeric value in Group {i+1}."
            )
            groups.append([])

    if st.button("Calculate ANOVA", type="primary"):

        if any(len(group) < 2 for group in groups):

            st.error(
                "Each group must contain at least two observations."
            )

        else:

            f_stat, p_value = anova(groups)

            st.subheader("Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "F Statistic",
                    f"{f_stat:.6f}"
                )

            with col2:
                st.metric(
                    "P-value",
                    f"{p_value:.6f}"
                )

            st.markdown(
                interpretation(p_value, alpha)
            )


# =========================================================
# INDEPENDENT t-TEST
# =========================================================

elif test == "Independent Samples t-test":

    st.header("🧪 Independent Samples t-test")

    st.write(
        "Tests whether the means of two independent groups "
        "are statistically different."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Group 1")

        group1_text = st.text_area(
            "Enter values separated by commas",
            value="10, 12, 11, 13, 12",
            key="t_group1"
        )

    with col2:

        st.subheader("Group 2")

        group2_text = st.text_area(
            "Enter values separated by commas",
            value="15, 16, 14, 17, 16",
            key="t_group2"
        )

    equal_var = st.checkbox(
        "Assume equal variances",
        value=True
    )

    try:

        group1 = [
            float(x.strip())
            for x in group1_text.split(",")
            if x.strip()
        ]

        group2 = [
            float(x.strip())
            for x in group2_text.split(",")
            if x.strip()
        ]

    except ValueError:

        group1 = []
        group2 = []

        st.error(
            "Please enter valid numeric values."
        )

    if st.button("Calculate t-test", type="primary"):

        if len(group1) < 2 or len(group2) < 2:

            st.error(
                "Each group must contain at least two observations."
            )

        else:

            t_stat, p_value = t_test(
                group1,
                group2,
                equal_var=equal_var
            )

            st.subheader("Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "t Statistic",
                    f"{t_stat:.6f}"
                )

            with col2:
                st.metric(
                    "P-value",
                    f"{p_value:.6f}"
                )

            test_type = (
                "Student's t-test"
                if equal_var
                else "Welch's t-test"
            )

            st.info(
                f"Test used: **{test_type}**"
            )

            st.markdown(
                interpretation(p_value, alpha)
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Statistical Test Calculator • Python + SciPy + Streamlit"
)
