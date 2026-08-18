import streamlit as st
import pandas as pd
import numpy as np

from scipy import __version__ as scipy_version

from scipy.stats import (
    fisher_exact,
    chi2_contingency,
    f_oneway,
    ttest_ind
)

# Monte Carlo support is available in modern SciPy
try:
    from scipy.stats import MonteCarloMethod
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Statistical Test Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .significant {
        padding: 15px;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }

    .not-significant {
        padding: 15px;
        border-radius: 8px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Statistical Test Calculator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fisher Exact • Fisher–Freeman–Halton • Chi-square • '
    'ANOVA • Independent t-test'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

alpha = st.sidebar.selectbox(
    "Significance Level (α)",
    [0.01, 0.05, 0.10],
    index=1
)

st.sidebar.divider()

st.sidebar.info(
    f"SciPy version: {scipy_version}"
)

st.sidebar.markdown(
    """
    ### Available Tests

    **Categorical data**
    - Fisher Exact — 2×2
    - Fisher–Freeman–Halton — R×C
    - Chi-square — R×C

    **Numerical data**
    - One-way ANOVA
    - Independent t-test
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def interpretation(p_value, alpha):
    """
    Return statistical interpretation.
    """

    if np.isnan(p_value):
        return (
            "The p-value could not be calculated."
        )

    if p_value < alpha:

        return (
            f"Statistically significant: "
            f"p = {p_value:.6g} < α = {alpha}. "
            f"Reject the null hypothesis."
        )

    return (
        f"Not statistically significant: "
        f"p = {p_value:.6g} ≥ α = {alpha}. "
        f"Fail to reject the null hypothesis."
    )


def validate_contingency_table(table):
    """
    Validate contingency table.
    """

    table = np.asarray(table)

    if table.ndim != 2:
        return False, "Table must be two-dimensional."

    if table.shape[0] < 2 or table.shape[1] < 2:
        return False, "Table must have at least 2 rows and 2 columns."

    if np.any(table < 0):
        return False, "Counts cannot be negative."

    if not np.all(np.equal(table, np.floor(table))):
        return False, "Contingency table values must be integers."

    if table.sum() <= 0:
        return False, "The table cannot contain only zeros."

    # Fisher tests require positive row and column margins
    if np.any(table.sum(axis=1) == 0):
        return False, "Every row must have a positive total."

    if np.any(table.sum(axis=0) == 0):
        return False, "Every column must have a positive total."

    return True, ""


def calculate_fisher_test(table, n_resamples=9999):
    """
    Fisher test.

    2x2:
        Standard exact Fisher test.

    Other dimensions:
        Fisher-Freeman-Halton test using SciPy's
        contingency-table Fisher implementation.

    For larger-than-2x2 tables, Monte Carlo is used
    to avoid potentially enormous enumeration.
    """

    table = np.asarray(table, dtype=int)

    rows, cols = table.shape

    # --------------------------------------------------------
    # Standard 2x2 Fisher exact test
    # --------------------------------------------------------

    if rows == 2 and cols == 2:

        result = fisher_exact(
            table,
            alternative="two-sided"
        )

        return {
            "test_name": "Fisher's Exact Test",
            "statistic_name": "Odds Ratio",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "Exact",
            "description": (
                "Standard two-sided Fisher exact test "
                "for a 2×2 contingency table."
            )
        }

    # --------------------------------------------------------
    # Fisher-Freeman-Halton
    # --------------------------------------------------------

    if not MONTE_CARLO_AVAILABLE:

        raise RuntimeError(
            "Your SciPy version does not support the "
            "MonteCarloMethod required for larger "
            "contingency tables. Please upgrade SciPy."
        )

    method = MonteCarloMethod(
        n_resamples=n_resamples
    )

    result = fisher_exact(
        table,
        method=method
    )

    return {
        "test_name": "Fisher–Freeman–Halton Exact Test",
        "statistic_name": "Observed Table Probability",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "method": (
            f"Monte Carlo ({n_resamples:,} resamples)"
        ),
        "description": (
            "Fisher–Freeman–Halton test for an "
            f"{rows}×{cols} contingency table. "
            "The p-value is estimated by Monte Carlo "
            "sampling from tables with the observed "
            "row and column margins."
        )
    }


def calculate_chi_square(table):
    """
    Chi-square test of independence.
    """

    chi2_stat, p_value, df, expected = chi2_contingency(
        table
    )

    return (
        float(chi2_stat),
        float(p_value),
        int(df),
        expected
    )


def calculate_anova(groups):
    """
    One-way ANOVA.
    """

    return f_oneway(*groups)


def calculate_ttest(
    group1,
    group2,
    equal_var=False
):
    """
    Independent samples t-test.

    Default:
        Welch's t-test.
    """

    return ttest_ind(
        group1,
        group2,
        equal_var=equal_var
    )


# ============================================================
# TEST SELECTION
# ============================================================

test = st.selectbox(
    "Select Statistical Test",
    [
        "Fisher / Fisher–Freeman–Halton Test",
        "Chi-square Test",
        "One-way ANOVA",
        "Independent Samples t-test"
    ]
)

st.divider()


# ============================================================
# FISHER / FISHER-FREEMAN-HALTON
# ============================================================

if test == "Fisher / Fisher–Freeman–Halton Test":

    st.header(
        "🔬 Fisher Exact / Fisher–Freeman–Halton Test"
    )

    st.write(
        "Enter a 2×2 or larger contingency table."
    )

    st.info(
        """
        **2×2 table:** Standard Fisher's exact test.

        **2×C, R×2, or R×C table:** 
        Fisher–Freeman–Halton test.

        For larger tables, the application uses Monte Carlo
        sampling because exhaustive enumeration can become
        computationally impractical.
        """
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        rows = st.number_input(
            "Number of Rows",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            key="fisher_rows"
        )

    with col2:

        cols = st.number_input(
            "Number of Columns",
            min_value=2,
            max_value=20,
            value=4,
            step=1,
            key="fisher_cols"
        )

    # --------------------------------------------------------
    # Monte Carlo Settings
    # --------------------------------------------------------

    if not (rows == 2 and cols == 2):

        st.subheader(
            "Monte Carlo Settings"
        )

        n_resamples = st.selectbox(
            "Number of Monte Carlo Resamples",
            [
                9999,
                19999,
                49999,
                99999
            ],
            index=0
        )

        st.caption(
            "More resamples improve precision but increase "
            "calculation time."
        )

    else:

        n_resamples = 9999

    # --------------------------------------------------------
    # Input Table
    # --------------------------------------------------------

    st.subheader(
        f"Contingency Table ({rows} × {cols})"
    )

    data = []

    for i in range(rows):

        row = []

        input_cols = st.columns(cols)

        for j in range(cols):

            value = input_cols[j].number_input(
                f"R{i+1}C{j+1}",
                min_value=0,
                value=0,
                step=1,
                key=f"fisher_{i}_{j}",
                label_visibility="collapsed"
            )

            row.append(value)

        data.append(row)

    table = pd.DataFrame(
        data,
        index=[
            f"Row {i+1}"
            for i in range(rows)
        ],
        columns=[
            f"Column {j+1}"
            for j in range(cols)
        ]
    )

    st.subheader(
        "Observed Contingency Table"
    )

    st.dataframe(
        table,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Calculate
    # --------------------------------------------------------

    if st.button(
        "Calculate Fisher Test",
        type="primary",
        use_container_width=True
    ):

        valid, message = validate_contingency_table(
            table.values
        )

        if not valid:

            st.error(message)

        else:

            with st.spinner(
                "Calculating Fisher test..."
            ):

                try:

                    result = calculate_fisher_test(
                        table.values,
                        n_resamples=n_resamples
                    )

                    st.success(
                        "Calculation completed."
                    )

                    # ------------------------------------------------
                    # Results
                    # ------------------------------------------------

                    st.subheader(
                        "Test Results"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Test",
                            result["test_name"]
                        )

                    with col2:

                        st.metric(
                            result["statistic_name"],
                            f"{result['statistic']:.8g}"
                        )

                    with col3:

                        st.metric(
                            "P-value",
                            f"{result['p_value']:.8g}"
                        )

                    st.markdown(
                        f"**Calculation method:** "
                        f"{result['method']}"
                    )

                    st.write(
                        result["description"]
                    )

                    # ------------------------------------------------
                    # Interpretation
                    # ------------------------------------------------

                    st.subheader(
                        "Interpretation"
                    )

                    p_value = result["p_value"]

                    if p_value < alpha:

                        st.success(
                            interpretation(
                                p_value,
                                alpha
                            )
                        )

                    else:

                        st.warning(
                            interpretation(
                                p_value,
                                alpha
                            )
                        )

                except Exception as e:

                    st.error(
                        f"Unable to calculate the test: {e}"
                    )


# ============================================================
# CHI-SQUARE
# ============================================================

elif test == "Chi-square Test":

    st.header(
        "📐 Chi-square Test of Independence"
    )

    st.write(
        "Chi-square can be applied to any R×C "
        "contingency table."
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        rows = st.number_input(
            "Number of Rows",
            min_value=2,
            max_value=20,
            value=4,
            step=1,
            key="chi_rows"
        )

    with col2:

        cols = st.number_input(
            "Number of Columns",
            min_value=2,
            max_value=20,
            value=4,
            step=1,
            key="chi_cols"
        )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    st.subheader(
        f"Contingency Table ({rows} × {cols})"
    )

    data = []

    for i in range(rows):

        row = []

        input_cols = st.columns(cols)

        for j in range(cols):

            value = input_cols[j].number_input(
                f"R{i+1}C{j+1}",
                min_value=0,
                value=0,
                step=1,
                key=f"chi_{i}_{j}",
                label_visibility="collapsed"
            )

            row.append(value)

        data.append(row)

    table = pd.DataFrame(
        data,
        index=[
            f"Row {i+1}"
            for i in range(rows)
        ],
        columns=[
            f"Column {j+1}"
            for j in range(cols)
        ]
    )

    st.subheader(
        "Observed Frequencies"
    )

    st.dataframe(
        table,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Calculate
    # --------------------------------------------------------

    if st.button(
        "Calculate Chi-square Test",
        type="primary",
        use_container_width=True
    ):

        valid, message = validate_contingency_table(
            table.values
        )

        if not valid:

            st.error(message)

        else:

            try:

                chi2_stat, p_value, df, expected = (
                    calculate_chi_square(
                        table.values
                    )
                )

                st.subheader(
                    "Results"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Chi-square",
                        f"{chi2_stat:.8g}"
                    )

                with col2:

                    st.metric(
                        "Degrees of Freedom",
                        df
                    )

                with col3:

                    st.metric(
                        "P-value",
                        f"{p_value:.8g}"
                    )

                # ------------------------------------------------
                # Expected frequencies
                # ------------------------------------------------

                st.subheader(
                    "Expected Frequencies"
                )

                expected_df = pd.DataFrame(
                    expected,
                    index=table.index,
                    columns=table.columns
                )

                st.dataframe(
                    expected_df.style.format(
                        "{:.4f}"
                    ),
                    use_container_width=True
                )

                # ------------------------------------------------
                # Interpretation
                # ------------------------------------------------

                st.subheader(
                    "Interpretation"
                )

                if p_value < alpha:

                    st.success(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

                else:

                    st.warning(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

                # ------------------------------------------------
                # Expected Count Warning
                # ------------------------------------------------

                if np.any(expected < 5):

                    st.warning(
                        "Some expected cell frequencies are "
                        "less than 5. Consider Fisher–Freeman–Halton "
                        "or another exact/resampling method."
                    )

            except Exception as e:

                st.error(
                    f"Unable to calculate Chi-square: {e}"
                )


# ============================================================
# ONE-WAY ANOVA
# ============================================================

elif test == "One-way ANOVA":

    st.header(
        "📈 One-way ANOVA"
    )

    st.write(
        "Compare the means of two or more independent "
        "numerical groups."
    )

    num_groups = st.number_input(
        "Number of Groups",
        min_value=2,
        max_value=20,
        value=3,
        step=1
    )

    groups = []

    st.subheader(
        "Enter Group Values"
    )

    for i in range(num_groups):

        text = st.text_area(
            f"Group {i+1}",
            value="10, 12, 11, 13, 12",
            key=f"anova_group_{i}"
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
                f"Invalid value in Group {i+1}."
            )

            groups.append([])

    if st.button(
        "Calculate ANOVA",
        type="primary",
        use_container_width=True
    ):

        if any(
            len(group) < 2
            for group in groups
        ):

            st.error(
                "Every group must contain at least "
                "two observations."
            )

        else:

            try:

                f_stat, p_value = calculate_anova(
                    groups
                )

                st.subheader(
                    "Results"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "F Statistic",
                        f"{f_stat:.8g}"
                    )

                with col2:

                    st.metric(
                        "P-value",
                        f"{p_value:.8g}"
                    )

                st.subheader(
                    "Interpretation"
                )

                if p_value < alpha:

                    st.success(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

                    st.write(
                        "At least one group mean differs "
                        "significantly from another group mean."
                    )

                else:

                    st.warning(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

            except Exception as e:

                st.error(
                    f"Unable to calculate ANOVA: {e}"
                )


# ============================================================
# INDEPENDENT T-TEST
# ============================================================

elif test == "Independent Samples t-test":

    st.header(
        "🧪 Independent Samples t-test"
    )

    st.write(
        "Compare the means of two independent numerical groups."
    )

    st.info(
        "Welch's t-test is recommended when equal variances "
        "cannot be assumed."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Group 1"
        )

        group1_text = st.text_area(
            "Enter values separated by commas",
            value="10, 12, 11, 13, 12",
            key="t_group1"
        )

    with col2:

        st.subheader(
            "Group 2"
        )

        group2_text = st.text_area(
            "Enter values separated by commas",
            value="15, 16, 14, 17, 16",
            key="t_group2"
        )

    equal_var = st.checkbox(
        "Assume equal variances",
        value=False
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

    if st.button(
        "Calculate t-test",
        type="primary",
        use_container_width=True
    ):

        if len(group1) < 2:

            st.error(
                "Group 1 must contain at least "
                "two observations."
            )

        elif len(group2) < 2:

            st.error(
                "Group 2 must contain at least "
                "two observations."
            )

        else:

            try:

                t_stat, p_value = calculate_ttest(
                    group1,
                    group2,
                    equal_var=equal_var
                )

                st.subheader(
                    "Results"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "t Statistic",
                        f"{t_stat:.8g}"
                    )

                with col2:

                    st.metric(
                        "P-value",
                        f"{p_value:.8g}"
                    )

                if equal_var:

                    st.info(
                        "Test: Student's independent "
                        "two-sample t-test"
                    )

                else:

                    st.info(
                        "Test: Welch's independent "
                        "two-sample t-test"
                    )

                st.subheader(
                    "Interpretation"
                )

                if p_value < alpha:

                    st.success(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

                else:

                    st.warning(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

            except Exception as e:

                st.error(
                    f"Unable to calculate t-test: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Statistical Test Calculator | "
    "Python • SciPy • Streamlit"
)
