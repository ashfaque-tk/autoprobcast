import streamlit as st

from demand_platform.data.ingestion import IngestionError, SUPPORTED_EXTENSIONS, load_dataset
from demand_platform.data.schema import confirm_mapping, interpret_schema
from demand_platform.data.standardization import standardize
from demand_platform.data.validation import validate

st.title("Demand Forecasting Platform")

uploaded = st.file_uploader(
    "Upload dataset", type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]
)

if uploaded is None:
    st.stop()

if st.session_state.get("file_id") != uploaded.file_id:
    st.session_state["file_id"] = uploaded.file_id
    st.session_state.pop("confirmed_schema", None)
    st.session_state.pop("standardized", None)
    st.session_state.pop("validation", None)

try:
    df = load_dataset(uploaded, uploaded.name)
except IngestionError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")
st.dataframe(df.head(100))

st.header("Schema mapping")
proposal = interpret_schema(df)
if proposal.layout == "unknown":
    st.error("Could not detect a date layout (wide date headers or a long date column).")
    st.stop()

st.write(
    f"Layout: **{proposal.layout}** · Frequency: **{proposal.frequency or 'unknown'}** · "
    f"{len(proposal.date_columns)} date column(s)"
)
if proposal.layout == "wide":
    st.write(f"Dates: {proposal.date_columns[0]} → {proposal.date_columns[-1]}")

options = [None, *(c for c in df.columns if c not in proposal.date_columns)]


def pick(label: str, default: str | None) -> str | None:
    return st.selectbox(label, options, index=options.index(default) if default in options else 0)


store_col = pick("Store column", proposal.id_columns.get("store_id"))
product_col = pick("Product column", proposal.id_columns.get("product_id"))
target_col = pick("Sales column", proposal.target_column) if proposal.layout == "long" else None

if st.button("Confirm mapping"):
    try:
        st.session_state["confirmed_schema"] = confirm_mapping(proposal, store_col, product_col, target_col)
        st.session_state["standardized"] = standardize(df, st.session_state["confirmed_schema"])
        st.session_state.pop("validation", None)
    except ValueError as exc:
        st.error(str(exc))

if "confirmed_schema" in st.session_state:
    st.success("Mapping confirmed.")
    st.write(st.session_state["confirmed_schema"])

if "standardized" in st.session_state:
    st.header("Standardised data")
    standardized = st.session_state["standardized"]
    st.write(f"{standardized.shape[0]:,} rows × {standardized.shape[1]:,} columns")
    st.dataframe(standardized.head(100))

    st.header("Validation")
    if st.button("Validate"):
        st.session_state["validation"] = validate(standardized, st.session_state["confirmed_schema"].frequency)

if "validation" in st.session_state:
    report = st.session_state["validation"]
    blocking = {"Duplicate keys": report.duplicate_keys, "Missing dates": report.missing_dates,
                "Non-finite sales": report.non_finite_sales}
    warnings = {"Missing sales": report.missing_sales, "Negative sales": report.negative_sales,
                "Non-integer sales": report.non_integer_sales, "Off-grid dates": report.off_grid_dates,
                "Missing periods": report.missing_periods, "Series with gaps": report.series_with_gaps}
    for label, count in blocking.items():
        if count:
            st.error(f"{label}: {count:,}")
    for label, count in warnings.items():
        if count:
            st.warning(f"{label}: {count:,}")
    if report.frequency is None:
        st.warning("Frequency unknown: calendar checks skipped.")
    if report.is_valid:
        st.success(f"Valid: {report.n_rows:,} rows, {report.n_series:,} series.")
    st.write(report)
