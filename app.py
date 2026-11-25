import calendar
import io

import streamlit as st

from demand_platform.analytics.capability import assess_capability
from demand_platform.analytics.eda import aggregate_trend, seasonal_profile, series_profile
from demand_platform.data.ingestion import IngestionError, SUPPORTED_EXTENSIONS, load_dataset
from demand_platform.data.schema import confirm_mapping, interpret_schema
from demand_platform.data.standardization import standardize
from demand_platform.data.validation import validate


@st.cache_data(show_spinner="Reading file…")
def load(content: bytes, name: str):
    raw = load_dataset(io.BytesIO(content), name)
    return raw, interpret_schema(raw)


@st.cache_data(show_spinner="Standardising, validating and profiling…")
def analyse(content: bytes, name: str, store: str | None, product: str | None, target: str | None):
    raw, proposal = load(content, name)
    schema = confirm_mapping(proposal, store, product, target)
    data = standardize(raw, schema)
    report = validate(data, schema.frequency)
    if not report.is_valid:
        return schema, data, report, None, None
    profile = series_profile(data, report.frequency)
    eda = {"profile": profile, "trend": aggregate_trend(data), "seasonal": seasonal_profile(data, report.frequency)}
    return schema, data, report, eda, assess_capability(profile, report)


# Base frequency code -> (name, unit, season). Display only; the analytics use SEASONAL_PERIODS.
FREQUENCIES = {"W": ("Weekly", "weeks", "year"), "D": ("Daily", "days", "week"),
               "M": ("Monthly", "months", "year"), "ME": ("Monthly", "months", "year"),
               "MS": ("Monthly", "months", "year")}
WEEKDAYS = {"MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday", "THU": "Thursday",
            "FRI": "Friday", "SAT": "Saturday", "SUN": "Sunday"}


def describe_frequency(frequency: str | None) -> tuple[str, str, str | None]:
    """Plain-language (name, unit, season) for a pandas alias, e.g. "W-MON" -> ("Weekly data (dates on Mondays)", "weeks", "year")."""
    if frequency is None:
        return "Unknown frequency", "periods", None
    base, _, anchor = frequency.partition("-")
    name, unit, season = FREQUENCIES.get(base, (f"Frequency {frequency}", "periods", None))
    if anchor in WEEKDAYS:
        name += f" data (dates on {WEEKDAYS[anchor]}s)"
    elif base in FREQUENCIES:
        name += " data"
    return name, unit, season


st.set_page_config(page_title="Demand Forecasting Platform", layout="wide")
st.title("Demand Forecasting Platform")

with st.sidebar:
    uploaded = st.file_uploader("Upload dataset", type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)])
    if uploaded is None:
        st.info("Upload a sales file to start.")
        st.stop()
    content = uploaded.getvalue()
    try:
        raw, proposal = load(content, uploaded.name)
    except IngestionError as exc:
        st.error(str(exc))
        st.stop()
    if proposal.layout == "unknown":
        st.error("Could not detect a date layout (wide date headers or a long date column).")
        st.stop()

    st.subheader("Column mapping")
    st.caption(f"Layout **{proposal.layout}** · frequency **{proposal.frequency or 'unknown'}** · "
               f"{len(proposal.date_columns)} date column(s)")
    options = [None, *(c for c in raw.columns if c not in proposal.date_columns)]

    def pick(label: str, default: str | None) -> str | None:
        return st.selectbox(label, options, index=options.index(default) if default in options else 0)

    store_col = pick("Store column", proposal.id_columns.get("store_id"))
    product_col = pick("Product column", proposal.id_columns.get("product_id"))
    target_col = pick("Sales column", proposal.target_column) if proposal.layout == "long" else None

    mapping = (uploaded.file_id, store_col, product_col, target_col)
    if st.button("Confirm mapping", type="primary"):
        st.session_state["confirmed"] = mapping
    if st.session_state.get("confirmed") != mapping:
        st.info("Check the columns, then confirm the mapping to run the analysis.")
        st.stop()

try:
    schema, data, report, eda, capability = analyse(content, uploaded.name, store_col, product_col, target_col)
except ValueError as exc:
    st.sidebar.error(str(exc))
    st.stop()

blocking = {"Duplicate keys": report.duplicate_keys, "Missing dates": report.missing_dates,
            "Non-finite sales": report.non_finite_sales}
warnings = {"Missing sales": report.missing_sales, "Negative sales": report.negative_sales,
            "Non-integer sales": report.non_integer_sales, "Off-grid dates": report.off_grid_dates,
            "Missing periods": report.missing_periods, "Series with gaps": report.series_with_gaps}
freq_name, unit, season = describe_frequency(report.frequency)

if not report.is_valid:
    st.error("The data can't be analysed: " + ", ".join(f"{k.lower()} {v:,}" for k, v in blocking.items() if v)
             + ". See the validation tab below.")
else:
    st.subheader("Sales")
    st.caption("History only for now; forecasts will be drawn on this chart.")
    level = st.radio("Level", ["Store × product", "Store", "Product", "All series"], horizontal=True)
    rows, cols = data, st.columns(2)
    if level in ("Store × product", "Store"):
        store = cols[0].selectbox("Store", sorted(data["store_id"].unique()))
        rows = rows[rows["store_id"] == store]
    if level in ("Store × product", "Product"):
        # Products are listed from the chosen store only, so every pick is a real series.
        product = cols[1].selectbox("Product", sorted(rows["product_id"].unique()))
        rows = rows[rows["product_id"] == product]
    st.line_chart(rows.groupby("date")["sales"].sum(min_count=1), y_label="sales")

    c, seasonal = capability, eda["seasonal"]
    lines = [
        f"**{freq_name}**, {data['date'].min():%d %b %Y} → {data['date'].max():%d %b %Y}: "
        f"{report.n_series:,} store–product series, {report.n_rows:,} rows.",
        f"**History:** half the series have at least {c.median_periods:.0f} {unit} of sales"
        + (f"; {c.share_two_cycles:.0%} have 2 or more {season}s, enough to compare a {unit[:-1]} with "
           f"the same {unit[:-1]} one {season} earlier." if season else "."),
    ]
    if season and not seasonal.empty:
        peak = seasonal.loc[seasonal["mean_index"].idxmax()]
        position = int(peak.position)
        when = {"W": f"week {position} of the year", "D": calendar.day_name[position % 7]}.get(
            report.frequency.split("-")[0], calendar.month_name[position % 13])
        lines.append(
            f"**Seasonality:** single series barely repeat from one {season} to the next "
            f"(typical correlation {c.median_seasonal_acf:.2f}), but across all series {when} "
            f"sells {peak.mean_index:.1f}× a normal {unit[:-1]}."
        )
    lines.append(
        "**Sales values:** "
        + ("whole units, no returns" if c.count_data else "contain fractions or negative values (returns)")
        + f"; {c.zero_share:.0%} of periods have no sales, and {c.share_overdispersed:.0%} of series "
        "swing more than a simple random count would."
    )
    issues = [f"{k.lower()} {v:,}" for k, v in warnings.items() if v]
    lines.append("**Data checks:** " + ("passed." if not issues else "passed with warnings: " + ", ".join(issues) + "."))
    st.markdown("\n".join(f"- {line}" for line in lines))

with st.expander("Details", expanded=not report.is_valid):
    tabs = st.tabs(["Validation", "Raw data", "Standardised data", "Series profile", "Trend & season", "Capability"])
    with tabs[0]:
        for label, count in blocking.items():
            if count:
                st.error(f"{label}: {count:,}")
        for label, count in warnings.items():
            if count:
                st.warning(f"{label}: {count:,}")
        st.write(report)
    with tabs[1]:
        st.write(f"{raw.shape[0]:,} rows × {raw.shape[1]:,} columns")
        st.dataframe(raw.head(100))
        st.write(schema)
    with tabs[2]:
        st.write(f"{data.shape[0]:,} rows × {data.shape[1]:,} columns")
        st.dataframe(data.head(100))
    if eda is not None:
        with tabs[3]:
            st.write(eda["profile"][["n_periods", "zero_share", "dispersion", "adi", "cv2", "seasonal_acf",
                                     "trend_ratio"]].describe())
            st.dataframe(eda["profile"])
        with tabs[4]:
            st.caption("Sum of sales over all series per date; active_series = series that have sold at least once by then.")
            st.line_chart(eda["trend"].set_index("date"))
            if not eda["seasonal"].empty:
                st.caption("Mean of sales / series mean across all series, each weighted equally; 1.0 = an average period.")
                st.bar_chart(eda["seasonal"].set_index("position")["mean_index"])
        with tabs[5]:
            st.write(capability)
