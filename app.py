import io
import pandas as pd
import pyreadstat
import streamlit as st
import weightipy as wp

st.set_page_config(
    page_title="Automated Survey Weighting Engine (Stats SA)",
    page_icon="🇿🇦",
    layout="wide",
)

st.title("🇿🇦 Automated Survey Rim Weighting App")
st.markdown(
    "Upload your SPSS `.sav` file, select your variables, and let the app"
    " automatically pull official **Stats SA** benchmarks to weight your"
    " sample."
)

# --- OFFICIAL STATS SA BENCHMARK DATA (Mock/Cached from latest P0302 / Census) ---
# These represent official national marginal proportions that can be updated annually.
STATS_SA_BENCHMARKS = {
    "province": {
        "Gauteng": 25.5,
        "KwaZulu-Natal": 19.4,
        "Western Cape": 11.9,
        "Eastern Cape": 11.4,
        "Limpopo": 10.2,
        "Mpumalanga": 8.0,
        "North West": 6.6,
        "Free State": 4.8,
        "Northern Cape": 2.2,
    },
    "gender": {"Male": 49.0, "Female": 51.0},
}

# --- SIDEBAR: FILE UPLOADER ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose an SPSS .sav file", type=["sav"]
)

if uploaded_file is not None:
  try:
    with open("temp_uploaded.sav", "wb") as f:
      f.write(uploaded_file.getbuffer())
    df, meta = pyreadstat.read_sav("temp_uploaded.sav")
    st.sidebar.success(
        f"Loaded successfully! Rows: {len(df):,}, Columns: {len(df.columns)}"
    )
  except Exception as e:
    st.sidebar.error(f"Error reading file: {e}")
    st.stop()

  st.header("2. Configure Targets & Stats SA Automation")

  cat_columns = df.select_dtypes(
      include=["object", "category", "int64", "float64"]
  ).columns.tolist()

  selected_vars = st.multiselect(
      "Choose columns in your dataset to weight:", options=cat_columns
  )

  # Automation Toggle
  use_stats_sa = st.checkbox(
      "🤖 Automatically apply official Stats SA benchmark targets where"
      " available",
      value=True,
  )

  targets = {}

  if selected_vars:
    for var in selected_vars:
      with st.expander(f"Targets for variable: **{var}**", expanded=True):
        unique_vals = df[var].dropna().unique().tolist()
        try:
          unique_vals = sorted(unique_vals)
        except Exception:
          pass

        var_targets = {}
        total_pct = 0.0

        # Check if we have automated Stats SA data matching this variable name
        matched_key = None
        for key in STATS_SA_BENCHMARKS.keys():
          if key in var.lower():
            matched_key = key
            break

        if use_stats_sa and matched_key:
          st.info(
              f"✨ Automatically matched `{var}` with official Stats SA"
              f" benchmark data for `{matched_key}`."
          )

        for val in unique_vals:
          sample_count = (df[var] == val).sum()
          sample_pct = (sample_count / len(df)) * 100
          val_str = str(val)

          # Default value logic: Check Stats SA map if available, otherwise use sample share
          default_val = float(round(sample_pct, 1))
          if use_stats_sa and matched_key:
            # Look for exact or fuzzy match in Stats SA dict
            for sa_name, sa_val in STATS_SA_BENCHMARKS[matched_key].items():
              if sa_name.lower() in val_str.lower():
                default_val = sa_val
                break

          col_a, col_b = st.columns([2, 1])
          with col_a:
            pct = st.number_input(
                f"Target % for `{val_str}`",
                min_value=0.0,
                max_value=100.0,
                value=default_val,
                step=0.5,
                key=f"target_{var}_{val_str}",
            )
            var_targets[val] = pct
            total_pct += pct
          with col_b:
            st.metric(label="Sample Count", value=f"{sample_count:,}")

        targets[var] = var_targets

        if abs(total_pct - 100.0) > 0.01:
          st.warning(f"⚠️ Total for {var} sums to {total_pct:.1f}% (Needs 100%)")
        else:
          st.success(f"✓ {var} targets sum to 100%")

    # --- EXECUTION ---
    st.markdown("---")
    st.header("3. Execute Weighting")
    weight_col_name = st.text_input(
        "Name for output weight column:", value="stats_sa_weight"
    )

    if st.button("🚀 Run Automated Weighting", type="primary"):
      valid = True
      for var, t_dict in targets.items():
        if abs(sum(t_dict.values()) - 100.0) > 0.01:
        valid = False
          st.error(
              f"Fix target totals for '{var}' (currently"
              f" {sum(t_dict.values()):.1f}%)"
          )

      if valid:
        try:
          with st.spinner("Calculating rim weights against benchmarks..."):
            scheme = wp.scheme_from_dict(targets)
            df_weighted = wp.weight_dataframe(
                df, scheme, weight_column=weight_col_name
            )
            efficiency = wp.weighting_efficiency(df_weighted[weight_col_name])

          st.success("🎉 Weighting completed successfully!")

          # Diagnostics
          st.subheader("📈 Quality Metrics")
          col1, col2, col3 = st.columns(3)
          col1.metric("Weighting Efficiency (Kish)", f"{efficiency:.1f}%")
          col2.metric(
              "Mean Weight", f"{df_weighted[weight_col_name].mean():.2f}"
          )
          col3.metric("Max Weight", f"{df_weighted[weight_col_name].max():.2f}")

          # Comparison view
          st.subheader("🔍 Distribution Comparison")
          for var in selected_vars:
            unweighted = (
                df[var].value_counts(normalize=True) * 100
            ).rename("Unweighted %")
            weighted = (
                df_weighted.groupby(var, observed=True)[weight_col_name]
                .sum()
                / df_weighted[weight_col_name].sum()
                * 100
            ).rename("Weighted %")

            comp_df = pd.DataFrame(
                {
                    "Unweighted %": unweighted,
                    "Stats SA Target %": pd.Series(targets[var]),
                    "Weighted %": weighted,
                }
            ).round(2)
            st.dataframe(comp_df, use_container_width=True)

          # Download
          out_file = "stats_sa_weighted_data.sav"
          pyreadstat.write_sav(df_weighted, out_file, compress=True)
          with open(out_file, "rb") as f:
            st.download_button(
                "📥 Download Weighted .sav File",
                f,
                file_name=out_file,
                mime="application/octet-stream",
            )
        except Exception as ex:
          st.error(f"Error during execution: {ex}")
  else:
    st.info("👈 Select columns above to begin.")
else:
  st.info("👋 Upload an SPSS `.sav` file to get started.")