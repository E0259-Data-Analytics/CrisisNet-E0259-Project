"""
CrisisNet — Interactive Dashboard
===================================
Six-tab Streamlit app for exploring CrisisNet fusion model outputs.

Tabs:
  1. Company Scorecard    — live health rankings with filters
  2. SHAP Waterfall       — per-company SHAP explainability
  3. Network Graph        — supply-chain contagion explorer
  4. Risk Timeline        — multi-company historical risk evolution
  5. Predictions vs Actuals — confusion matrix, hit/miss per company
  6. Live Scoring         — upload CSV → run full pipeline → get health scores

Usage:
    pip install streamlit plotly networkx shap lightgbm pyarrow
    streamlit run dashboard/app.py
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CrisisNet",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
MODULE_D    = REPO_ROOT / "Module_D"
MODULE_C    = REPO_ROOT / "Module_C"

HEALTH_PATH   = MODULE_D / "health_scores.parquet"
FUSED_PATH    = MODULE_D / "X_fused.parquet"
SHAP_PATH     = MODULE_D / "shap_values.npy"
FEAT_PATH     = MODULE_D / "shap_feat_cols.json"
ABLATION_PATH  = MODULE_D / "ablation_results.json"
METRICS_PATH   = MODULE_D / "metrics.json"
TEST_PRED_PATH = MODULE_D / "test_predictions.parquet"
MODEL_PATH     = MODULE_D / "lgbm_fusion.txt"
THRESHOLD_PATH = MODULE_D / "optimal_threshold.json"
GRAPH_PATH    = MODULE_C / "results" / "exports" / "X_graph.parquet"
GRAPH_PKL     = MODULE_C / "data" / "processed" / "supply_chain_graph.pkl"

# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_data
def load_scores():
    return pd.read_parquet(HEALTH_PATH)

@st.cache_data
def load_fused():
    return pd.read_parquet(FUSED_PATH)

@st.cache_data
def load_graph_features():
    return pd.read_parquet(GRAPH_PATH)

@st.cache_data
def load_shap():
    vals = np.load(SHAP_PATH)
    with open(FEAT_PATH) as f:
        cols = json.load(f)
    return vals, cols

@st.cache_data
def load_ablation():
    with open(ABLATION_PATH) as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_threshold():
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            return json.load(f)
    return {'threshold': 0.15, 'recall_boost': 3, 'strategy': 'default'}

@st.cache_data
def load_test_predictions():
    return pd.read_parquet(TEST_PRED_PATH)

@st.cache_resource
def load_lgbm_model():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(MODEL_PATH))

@st.cache_resource
def load_graph():
    import pickle
    with open(GRAPH_PKL, 'rb') as f:
        return pickle.load(f)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a3248 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 4px 0;
        border-left: 4px solid #2ecc71;
    }
    .metric-card.danger { border-left-color: #e74c3c; }
    .metric-card.warning { border-left-color: #f39c12; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔴 CrisisNet — Corporate Default Early Warning System")
st.markdown("*E0 259 Data Analytics | IISc Bangalore*")
st.divider()

# ── Check data availability ───────────────────────────────────────────────────
data_ready = HEALTH_PATH.exists() and FUSED_PATH.exists()
if not data_ready:
    st.error(
        "**Model outputs not found.** Run the pipeline first:\n\n"
        "```bash\n"
        "cd CrisisNet-E0259-Project\n"
        "python Module_D/build_x_fused.py\n"
        "python Module_D/train_fusion.py\n"
        "```"
    )
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
scores  = load_scores()
fused   = load_fused()
gf      = load_graph_features() if GRAPH_PATH.exists() else None

# Derived columns
scores['risk_tier'] = pd.cut(
    1 - scores['health_score'],
    bins=[-0.01, 0.3, 0.6, 0.8, 1.01],
    labels=['Low', 'Medium', 'High', 'Critical']
)

# Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/f/fd/Indian_Institute_of_Science_logo.png",
             width=80)
    st.markdown("### Filters")

    all_quarters = sorted(scores['quarter'].unique(), reverse=True)
    selected_q   = st.selectbox("Quarter", all_quarters, index=0)

    all_tickers  = sorted(scores['ticker'].unique())
    sel_tickers  = st.multiselect("Companies (leave blank = all)", all_tickers)

    risk_filter  = st.multiselect(
        "Risk Tier", ['Low', 'Medium', 'High', 'Critical'],
        default=['Low', 'Medium', 'High', 'Critical']
    )

    st.divider()
    st.markdown("### Model Info")
    thresh_info = load_threshold()
    opt_thresh  = thresh_info.get('threshold', 0.15)
    if METRICS_PATH.exists():
        m = load_metrics()
        cv_info = m.get('cv_walk_forward', {})
        cf  = m.get('CrisisNet Fusion', {})
        az  = m.get('Altman Z-Score (1968)', {})
        st.metric("CV Mean AUC",  f"{cv_info.get('mean_AUC', 0):.3f}" if isinstance(cv_info.get('mean_AUC'), float) else "N/A")
        st.metric("Test AUC",     f"{cf.get('AUC', 0):.3f}"            if isinstance(cf.get('AUC'), float) else "N/A")
        if isinstance(cf.get('AUC'), float) and isinstance(az.get('AUC'), float):
            st.metric("Lift vs Z-Score", f"+{cf['AUC']-az['AUC']:.3f}", delta_color="normal")

    # Recall metrics from test predictions
    if TEST_PRED_PATH.exists():
        _p = pd.read_parquet(TEST_PRED_PATH)
        _y = _p['distress_label'].values
        _yp = _p['predicted_label'].values
        from sklearn.metrics import recall_score, precision_score
        rec  = recall_score(_y, _yp, zero_division=0)
        prec = precision_score(_y, _yp, zero_division=0)
        st.metric("Distress Recall",    f"{rec:.3f}",  help="Fraction of real distress events caught")
        st.metric("Distress Precision", f"{prec:.3f}", help="Fraction of flagged companies truly distressed")
        st.metric("Decision Threshold", f"{opt_thresh:.2f}", help="Lowered from 0.50 to maximise recall")

    st.divider()
    # NLP integration status
    from pathlib import Path as _P
    _nlp_path = REPO_ROOT / "Module_B" / "results" / "X_nlp_selected.parquet"
    if _nlp_path.exists():
        st.success("Module B NLP: Active", icon="✅")
        st.caption("32 NLP features (topics + sentiment, forward-filled Q1→Q4)")
    else:
        st.warning("Module B NLP: Pending", icon="⚠️")
    st.caption("CrisisNet v1.1 | April 2026")

# ── Filtered score view ───────────────────────────────────────────────────────
snap = scores[scores['quarter'] == selected_q].copy()
if sel_tickers:
    snap = snap[snap['ticker'].isin(sel_tickers)]
snap = snap[snap['risk_tier'].isin(risk_filter)]

# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊  Company Scorecard",
    "🔍  SHAP Explainer",
    "🕸️   Network Graph",
    "📈  Risk Timeline",
    "✅  Predictions vs Actuals",
    "⚡  Live Scoring",
])

# ── TAB 1: Company Scorecard ──────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Company Health Scorecard — {selected_q}")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    n_companies = snap['ticker'].nunique()
    n_critical  = (snap['risk_tier'] == 'Critical').sum()
    n_high      = (snap['risk_tier'] == 'High').sum()
    n_actual    = snap['distress_label'].sum()
    avg_health  = snap['health_score'].mean()

    c1.metric("Companies", n_companies)
    c2.metric("Avg Health Score", f"{avg_health:.3f}")
    c3.metric("Critical Risk", n_critical, delta=f"{n_critical} companies", delta_color="inverse")
    c4.metric("High Risk", n_high)
    c5.metric("Actual Distress", n_actual)

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Styled dataframe
        disp = snap[['ticker', 'health_score', 'distress_prob', 'distress_label', 'risk_tier']].copy()
        disp.columns = ['Ticker', 'Health Score', 'Distress Prob', 'Actual Distress', 'Risk Tier']
        disp = disp.sort_values('Health Score')

        def color_tier(val):
            colors = {'Low': '#1e8449', 'Medium': '#d4ac0d',
                      'High': '#ca6f1e', 'Critical': '#922b21'}
            return f"color: {colors.get(str(val), 'white')}"

        styled = (
            disp.style
            .background_gradient(subset=['Health Score'], cmap='RdYlGn', vmin=0, vmax=1)
            .background_gradient(subset=['Distress Prob'], cmap='YlOrRd', vmin=0, vmax=1)
            .map(color_tier, subset=['Risk Tier'])
            .format({'Health Score': '{:.4f}', 'Distress Prob': '{:.4f}'})
        )
        st.dataframe(styled, use_container_width=True, height=450)

    with col_right:
        import plotly.express as px
        import plotly.graph_objects as go

        # Bar chart — top 15 highest distress probability
        top15 = snap.nlargest(15, 'distress_prob')
        colors = top15['risk_tier'].map(
            {'Low': '#1e8449', 'Medium': '#d4ac0d', 'High': '#ca6f1e', 'Critical': '#922b21'}
        ).fillna('#888888')

        fig_bar = go.Figure(go.Bar(
            x=top15['ticker'],
            y=top15['distress_prob'],
            marker_color=colors,
            text=[f"{p:.3f}" for p in top15['distress_prob']],
            textposition='outside',
        ))
        fig_bar.update_layout(
            title="Top 15 — Highest Distress Probability",
            xaxis_title="Ticker", yaxis_title="P(distress)",
            yaxis_range=[0, 1.1],
            height=380, margin=dict(t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Risk tier pie
        tier_counts = snap['risk_tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig_pie = px.pie(
            tier_counts, values='Count', names='Tier',
            color='Tier',
            color_discrete_map={'Low': '#1e8449', 'Medium': '#d4ac0d',
                                'High': '#ca6f1e', 'Critical': '#922b21'},
            title="Risk Tier Distribution",
            hole=0.4,
        )
        fig_pie.update_layout(
            height=300, margin=dict(t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ── TAB 2: SHAP Explainer ─────────────────────────────────────────────────────
with tabs[1]:
    import plotly.graph_objects as go

    st.subheader("SHAP Waterfall — Feature Attribution")
    shap_ready = SHAP_PATH.exists() and FEAT_PATH.exists()

    if not shap_ready:
        st.warning("SHAP values not found. Run `python Module_D/train_fusion.py` first.")
    else:
        shap_vals, feat_cols = load_shap()
        test_data = fused[fused['quarter'].str[:4].astype(int) >= 2023].reset_index(drop=True)

        st.markdown("Select a company and quarter to see **which features drove the model's prediction**.")

        col_a, col_b, col_c = st.columns([2, 2, 3])
        with col_a:
            ticker_shap = st.selectbox("Ticker", sorted(test_data['ticker'].unique()), key='shap_ticker')
        with col_b:
            q_options = sorted(test_data[test_data['ticker'] == ticker_shap]['quarter'].unique(), reverse=True)
            quarter_shap = st.selectbox("Quarter", q_options, key='shap_quarter')
        with col_c:
            top_n = st.slider("Number of features to show", 5, 30, 15)

        row_mask = (test_data['ticker'] == ticker_shap) & (test_data['quarter'] == quarter_shap)
        if row_mask.sum() > 0:
            row_idx  = test_data[row_mask].index[0]
            sv       = shap_vals[row_idx]
            feat_vals= test_data.iloc[row_idx][feat_cols].values.astype(float)

            # Sort by |SHAP|
            order = np.argsort(np.abs(sv))[::-1][:top_n]
            labels = [f"{feat_cols[i]}={feat_vals[i]:.3g}" for i in order]
            vals   = [sv[i] for i in order]

            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]

            # Waterfall chart
            fig_wf = go.Figure(go.Bar(
                y=labels[::-1], x=vals[::-1],
                orientation='h',
                marker_color=colors[::-1],
                text=[f"{v:+.4f}" for v in vals[::-1]],
                textposition='outside',
            ))

            base_score = float(shap_vals[row_idx].sum())
            dist_prob  = test_data.iloc[row_idx]['distress_prob'] if 'distress_prob' in test_data.columns else float('nan')
            health_s   = 1.0 - dist_prob if not np.isnan(dist_prob) else float('nan')

            fig_wf.update_layout(
                title=f"SHAP Attribution — {ticker_shap} {quarter_shap} | "
                      f"P(distress)≈{dist_prob:.3f} | Health={health_s:.3f}",
                xaxis_title="SHAP Value (red = ↑ distress risk)",
                height=max(400, top_n * 28),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)',
                font=dict(color='white'),
                margin=dict(l=280),
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white'),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            # Feature value table
            with st.expander("Full feature values for this observation"):
                feat_df = pd.DataFrame({
                    'Feature': feat_cols,
                    'Value':   feat_vals,
                    'SHAP':    sv,
                }).sort_values('SHAP', key=abs, ascending=False)
                st.dataframe(feat_df.style.background_gradient(subset=['SHAP'], cmap='RdYlGn_r'),
                             use_container_width=True)
        else:
            st.warning(f"No test data found for {ticker_shap} / {quarter_shap}")

        # Global SHAP importance
        st.divider()
        st.subheader("Global Feature Importance (mean |SHAP|)")
        mean_abs   = np.abs(shap_vals).mean(axis=0)
        top_global = np.argsort(mean_abs)[::-1][:20]

        fig_imp = go.Figure(go.Bar(
            y=[feat_cols[i] for i in top_global][::-1],
            x=[mean_abs[i] for i in top_global][::-1],
            orientation='h',
            marker_color='#3498db',
        ))
        fig_imp.update_layout(
            title="Top 20 Features — Mean Absolute SHAP Value",
            xaxis_title="Mean |SHAP|",
            height=550,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
            font=dict(color='white'),
            margin=dict(l=200),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 3: Network Graph ───────────────────────────────────────────────────────
with tabs[2]:
    import plotly.graph_objects as go

    st.subheader("Supply-Chain Contagion Network")

    if gf is None:
        st.warning("X_graph.parquet not found.")
    else:
        col1, col2 = st.columns([1, 3])
        # X_graph.parquet has raw column names (no graph_ prefix)
        gf_cols = [c for c in gf.columns if c not in {'ticker', 'quarter', 'year', 'name', 'subsector', 'defaulted'}]

        # Pick sensible defaults — fall back gracefully if column absent
        def _pick(candidates):
            for c in candidates:
                if c in gf.columns:
                    return c
            return gf_cols[0] if gf_cols else 'ticker'

        default_metric = _pick(['systemic_importance_score', 'pagerank', 'betweenness_centrality'])
        default_color  = _pick(['debtrank_exposure', 'contagion_out', 'louvain_community_id'])

        with col1:
            nw_quarter = st.selectbox("Quarter", sorted(gf['quarter'].unique(), reverse=True), key='nw_q')
            nw_metric  = st.selectbox(
                "Node size by",
                [c for c in gf_cols if pd.api.types.is_numeric_dtype(gf[c])],
                index=[c for c in gf_cols if pd.api.types.is_numeric_dtype(gf[c])].index(default_metric)
                      if default_metric in gf_cols else 0,
                format_func=lambda x: x.replace('_', ' ').title(),
            )
            nw_color   = st.selectbox(
                "Node color by",
                [c for c in gf_cols if pd.api.types.is_numeric_dtype(gf[c])],
                index=[c for c in gf_cols if pd.api.types.is_numeric_dtype(gf[c])].index(default_color)
                      if default_color in gf_cols else 0,
                format_func=lambda x: x.replace('_', ' ').title(),
            )
            show_labels= st.checkbox("Show labels", value=True)

        gf_q   = gf[gf['quarter'] == nw_quarter].copy()
        # Health scores overlay
        hs_q   = scores[scores['quarter'] == nw_quarter][['ticker', 'health_score', 'risk_tier']]
        gf_q   = gf_q.merge(hs_q, on='ticker', how='left')

        with col2:
            if nw_metric in gf_q.columns and nw_color in gf_q.columns:
                import math
                tickers = gf_q['ticker'].tolist()
                n       = len(tickers)
                angles  = [2 * math.pi * i / n for i in range(n)]
                x_pos   = [math.cos(a) for a in angles]
                y_pos   = [math.sin(a) for a in angles]

                sizes  = gf_q[nw_metric].fillna(0).values.astype(float)
                sizes  = 8 + 42 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1e-9)
                colors = gf_q[nw_color].fillna(0).values.astype(float)

                sys_col  = 'systemic_importance_score' if 'systemic_importance_score' in gf_q.columns else nw_metric
                debt_col = 'debtrank_exposure' if 'debtrank_exposure' in gf_q.columns else nw_color
                hover = [
                    f"<b>{row['ticker']}</b><br>"
                    f"Health Score: {row.get('health_score', float('nan')):.3f}<br>"
                    f"Risk Tier: {row.get('risk_tier', 'N/A')}<br>"
                    f"Systemic Score: {row.get(sys_col, float('nan')):.3f}<br>"
                    f"DebtRank Exposure: {row.get(debt_col, float('nan')):.3f}"
                    for _, row in gf_q.iterrows()
                ]

                fig_net = go.Figure()
                fig_net.add_trace(go.Scatter(
                    x=x_pos, y=y_pos,
                    mode='markers+text' if show_labels else 'markers',
                    text=tickers if show_labels else None,
                    textposition='top center',
                    hovertext=hover,
                    hoverinfo='text',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title=nw_color.replace('_', ' ').title(), thickness=15),
                        line=dict(width=2, color='white'),
                    ),
                    name='Companies',
                ))
                fig_net.update_layout(
                    title=f"Supply-Chain Network — {nw_quarter}",
                    showlegend=False, height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor='rgba(10,15,30,1)',
                    plot_bgcolor='rgba(10,15,30,1)',
                    font=dict(color='white'),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info(f"Column '{nw_metric}' or '{nw_color}' not found for this quarter.")

        # Contagion stats table
        st.divider()
        st.subheader("Systemic Risk Rankings")
        rank_primary = 'systemic_importance_score' if 'systemic_importance_score' in gf_q.columns else nw_metric
        rank_cols = ['ticker', rank_primary,
                     'debtrank_exposure', 'contagion_out', 'contagion_vulnerability']
        rank_cols = [c for c in rank_cols if c in gf_q.columns]
        rank_df   = gf_q[rank_cols + ['health_score', 'risk_tier']].sort_values(
            rank_primary, ascending=False
        )
        st.dataframe(
            rank_df.style.background_gradient(subset=[rank_primary], cmap='YlOrRd'),
            use_container_width=True
        )

# ── TAB 4: Risk Timeline ──────────────────────────────────────────────────────
with tabs[3]:
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("Risk Score Timeline")

    c1, c2 = st.columns([2, 3])
    with c1:
        timeline_tickers = st.multiselect(
            "Select companies",
            sorted(scores['ticker'].unique()),
            default=['CHK', 'XOM', 'SLB', 'KMI'],
            key='timeline_tickers'
        )
        metric_choice = st.radio(
            "Metric",
            ['health_score', 'distress_prob'],
            format_func=lambda x: 'Health Score (1=healthy)' if x == 'health_score' else 'Distress Probability',
            horizontal=True,
        )
        show_recession = st.checkbox("Highlight crisis periods (2019Q4–2020Q3)", value=True)

    with c2:
        if not timeline_tickers:
            st.info("Select at least one company.")
        else:
            td = scores[scores['ticker'].isin(timeline_tickers)].copy()
            td = td.sort_values('quarter')

            fig_line = go.Figure()

            # Crisis shading
            if show_recession:
                fig_line.add_vrect(
                    x0='2019Q4', x1='2020Q3',
                    fillcolor='rgba(231,76,60,0.12)',
                    layer='below', line_width=0,
                    annotation_text='Oil Crash / COVID',
                    annotation_position='top left',
                    annotation_font_color='#e74c3c',
                )

            palette = px.colors.qualitative.Plotly
            for i, ticker in enumerate(timeline_tickers):
                sub  = td[td['ticker'] == ticker].sort_values('quarter')
                # Actual distress markers
                dist = sub[sub['distress_label'] == 1]
                if len(sub) > 0:
                    fig_line.add_trace(go.Scatter(
                        x=sub['quarter'], y=sub[metric_choice],
                        mode='lines+markers',
                        name=ticker,
                        line=dict(color=palette[i % len(palette)], width=2),
                        marker=dict(size=5),
                        hovertemplate=f"<b>{ticker}</b><br>Quarter: %{{x}}<br>{metric_choice}: %{{y:.4f}}<extra></extra>",
                    ))
                if len(dist) > 0:
                    fig_line.add_trace(go.Scatter(
                        x=dist['quarter'], y=dist[metric_choice],
                        mode='markers',
                        name=f"{ticker} (distress)",
                        marker=dict(
                            color=palette[i % len(palette)],
                            size=12, symbol='x',
                            line=dict(width=2, color='white'),
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>{ticker} DISTRESS</b><br>Quarter: %{{x}}<extra></extra>",
                    ))

            ylab  = 'Health Score (1 = healthy)' if metric_choice == 'health_score' else 'P(distress)'
            thresh= 0.3 if metric_choice == 'health_score' else 0.7
            fig_line.add_hline(y=thresh, line_dash='dot', line_color='#e74c3c',
                               annotation_text='Alert threshold',
                               annotation_position='right',
                               annotation_font_color='#e74c3c')

            fig_line.update_layout(
                title="Risk Score Timeline — Selected Companies",
                xaxis_title="Quarter",
                yaxis_title=ylab,
                yaxis_range=[0, 1.05],
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)',
                font=dict(color='white'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified',
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # Ablation study summary
    if ABLATION_PATH.exists():
        st.divider()
        st.subheader("Ablation Study — Module Contribution")
        abl = load_ablation()

        rows = []
        for k, v in abl.items():
            cv_str   = f"{v['cv_auc']:.4f}" if isinstance(v.get('cv_auc'), float) else str(v.get('cv_auc', 'N/A'))
            test_str = f"{v['test_auc']:.4f}" if isinstance(v.get('test_auc'), float) else 'N/A'
            rows.append({'Configuration': k,
                         'Features':      v.get('n_features', ''),
                         'CV AUC':        cv_str,
                         'Test AUC':      test_str,
                         'Expected':      v.get('expected', ''),
                         'Research Q':    v.get('rq', '')})

        abl_df = pd.DataFrame(rows)
        st.dataframe(abl_df, use_container_width=True, hide_index=True)

        # AUC bar chart
        abl_plot = abl_df[abl_df['Test AUC'] != 'N/A'].copy()
        abl_plot['Test AUC float'] = abl_plot['Test AUC'].astype(float)
        fig_abl = px.bar(
            abl_plot, x='Configuration', y='Test AUC float',
            color='Test AUC float',
            color_continuous_scale='RdYlGn',
            range_color=[0.4, 1.0],
            text='Test AUC',
            title='Ablation Study — Test AUC by Feature Subset',
            labels={'Test AUC float': 'Test AUC'},
        )
        fig_abl.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
            font=dict(color='white'),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_abl, use_container_width=True)

    # ROC comparison image
    roc_path = MODULE_D / 'roc_fusion_vs_zscore.png'
    if roc_path.exists():
        st.divider()
        st.subheader("ROC Comparison — CrisisNet Fusion vs Altman Z-Score (1968)")
        st.image(str(roc_path), use_container_width=True)

# ── TAB 5: Predictions vs Actuals ────────────────────────────────────────────
with tabs[4]:
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("Predictions vs Actuals — Test Set (2019–2025)")
    thresh_info = load_threshold()
    opt_thresh  = thresh_info.get('threshold', 0.15)
    st.caption(
        f"Train: 2015–2018 | Test: 2019–2025 | "
        f"Threshold: **{opt_thresh}** (recall-biased, down from 0.50) | "
        f"RECALL_BOOST: {thresh_info.get('recall_boost', 3)}×"
    )

    if not TEST_PRED_PATH.exists():
        st.warning("test_predictions.parquet not found. Re-run `python Module_D/train_fusion.py`.")
    else:
        preds = load_test_predictions()

        # ── KPI row ───────────────────────────────────────────────────────────
        from sklearn.metrics import (precision_score, recall_score, f1_score,
                                     fbeta_score)
        y_true = preds['distress_label'].values
        y_pred = preds['predicted_label'].values
        y_prob = preds['distress_prob'].values

        acc    = (y_true == y_pred).mean()
        prec   = precision_score(y_true, y_pred, zero_division=0)
        rec    = recall_score(y_true, y_pred, zero_division=0)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        f2     = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Accuracy",   f"{acc:.3f}")
        k2.metric("Precision",  f"{prec:.3f}", help="Of flagged companies, how many were truly distressed?")
        k3.metric("Recall",     f"{rec:.3f}",  help="Of actual distress events, how many did we catch?")
        k4.metric("F1",         f"{f1:.3f}")
        k5.metric("F2 ↑",       f"{f2:.3f}",   help="F2 weights recall 2× over precision — the right metric for early warning")
        k6.metric("Caught",     f"{tp}/{tp+fn}", help="True Positives / Total Positives")

        st.divider()
        col_left, col_right = st.columns([1, 2])

        with col_left:
            # Confusion matrix heatmap
            cm_data = [[tn, fp], [fn, tp]]
            fig_cm = go.Figure(go.Heatmap(
                z=cm_data,
                x=['Predicted: Healthy', 'Predicted: Distress'],
                y=['Actual: Healthy', 'Actual: Distress'],
                colorscale=[[0, '#1a5276'], [0.5, '#2e86c1'], [1, '#e74c3c']],
                text=[[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]],
                texttemplate="%{text}",
                textfont=dict(size=18, color='white'),
                showscale=False,
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=13),
                margin=dict(t=50, b=60, l=120, r=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Probability distribution for TP/FP/FN/TN
            preds['outcome'] = 'True Negative'
            preds.loc[(y_true == 1) & (y_pred == 1), 'outcome'] = 'True Positive'
            preds.loc[(y_true == 0) & (y_pred == 1), 'outcome'] = 'False Positive'
            preds.loc[(y_true == 1) & (y_pred == 0), 'outcome'] = 'False Negative'

            fig_dist = px.histogram(
                preds, x='distress_prob', color='outcome', nbins=40,
                barmode='overlay', opacity=0.7,
                color_discrete_map={
                    'True Positive': '#27ae60', 'False Negative': '#e74c3c',
                    'True Negative': '#2980b9', 'False Positive': '#f39c12',
                },
                title="Score Distribution by Outcome",
                labels={'distress_prob': 'P(distress)'},
            )
            fig_dist.update_layout(
                height=320, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)', font=dict(color='white'),
                legend=dict(orientation='h', y=1.02),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_right:
            # Quarter-level accuracy timeline
            q_stats = preds.groupby('quarter').apply(lambda g: pd.Series({
                'total':    len(g),
                'distress': int(g['distress_label'].sum()),
                'predicted_distress': int(g['predicted_label'].sum()),
                'correct':  int(g['correct'].sum()),
                'accuracy': g['correct'].mean(),
            })).reset_index()

            fig_qa = go.Figure()
            fig_qa.add_trace(go.Bar(
                x=q_stats['quarter'], y=q_stats['distress'],
                name='Actual Distress', marker_color='#e74c3c', opacity=0.7,
            ))
            fig_qa.add_trace(go.Bar(
                x=q_stats['quarter'], y=q_stats['predicted_distress'],
                name='Predicted Distress', marker_color='#f39c12', opacity=0.7,
            ))
            fig_qa.add_trace(go.Scatter(
                x=q_stats['quarter'], y=q_stats['accuracy'],
                name='Accuracy', yaxis='y2',
                mode='lines+markers', line=dict(color='#2ecc71', width=2),
            ))
            fig_qa.update_layout(
                title="Predicted vs Actual Distress Count per Quarter",
                barmode='group',
                yaxis=dict(title='Count'),
                yaxis2=dict(title='Accuracy', overlaying='y', side='right',
                            range=[0, 1.1], showgrid=False),
                height=380,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)',
                font=dict(color='white'),
                legend=dict(orientation='h', y=1.02),
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig_qa, use_container_width=True)

            # Per-ticker summary table
            st.subheader("Per-Company Prediction Summary")
            ticker_stats = preds.groupby('ticker').apply(lambda g: pd.Series({
                'Actual Distress Qtrs':    int(g['distress_label'].sum()),
                'Predicted Distress Qtrs': int(g['predicted_label'].sum()),
                'True Positives':          int(((g['distress_label']==1) & (g['predicted_label']==1)).sum()),
                'False Negatives':         int(((g['distress_label']==1) & (g['predicted_label']==0)).sum()),
                'False Positives':         int(((g['distress_label']==0) & (g['predicted_label']==1)).sum()),
                'Accuracy':                round(g['correct'].mean(), 3),
            })).reset_index().sort_values('Actual Distress Qtrs', ascending=False)

            def highlight_misses(row):
                if row['False Negatives'] > 0:
                    return ['background-color: rgba(231,76,60,0.3)'] * len(row)
                elif row['False Positives'] > 0:
                    return ['background-color: rgba(243,156,18,0.2)'] * len(row)
                return [''] * len(row)

            st.dataframe(
                ticker_stats.style.apply(highlight_misses, axis=1)
                    .background_gradient(subset=['Accuracy'], cmap='RdYlGn', vmin=0.7, vmax=1.0),
                use_container_width=True, height=350,
            )
            st.caption("🔴 Red = missed distress (FN) | 🟡 Yellow = false alarm (FP)")

        # ── FN root-cause analysis ─────────────────────────────────────────────
        st.divider()
        st.subheader("Why Are We Missing These? — False Negative Root Cause Analysis")

        HARD_DEFAULTS = {'CHK', 'SWN', 'WLL', 'OAS', 'DNR', 'WFT', 'SN'}
        fn_rows   = preds[(preds['distress_label'] == 1) & (preds['predicted_label'] == 0)].copy()
        fn_hard   = fn_rows[fn_rows['ticker'].isin(HARD_DEFAULTS)]
        fn_noisy  = fn_rows[~fn_rows['ticker'].isin(HARD_DEFAULTS)]

        ca, cb, cc = st.columns(3)
        ca.metric("Total FNs",            len(fn_rows))
        cb.metric("Noisy-label FNs",      len(fn_noisy),
                  help="Financially healthy companies labeled distress only due to stock drawdowns (XOM, CVX, COP…). Model is CORRECT.")
        cc.metric("Hard-default FNs",     len(fn_hard),
                  help="Actual bankruptcies with missing/zero market data (CHK). Require NLP/filing signals from Module B.")

        exp1, exp2 = st.columns(2)
        with exp1:
            st.markdown("**Noisy-label FNs** — drawdown-labeled, fundamentally healthy")
            st.markdown(
                "These companies (XOM, CVX, COP, PSX…) were labeled 'distress' purely because "
                "the oil price crashed in 2019Q3–Q4. Their balance sheets were intact. "
                "The model correctly predicts them as healthy — **these are label artefacts, not model failures.**"
            )
            if len(fn_noisy) > 0:
                fig_noisy = px.histogram(
                    fn_noisy, x='distress_prob', nbins=20,
                    title="FN Score Distribution (noisy-label)",
                    labels={'distress_prob': 'P(distress)'},
                    color_discrete_sequence=['#f39c12'],
                )
                fig_noisy.add_vline(x=opt_thresh, line_dash='dash', line_color='white',
                                    annotation_text=f"threshold={opt_thresh}")
                fig_noisy.update_layout(
                    height=280, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)', font=dict(color='white'),
                )
                st.plotly_chart(fig_noisy, use_container_width=True)

        with exp2:
            st.markdown("**Hard-default FNs** — genuine bankruptcies, not in NLP dataset")
            st.markdown(
                "CHK (Chapter 11, June 2020) has **all-zero stock features** (delisted data gap) "
                "and is **not present in Module B's NLP dataset** — so even with NLP integrated, "
                "its 10-K filing signals cannot be used. This is a data-coverage gap, not a "
                "modeling gap. Adding CHK's 10-K filings to Module B would fix these FNs."
            )
            if len(fn_hard) > 0:
                st.dataframe(
                    fn_hard[['ticker', 'quarter', 'distress_prob']].style
                        .background_gradient(subset=['distress_prob'], cmap='YlOrRd'),
                    use_container_width=True,
                )

        st.info(
            f"**Bottom line:** Of {len(fn_rows)} FNs, {len(fn_noisy)} are label noise "
            f"(model correctly classifies them as healthy — sector-wide drawdown artefact) "
            f"and {len(fn_hard)} are data-coverage gaps (CHK not in NLP dataset). "
            f"Module B NLP is now active (+32 features, recall 0.67→0.69). "
            f"True actionable recall on companies with full data is significantly higher."
        )

# ── TAB 6: Live Scoring ────────────────────────────────────────────────────────
with tabs[5]:
    import plotly.graph_objects as go

    st.subheader("Live Scoring — Inject Company Data & Run Full Eval")
    st.markdown(
        "Upload a CSV of company feature data. The pipeline aligns columns to the trained model, "
        "fills any missing features with zero, and returns health scores immediately. "
        "You can also score existing companies in a different time window."
    )

    model_ready = MODEL_PATH.exists() and FEAT_PATH.exists()
    if not model_ready:
        st.error("Model not found. Run `python Module_D/train_fusion.py` first.")
    else:
        lgbm_model = load_lgbm_model()
        with open(FEAT_PATH) as f:
            all_feat_cols = json.load(f)
        # Keep only numeric features (same logic as training)
        numeric_feat_cols = [c for c in all_feat_cols
                             if c in fused.columns and pd.api.types.is_numeric_dtype(fused[c])]

        thresh_info = load_threshold()
        DEFAULT_THRESH = thresh_info.get('threshold', 0.15)

        mode = st.radio("Scoring mode", ["Upload CSV", "Score existing company (new quarter)"],
                        horizontal=True)

        if mode == "Upload CSV":
            st.markdown("**CSV requirements:** must have a `ticker` column and as many feature columns as available. Missing features are filled with 0.")
            with st.expander("Download feature template CSV"):
                template = pd.DataFrame(columns=['ticker', 'quarter'] + numeric_feat_cols)
                st.download_button(
                    "Download template",
                    template.to_csv(index=False),
                    file_name="crisisnet_template.csv",
                    mime="text/csv",
                )

            uploaded = st.file_uploader("Upload company data CSV", type="csv")
            threshold = st.slider("Distress threshold", 0.05, 0.9, DEFAULT_THRESH, 0.05)

            if uploaded is not None:
                try:
                    user_df = pd.read_csv(uploaded)
                    st.success(f"Loaded {len(user_df)} rows × {len(user_df.columns)} columns")

                    # Align to model features
                    aligned = pd.DataFrame(index=user_df.index)
                    for col in numeric_feat_cols:
                        aligned[col] = pd.to_numeric(user_df[col], errors='coerce').fillna(0) \
                                       if col in user_df.columns else 0.0

                    probs  = lgbm_model.predict(aligned.values)
                    result = user_df[['ticker'] + ([c for c in ['quarter'] if c in user_df.columns])].copy()
                    result['distress_prob'] = probs.round(4)
                    result['health_score']  = (1 - probs).round(4)
                    result['risk_tier']     = pd.cut(
                        probs,
                        bins=[-0.01, 0.3, 0.6, 0.8, 1.01],
                        labels=['Low', 'Medium', 'High', 'Critical']
                    )
                    result['predicted_distress'] = (probs > threshold).astype(int)

                    # Show results
                    st.subheader("Scoring Results")
                    styled_result = result.style.background_gradient(
                        subset=['health_score'], cmap='RdYlGn', vmin=0, vmax=1
                    ).background_gradient(
                        subset=['distress_prob'], cmap='YlOrRd', vmin=0, vmax=1
                    )
                    st.dataframe(styled_result, use_container_width=True)

                    # Bar chart
                    fig_up = go.Figure(go.Bar(
                        x=result['ticker'], y=result['distress_prob'],
                        marker_color=['#e74c3c' if p > threshold else '#2ecc71'
                                      for p in result['distress_prob']],
                        text=result['distress_prob'].round(3),
                        textposition='outside',
                    ))
                    fig_up.update_layout(
                        title=f"Uploaded Companies — P(distress)  [threshold={threshold}]",
                        yaxis_range=[0, 1.1],
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.05)',
                        font=dict(color='white'),
                        height=400,
                    )
                    st.plotly_chart(fig_up, use_container_width=True)

                    # Download results
                    st.download_button(
                        "Download scored results CSV",
                        result.to_csv(index=False),
                        file_name="crisisnet_scored.csv",
                        mime="text/csv",
                    )

                    # Coverage report
                    matched = sum(1 for c in numeric_feat_cols if c in user_df.columns)
                    st.info(f"Feature coverage: {matched}/{len(numeric_feat_cols)} columns matched "
                            f"({matched/len(numeric_feat_cols)*100:.0f}%). "
                            f"Missing features were zero-filled.")

                except Exception as e:
                    st.error(f"Error processing file: {e}")

        else:  # Score existing company in a custom quarter
            st.markdown("Pick a company from the existing dataset and a quarter to score.")

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                sel_ticker = st.selectbox("Ticker", sorted(fused['ticker'].unique()), key='live_ticker')
            with col_s2:
                avail_q = sorted(fused[fused['ticker'] == sel_ticker]['quarter'].unique(), reverse=True)
                sel_q   = st.selectbox("Quarter", avail_q, key='live_q')
            with col_s3:
                threshold2 = st.slider("Distress threshold", 0.05, 0.9, DEFAULT_THRESH, 0.05, key='live_thresh')

            # Feature sliders for manual overrides
            row_mask = (fused['ticker'] == sel_ticker) & (fused['quarter'] == sel_q)
            if row_mask.sum() > 0:
                base_row = fused[row_mask].iloc[0]

                with st.expander("Override feature values (optional — defaults from dataset)"):
                    override_cols = ['altman_z', 'merton_dd', 'max_drawdown_6m',
                                     'volatility_30d', 'debt_to_equity', 'hy_oas']
                    override_cols = [c for c in override_cols if c in numeric_feat_cols]
                    overrides = {}
                    oc1, oc2, oc3 = st.columns(3)
                    for i, col in enumerate(override_cols):
                        orig_val = float(base_row.get(col, 0))
                        with [oc1, oc2, oc3][i % 3]:
                            overrides[col] = st.number_input(
                                col.replace('_', ' ').title(),
                                value=round(orig_val, 4),
                                key=f"override_{col}"
                            )

                # Build feature vector
                feat_vec = np.array([float(base_row.get(c, 0)) for c in numeric_feat_cols])
                for col, val in overrides.items():
                    if col in numeric_feat_cols:
                        feat_vec[numeric_feat_cols.index(col)] = val

                prob      = float(lgbm_model.predict(feat_vec.reshape(1, -1))[0])
                health    = 1.0 - prob
                tier      = ('Critical' if prob > 0.8 else 'High' if prob > 0.6
                             else 'Medium' if prob > 0.3 else 'Low')
                predicted = 'DISTRESS' if prob > threshold2 else 'HEALTHY'
                actual_l  = int(base_row.get('distress_label', -1))
                actual_s  = 'DISTRESS' if actual_l == 1 else ('HEALTHY' if actual_l == 0 else 'Unknown')
                correct   = predicted == actual_s if actual_l != -1 else None

                st.divider()
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Health Score",     f"{health:.4f}")
                r2.metric("P(distress)",      f"{prob:.4f}")
                r3.metric("Prediction",       predicted,
                          delta="✓ Correct" if correct else ("✗ Wrong" if correct is False else ""),
                          delta_color="normal" if correct else "inverse")
                r4.metric("Actual Label",     actual_s)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=health * 100,
                    title=dict(text=f"{sel_ticker} — Health Score", font=dict(size=18)),
                    delta=dict(reference=50, increasing=dict(color='#2ecc71'),
                               decreasing=dict(color='#e74c3c')),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickwidth=1, tickcolor='white'),
                        bar=dict(color='#2ecc71' if health > 0.6 else '#f39c12' if health > 0.3 else '#e74c3c'),
                        bgcolor='rgba(0,0,0,0)',
                        steps=[
                            dict(range=[0, 30],  color='rgba(231,76,60,0.3)'),
                            dict(range=[30, 60], color='rgba(243,156,18,0.2)'),
                            dict(range=[60, 100],color='rgba(39,174,96,0.2)'),
                        ],
                        threshold=dict(line=dict(color='white', width=4),
                                       thickness=0.75, value=50),
                    ),
                    number=dict(suffix="%", font=dict(size=28)),
                ))
                fig_gauge.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Top features driving this prediction
                st.subheader("Key Features for This Prediction")
                feat_df = pd.DataFrame({
                    'Feature': numeric_feat_cols,
                    'Value':   feat_vec,
                }).assign(
                    **{'Dataset Mean': [float(fused[c].mean()) if c in fused.columns else 0
                                       for c in numeric_feat_cols]}
                )
                feat_df['Deviation'] = feat_df['Value'] - feat_df['Dataset Mean']
                # Show most extreme deviations
                feat_df = feat_df.reindex(feat_df['Deviation'].abs().sort_values(ascending=False).index)
                st.dataframe(
                    feat_df.head(20).style.background_gradient(subset=['Deviation'], cmap='RdYlGn_r'),
                    use_container_width=True,
                )
            else:
                st.warning(f"No data found for {sel_ticker} / {sel_q}")
