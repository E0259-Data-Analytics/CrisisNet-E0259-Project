"""
CrisisNet — Interactive Dashboard
===================================
Four-tab Streamlit app for exploring CrisisNet fusion model outputs.

Tabs:
  1. Company Scorecard    — live health rankings with filters
  2. SHAP Waterfall       — per-company SHAP explainability
  3. Network Graph        — supply-chain contagion explorer
  4. Risk Timeline        — multi-company historical risk evolution

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

HEALTH_PATH  = MODULE_D / "health_scores.parquet"
FUSED_PATH   = MODULE_D / "X_fused.parquet"
SHAP_PATH    = MODULE_D / "shap_values.npy"
FEAT_PATH    = MODULE_D / "shap_feat_cols.json"
ABLATION_PATH= MODULE_D / "ablation_results.json"
METRICS_PATH = MODULE_D / "metrics.json"
GRAPH_PATH   = MODULE_C / "results" / "exports" / "X_graph.parquet"
GRAPH_PKL    = MODULE_C / "data" / "processed" / "supply_chain_graph.pkl"

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
    if METRICS_PATH.exists():
        m = load_metrics()
        cv_info = m.get('cv_walk_forward', {})
        st.metric("CV Mean AUC", f"{cv_info.get('mean_AUC', 'N/A'):.3f}" if isinstance(cv_info.get('mean_AUC'), float) else "N/A")
        cf = m.get('CrisisNet Fusion', {})
        st.metric("Test AUC", f"{cf.get('AUC', 'N/A'):.3f}" if isinstance(cf.get('AUC'), float) else "N/A")
        az = m.get('Altman Z-Score (1968)', {})
        if isinstance(cf.get('AUC'), float) and isinstance(az.get('AUC'), float):
            lift = cf['AUC'] - az['AUC']
            st.metric("Lift vs Z-Score", f"+{lift:.3f}", delta_color="normal")

    st.divider()
    st.caption("CrisisNet v1.0 | April 2026")

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
            .applymap(color_tier, subset=['Risk Tier'])
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
        with col1:
            nw_quarter = st.selectbox("Quarter", sorted(gf['quarter'].unique(), reverse=True), key='nw_q')
            nw_metric  = st.selectbox(
                "Node size by",
                ['graph_systemic_importance_score', 'graph_pagerank',
                 'graph_betweenness_centrality', 'graph_contagion_vulnerability'],
                format_func=lambda x: x.replace('graph_', '').replace('_', ' ').title()
            )
            nw_color   = st.selectbox(
                "Node color by",
                ['graph_debtrank_exposure', 'graph_contagion_out',
                 'graph_louvain_community_id', 'graph_systemic_risk_contribution'],
                format_func=lambda x: x.replace('graph_', '').replace('_', ' ').title()
            )
            show_labels= st.checkbox("Show labels", value=True)

        gf_q   = gf[gf['quarter'] == nw_quarter].copy()
        # Health scores overlay
        hs_q   = scores[scores['quarter'] == nw_quarter][['ticker', 'health_score', 'risk_tier']]
        gf_q   = gf_q.merge(hs_q, on='ticker', how='left')

        with col2:
            if nw_metric in gf_q.columns and nw_color in gf_q.columns:
                # Layout: circular by subsector
                import math
                tickers = gf_q['ticker'].tolist()
                n       = len(tickers)
                angles  = [2 * math.pi * i / n for i in range(n)]
                x_pos   = [math.cos(a) for a in angles]
                y_pos   = [math.sin(a) for a in angles]
                pos     = {t: (x, y) for t, x, y in zip(tickers, x_pos, y_pos)}

                sizes  = gf_q[nw_metric].fillna(0).values
                sizes  = 8 + 42 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1e-9)
                colors = gf_q[nw_color].fillna(0).values
                health = gf_q['health_score'].fillna(0.5).values

                # hover text
                hover = [
                    f"<b>{row['ticker']}</b><br>"
                    f"Health Score: {row.get('health_score', float('nan')):.3f}<br>"
                    f"Risk Tier: {row.get('risk_tier', 'N/A')}<br>"
                    f"Systemic Score: {row.get('graph_systemic_importance_score', float('nan')):.3f}<br>"
                    f"DebtRank Exposure: {row.get('graph_debtrank_exposure', float('nan')):.3f}"
                    for _, row in gf_q.iterrows()
                ]

                fig_net = go.Figure()

                # Nodes
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
                        colorbar=dict(
                            title=nw_color.replace('graph_', '').replace('_', ' ').title(),
                            thickness=15,
                        ),
                        line=dict(width=2, color='white'),
                    ),
                    name='Companies',
                ))

                fig_net.update_layout(
                    title=f"Supply-Chain Network — {nw_quarter}",
                    showlegend=False,
                    height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor='rgba(10,15,30,1)',
                    plot_bgcolor='rgba(10,15,30,1)',
                    font=dict(color='white'),
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info(f"Column {nw_metric} or {nw_color} not found in X_graph for this quarter.")

        # Contagion stats table
        st.divider()
        st.subheader("Systemic Risk Rankings")
        rank_cols = ['ticker', 'graph_systemic_importance_score', 'graph_debtrank_exposure',
                     'graph_contagion_out', 'graph_contagion_vulnerability']
        rank_cols = [c for c in rank_cols if c in gf_q.columns]
        rank_df   = gf_q[rank_cols + ['health_score', 'risk_tier']].sort_values(
            'graph_systemic_importance_score', ascending=False
        )
        st.dataframe(
            rank_df.style.background_gradient(subset=['graph_systemic_importance_score'], cmap='YlOrRd'),
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
