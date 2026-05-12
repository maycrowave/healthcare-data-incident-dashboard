import plotly.graph_objects as go
from typing import Dict, List, Optional
import pandas as pd
from utils.constants import _PER_CLASS_COLOURS

def horizontal_probability_bar(probability: float, baseline: Optional[float] = None, height: int = 70) -> go.Figure:
    """
    A single horizontal bar showing a probability on a fixed [0, 1] axis with an optional vertical line mark for the historical baseline
    Designed to be rendered one per class on the simulator's classifier panel
    """
    # Validation to catch any values out of range 0 to 1 (should never but just checking)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability out of range [0, 1]: {probability}")
    if baseline is not None and not 0.0 <= baseline <= 1.0:
        raise ValueError(f"Baseline out of range [0, 1]: {baseline}")

    fig = go.Figure()

    # Background unfilled portion of the bar
    fig.add_shape(
        type="rect",
        x0=0,
        x1=1,
        y0=-0.1,
        y1=1,
        xref="x",
        yref="y",
        fillcolor="#EEEEEE",
        line=dict(width=0),
        layer="below"
    )

    # The probability fill
    fig.add_shape(
        type="rect",
        x0=0,
        x1=probability,
        y0=-0.1,
        y1=1.05,
        xref="x",
        yref="y",
        fillcolor="#BE83E2",
        line=dict(width=0)
    )

    # The baseline tick mark, a vertical line slightly taller than the bar
    if baseline is not None:
        fig.add_shape(
            type="line",
            x0=baseline,
            x1=baseline,
            y0=-0.15,
            y1=1.15,
            xref="x",
            yref="y",
            line=dict(color="#000000", width=2)
        )

    # Invisible scatter trace to hold the hover tooltip
    hover_text = f"Predicted: {probability:.1%}"
    if baseline is not None:
        hover_text += f"<br>Historical baseline: {baseline:.1%}"
        
    fig.add_trace(go.Scatter(
        x=[probability],
        y=[0.5],
        mode="markers",
        marker=dict(size=0.1, opacity=0),
        hovertemplate=hover_text + "<extra></extra>",
        showlegend=False
    ))

    # Axis and layout settings to make a bar with no gaps or ticks and a fixed width for the probability range
    fig.update_xaxes(
        range=[0, 1],
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ticktext=["0%", "25%", "50%", "75%", "100%"],
        showgrid=False,
        zeroline=False,
        ticks="outside",
        ticklen=4
    )

    fig.update_yaxes(
        range=[-0.2, 1.2],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True
    )
    
    fig.update_layout(
        barmode="stack",
        height=height,
        margin=dict(l=0, r=0, t=4, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="left",
            x=0,
            font=dict(size=12)
        ),
        xaxis=dict(
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True
        )
    )
    return fig

def horizontal_stacked_proportions(proportions: Dict[str, float], class_order: List[str], height: int = 80) -> go.Figure:
    """
    A single horizontal bar split into segments one per class sized by proportion
    Used for the cluster panel's cluster decision distribution
    """
    fig = go.Figure()

    # Validation to catch any values out of range 0 to 1 (should never but just checking)
    for class_label in class_order:
        proportion = proportions.get(class_label, 0.0)
        if not 0.0 <= proportion <= 1.0:
            raise ValueError(
                f"Proportion out of range [0, 1] for {class_label}: {proportion}"
            )

        # Bar segment for this class
        fig.add_trace(go.Bar(
            x=[proportion],
            y=[""],
            orientation="h",
            name=class_label,
            marker=dict(color=_PER_CLASS_COLOURS.get(class_label, "#888888")),
            text=f"{proportion:.0%}" if proportion >= 0.05 else "",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#000000", size=14),
            hovertemplate=f"{class_label}: {proportion:.1%}<extra></extra>",
        ))

    # Layout settings for a stacked bar with no gaps or ticks
    fig.update_layout(
        barmode="stack",
        height=100,
        margin=dict(l=0, r=0, t=4, b=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-1,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        xaxis=dict(
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True
        )
    )
    return fig


def vertical_stacked_proportions_by_feature(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    class_order: List[str],
    feature_value_order: Optional[List[str]] = None,
    height: int = 380,
    barmode: str = "stack"
) -> go.Figure:
    """
    Stacked vertical bar chart: one bar per feature value, each bar split into class proportions
    Used for the "decision outcomes by feature" panels on the explore page
    """
    # Counts per feature_value, class reindex to make sure all classes are present
    counts = (
        df.groupby([feature_col, target_col]).size()
        .unstack(target_col)
        .reindex(columns=class_order, fill_value=0)
    )

    # Determines the feature value order on the x-axis, default is by total count 
    if feature_value_order is None:
        feature_value_order = counts.sum(axis=1).sort_values(ascending=False).index.tolist()
    counts = counts.reindex(feature_value_order)

    # Convert to normalised row proportions for the stack
    row_totals = counts.sum(axis=1).replace(0, 1)
    proportions = counts.div(row_totals, axis=0)

    # Class proportions stacked bar chart, one segment per class label
    fig = go.Figure()
    for class_label in class_order:
        fig.add_trace(go.Bar(
            x=proportions.index.astype(str).tolist(),
            y=proportions[class_label].tolist(),
            name=class_label,
            marker=dict(color=_PER_CLASS_COLOURS.get(class_label, "#888888")),
            customdata=counts[class_label].tolist(),
            hovertemplate=(
                f"<b>{class_label}</b><br>"
                f"%{{x}}<br>"
                f"%{{y:.1%}} of breaches in this group<br>"
                f"%{{customdata:,}} breaches<extra></extra>"
            )
        ))

    # Layout settings to make a stacked bar with no gaps and a fixed width for the probability range
    # Replace the existing layout block with:
    many_values = len(feature_value_order) > 7
    fig.update_layout(
        barmode=barmode,
        height=520 if many_values else 380,
        margin=dict(l=0, r=0, t=4, b=140 if many_values else 60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.50 if many_values else -0.25,
            xanchor="left",
            x=0
        ),
        
        xaxis=dict(
            title=None,
            tickangle=-30 if many_values else 0,
            automargin=True
        ),
        
        yaxis=dict(
            title="Share of breaches",
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)"
        )
    )
    return fig


def time_trend_counts(
    df: pd.DataFrame,
    period_col: str,
    target_col: str,
    class_order: List[str],
    period_order: List[str],
    height: int = 380
) -> go.Figure:
    """
    Line chart: breach counts per quarter, one line per decision outcome
    """
    # Counts per quarter, class reindexed to make sure all classes and quarters are present
    counts = (
        df.groupby([period_col, target_col]).size()
        .unstack(target_col)
        .reindex(index=period_order, columns=class_order, fill_value=0)
    )

    # Line chart with one line per class matching the per class colours
    fig = go.Figure()
    for class_label in class_order:
        fig.add_trace(go.Scatter(
            x=counts.index.tolist(),
            y=counts[class_label].tolist(),
            mode="lines+markers",
            name=class_label,
            line=dict(color=_PER_CLASS_COLOURS.get(class_label, "#888888"), width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{class_label}</b><br>%{{x}}<br>%{{y:,}} breaches<extra></extra>"
        ))

    # Line chart layout
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
        xaxis=dict(title=None, tickangle=-30),
        yaxis=dict(title="Breaches reported", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        hovermode="x unified"
    )
    return fig


def time_trend_proportions(
    df: pd.DataFrame,
    period_col: str,
    target_col: str,
    class_order: List[str],
    period_order: List[str],
    height: int = 380
) -> go.Figure:
    """
    Line chart: per quarter share of each decision outcome, sums to 100% within each quarter, one line per outcome
    """
    # Counts per quarter, class reindexed to make sure all classes and quarters are present
    counts = (
        df.groupby([period_col, target_col]).size()
        .unstack(target_col)
        .reindex(index=period_order, columns=class_order, fill_value=0)
    )
    row_totals = counts.sum(axis=1).replace(0, 1)
    proportions = counts.div(row_totals, axis=0)

    # Line chart with one line per class matching the per class colours
    fig = go.Figure()
    for class_label in class_order:
        fig.add_trace(go.Scatter(
            x=proportions.index.tolist(),
            y=proportions[class_label].tolist(),
            mode="lines+markers",
            name=class_label,
            line=dict(color=_PER_CLASS_COLOURS.get(class_label, "#888888"), width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{class_label}</b><br>%{{x}}<br>%{{y:.1%}} of quarter<extra></extra>"
        ))

    # Line chart layout
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
        xaxis=dict(title=None, tickangle=-30),
        yaxis=dict(
            title="Share of quarter's breaches",
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)"
        ),
        hovermode="x unified"
    )
    return fig