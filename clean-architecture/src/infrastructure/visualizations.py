# src/simulation_dashboard/infrastructure/visualizations.py
import plotly.express as px
import plotly.graph_objs as go
import networkx as nx


def create_n1_chart(node_indices, lost_ratios):
    fig = px.bar(x=node_indices, y=lost_ratios,
                 labels={'x': 'Node Index', 'y': 'Lost Production (%)'},
                 title="N-1 Contingency Analysis")
    fig.update_layout(yaxis_range=[0, max(lost_ratios)*1.2])
    return fig

def create_renewable_chart(scenario_ids, total_production):
    fig = px.scatter(x=scenario_ids, y=total_production,
                     labels={'x': 'Scenario ID', 'y': 'Total Production (MW)'},
                     title="Renewable Sensitivity Analysis")
    return fig

def create_maintenance_chart(scenarios, threshold_fraction, baseline):
    labels = [s["scenario"] for s in scenarios]
    remaining = [s["remaining"] for s in scenarios]
    colors = ['green' if r >= threshold_fraction * baseline else 'red' for r in remaining]
    fig = px.bar(x=labels, y=remaining,
                 labels={'x': 'Scenario', 'y': 'Remaining Production (MW)'},
                 title="Preventive Maintenance Simulation")
    fig.update_traces(marker_color=colors)
    fig.add_hline(y=threshold_fraction * baseline,
                  line_dash="dot", annotation_text=f"Threshold ({threshold_fraction*100:.0f}%)", 
                  annotation_position="top left")
    return fig

def create_investment_chart(scenarios):
    labels = [s["scenario"] for s in scenarios]
    totals = [s["total"] for s in scenarios]
    baseline = scenarios[0]["total"]
    fig = px.bar(x=labels, y=totals,
                 labels={'x': 'Scenario', 'y': 'Total Production Capacity (MW)'},
                 title="Infrastructure Investment Scenarios")
    fig.add_hline(y=baseline, line_dash="dash", 
                  annotation_text=f"Baseline: {baseline} MW", annotation_position="bottom right")
    return fig

def create_cyber_grid_figure(network, title="Grid Topology"):
    pos = nx.spring_layout(network, seed=42)
    node_x, node_y, node_text, node_color = [], [], [], []
    for node, data in network.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        typ = data.get("type")
        if typ == "generator":
            node_color.append("green")
            node_text.append(f"{node}<br>Gen: {data.get('capacity')}")
        elif typ == "consumer":
            node_color.append("red")
            node_text.append(f"{node}<br>Demand: {data.get('demand')}")
        else:
            node_color.append("orange")
            node_text.append(f"{node}<br>Substation")
    edge_x, edge_y = [], []
    for u, v in network.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=2, color="#888"),
                            hoverinfo="none", mode="lines")
    node_trace = go.Scatter(x=node_x, y=node_y,
                            mode="markers+text",
                            text=node_text, hoverinfo="text",
                            marker=dict(color=node_color, size=30, line_width=2))
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=dict(text=title, font=dict(size=16)),
                                     showlegend=False,
                                     hovermode="closest",
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig
