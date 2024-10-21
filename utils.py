import plotly.graph_objects as go

# Function to create the gauge chart
def create_gauge_chart(probability):
    # Determine color based on churn probability
    if probability < 0.3:
        color = "green"
    elif probability < 0.6:
        color = "yellow"
    else:
        color = "red"

    # Create a gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': "Churn Probability",
                'font': {
                    'size': 22,
                    'color': 'black'  # Set title font color to black
                }
            },
            number={'font': {'size': 38, 'color': 'black'}},  # Set the number font color to black
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 2,
                    'tickcolor': "black",  # Change tick color to black to be visible on a white background
                    'tickfont': {'size': 12, 'color': 'black'}  # Tick font color is black
                },
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "black",  # Set border color to black
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.4)"},
                    {'range': [30, 60], 'color': "rgba(255, 255, 0, 0.4)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.4)"}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},  # Set threshold line color to black
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        )
    )

    # Update chart layout for a sleek look
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "black"},  # General font color set to black
        width=450,  # Slightly increased width for better balance
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Function to create the model probability bar chart
def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(data=[
        go.Bar(
            y=models,
            x=probs,
            orientation='h',
            text=[f'{p:.2%}' for p in probs],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Churn Probability by Model',
        yaxis_title='Models',
        xaxis_title='Probability',
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig
