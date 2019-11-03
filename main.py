import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import evolutionary_algorithms as ea

avg_fitness_per_generation = []
variance_per_generation = []
best_chromosome = [[0]]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(name=__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div([
        html.Span('Mutation Algorithms'),
        dcc.Dropdown(id='mutation_dropdown',
                     options=[
                         {'label': 'default', 'value': 0},
                     ]),
    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='Mutation-div',
    ),
    html.Div([
        html.Span('Cross over Algorithms'),
        dcc.Dropdown(id='cross-over-dropdown',
                     options=[
                         {'label': 'default', 'value': 0},
                     ]),
    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='Cross-over-div',
    ),
    html.Div([
        html.Span('Selection Algorithms'),
        dcc.Dropdown(id='selection_algorithm_dropdown',
                     options=[
                         {'label': 'uniform', 'value': 0},

                     ]),

    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='Selection-div',
    ),
    html.Div([
        html.Span('Parents'),
        dcc.Input(id='parents', value='0'),
        html.Span('Population'),
        dcc.Input(id='population', value='0'),
        html.Span('Children'),
        dcc.Input(id='children', value='0'),
        html.Span('Number of Queen'),
        dcc.Input(id='Queen-number', value='0')

    ], style={'width': '100%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'}),
    html.Div([
        html.Button('RUN', id='run-btn', style={'float': 'left', 'margin': '10px'}, n_clicks=0),
    ], style={'align': 'center', 'margin': 'auto', 'width': '40%'}),
    html.Div(
        [
            html.Div(id='best-solution-graph', style={'width': '34%', 'float': 'right'}),
            html.Div(id='fitness-graph', style={'width': '70%'}),
        ],
        style={'margin-top': '100px'}
    ),
    html.Div(
        [
            html.Div(id='variance-graph', style={'width': '70%'}),
        ],
        style={'margin-top': '100px'}
    ),
    dcc.Interval(id='interval', interval=2 * 1000),

], style={'align': 'center', 'background': '#EEE'})


@app.callback(Output(component_id='Selection-div', component_property='children'),
              [Input(component_id='selection_algorithm_dropdown', component_property='value')])
def mutation_drop_down(input):
    if input is None:
        return [
            html.Span('Selection Algorithms'),
            dcc.Dropdown(id='selection_algorithm_dropdown',
                         options=[
                             {'label': 'uniform', 'value': 1},

                         ], value=None),

        ]
    elif input == 0:
        return [
            html.Span('Selection Algorithms'),
            dcc.Dropdown(id='selection_algorithm_dropdown',
                         options=[
                             {'label': 'uniform', 'value': 1},

                         ], value=0),

        ]


@app.callback(Output(component_id='Cross-over-div', component_property='children'),
              [Input(component_id='cross-over-dropdown', component_property='value')])
def mutation_drop_down(input):
    if input is None:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=[
                             {'label': 'default', 'value': 0},
                         ], value=None),
        ]
    elif input == 0:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=[
                             {'label': 'default', 'value': 0},
                         ], value=0),
        ]


@app.callback(Output(component_id='Mutation-div', component_property='children'),
              [Input(component_id='mutation_dropdown', component_property='value')])
def mutation_drop_down(input):
    if input is None:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation_dropdown',
                         options=[
                             {'label': 'default', 'value': 0},
                         ], value=None),
        ]
    elif input == 0:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation_dropdown',
                         options=[
                             {'label': 'default', 'value': 0},
                         ], value=0),

            html.Span('Probability'),
            dcc.Input(id='mutation_probability', value='0.05'),
        ]


@app.callback(
    Output(component_id='run-btn', component_property='children'),
    [Input(component_id='run-btn', component_property='n_clicks')],
)
def run_btn(n_clicks):
    global avg_fitness_per_generation, variance_per_generation, best_chromosome
    if n_clicks > 0:
        avg_fitness_per_generation = []
        variance_per_generation = []
        best_chromosome = [[0]]
        ga = ea.GeneticAlgorithms()
        ga.run(variance_per_generation,
               avg_fitness_per_generation,
               best_chromosome)
    return 'RUN'


@app.callback(
    Output(component_id='fitness-graph', component_property='children'),
    [Input(component_id='interval', component_property='n_intervals'), ]
)
def update_fittness_graph(_):
    global avg_fitness_per_generation
    global variance_per_generation
    return [
        dcc.Graph(
            id='fitness-graph-plot',
            figure={
                'data': [
                    {'x': np.arange(0, len(avg_fitness_per_generation)),
                     'y': avg_fitness_per_generation,
                     'type': 'line',
                     'name': 'Avg. fitness'},
                ],
                'layout': {
                    'title': ' Average Fitness per generation',
                }
            },
        ),
        dcc.Graph(
            id='variance-graph-plot',
            figure={
                'data': [
                    {'x': np.arange(0, len(variance_per_generation)),
                     'y': variance_per_generation,
                     'type': 'line',
                     'name': 'variance fitness'},
                ],
                'layout': {
                    'title': ' variance Fitness per generation',
                }
            },
        ),
    ]


@app.callback(
    Output(component_id='best-solution-graph', component_property='children'),
    [Input(component_id='interval', component_property='n_intervals'),
     Input(component_id='run-btn', component_property='n_clicks')],
)
def update_best_solution_graph(_, n_clicks):
    global best_chromosome
    # print(n_clicks)
    if n_clicks == 0:
        return
    # print(np.array(best_chromosome[0]))
    fig = go.Figure(
        data=go.Heatmap(
            z=np.array(best_chromosome[0]),
            colorscale=[
                # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
                [0, "rgb(200, 200, 200)"],
                [0.6, "rgb(0, 0, 0)"],
                [1, "rgb(70, 255, 70)"]
            ],
            showscale=False,
        ))
    fig.update_layout(title='-----Best Chromosome',
                      )
    return dcc.Graph(
        id='fitness-graph-plot',
        figure=fig,
    )


if __name__ == '__main__':
    app.run_server(debug=True)
