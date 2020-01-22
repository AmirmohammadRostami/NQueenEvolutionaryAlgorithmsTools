import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import evolutionary_algorithms
import os
import pickle
import datetime
from evolutionary_algorithms_functions import *

'''
variables
'''
data_fit = []
data_var = []
x_max_value = 1
y_avg_max_value = 0
y_var_max_value = 0
y_avg_min_value = 0
y_var_min_value = 0

interval_counter = 0
running = False
log_value = []
x_axis_thick_number = 15
y_axis_thick_number = 10
avg_fitness_per_generation = []
variance_per_generation = []
best_chromosome = [[0]]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
logs = []
drop_down_logs = []

mutation_options = [
    {'label': 'Default', 'value': 0},
    {'label': 'Random swap mutation', 'value': 1},
    {'label': 'Shuffle index mutation', 'value': 2},
    {'label': 'Neighbor based mutation', 'value': 3},
    {'label': 'Scramble mutation', 'value': 4},
    {'label': 'Insertion mutation', 'value': 5},
    {'label': 'Reverse mutation', 'value': 6},

]
cross_over_options = [
    {'label': 'Default', 'value': 0},
    {'label': 'Multi points crossover', 'value': 1},
    {'label': 'UPMX crossover', 'value': 2},
    {'label': 'Edge crossover', 'value': 3},
    {'label': 'Order one crossover', 'value': 4},
    {'label': 'Masked crossover crossover', 'value': 5},
    {'label': 'Maximal preservation crossover', 'value': 6},
    {'label': 'Position based crossover', 'value': 7},

]
parent_selection_options = [
    {'label': 'Uniform', 'value': 0},

]
remaining_selection_options = [
    {'label': 'Uniform', 'value': 0},
    {'label': 'Fitness based selection', 'value': 1},
    {'label': 'Boltzmann selection', 'value': 2},
    {'label': 'Fitness + Q tournament selection', 'value': 2},

]
evaluation_options = [
    {'label': 'Default', 'value': 0},
]

stop_condition_options = [
    {'label': 'Max generation', 'value': 0},
    {'label': 'Max evaluation', 'value': 1},
]


# read logs
def read_logs():
    for i, file in enumerate(os.listdir('./log_files/')):
        con = True
        for log in logs:
            if log['name'] == file[:-7]:
                con = False
                break
        if file.endswith('.pickle') and con:
            # print(file)
            p = pickle.load(open('./log_files/' + file, 'rb'))
            var_fit = []
            avg_fit = []
            for data in p:
                var_fit.append(data['var_fitness'])
                avg_fit.append((data['avg_fitness']))
            drop_down_logs.append({'label': file[:-7], 'value': len(logs)})
            logs.append({'name': file[:-7], 'var_fit': var_fit, 'avg_fit': avg_fit})
    # print(logs)


read_logs()

app = dash.Dash(name=__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.Span('Name', style={'margin': '20px'}),
        dcc.Input(id='name-input', placeholder='Enter name of experiment', style={'width': '50%'}),
    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='name-div',
    ),
    html.Div([
        html.Span('Mutation Algorithms'),
        dcc.Dropdown(id='mutation-dropdown',
                     options=mutation_options,
                     value=0),

        html.Span('Probability', style={'display': 'None'}),
        dcc.Input(id='mutation-probability', value='0.5', style={'display': 'None'}),
    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='mutation-div',
    ),
    html.Div([
        html.Span('Cross over Algorithms'),
        dcc.Dropdown(id='cross-over-dropdown',
                     options=cross_over_options,
                     value=0),
        html.Span('Probability of which parents', style={'display': 'None'}),
        dcc.Input(id='parents-probability', value='0.5', style={'display': 'None'}),
        html.Span('Points', style={'display': 'None'}),
        dcc.Input(id='cross-over-points-number', value='2', style={'display': 'None'}),
    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='cross-over-div',
    ),
    html.Div([
        html.Span('Parents Selection Algorithms'),
        dcc.Dropdown(id='parents-selection-dropdown',
                     options=parent_selection_options,
                     value=0),

    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='parent-selection-div',
    ),
    html.Div([
        html.Span('Remaining Population Selection Algorithms'),
        dcc.Dropdown(id='remaining-selection-dropdown',
                     options=parent_selection_options,
                     value=0),
        html.Span('Parameter', style={'display': 'None'}),
        dcc.Input(id='remaining_pop_parameter', value='1', style={'display': 'None'}),

    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='remaining-selection-div',
    ),
    html.Div([
        html.Span('Stop condition'),
        dcc.Dropdown(id='stop-condition-dropdown',
                     options=stop_condition_options,
                     value=0),

    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='stop-condition-div',
    ),
    html.Div([
        html.Span('select logs for show'),
        dcc.Dropdown(
            id='log-dropdown',
            options=drop_down_logs,
            multi=True
        ),
        html.Div(id='output-container')
        ,
    ], style={'width': '80%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'},
        id='log-div',
    ),

    html.Div([
        html.Span('Generation/Evaluation'),
        dcc.Input(id='generation-input', value='750'),
        html.Span('Population'),
        dcc.Input(id='population-input', value='500'),
        html.Span('Children'),
        dcc.Input(id='children-input', value='100'),
        html.Span('Number of Queen'),
        dcc.Input(id='queen-number-input', value='8')

    ], style={'width': '100%', 'align': 'right', 'display': 'inline-block', 'margin': '10px'}),
    html.Div([
        html.Button('RUN', id='run-btn', style={'float': 'left', 'margin': '10px'}, n_clicks=0),
    ], style={'align': 'center', 'margin': 'auto', 'width': '40%'}),
    html.Div(
        [
            html.Div(id='best-solution-graph', style={'width': '34%', 'float': 'right'}),
            html.Div(id='avg_graph', style={'width': '70%'}),
        ],
        style={'margin-top': '100px', 'background': '#FFF'}
    ),
    html.Div(
        [
            html.Div(id='variance-graph', style={'width': '70%'}),
        ],
        style={'margin-top': '100px'}
    ),
    dcc.Interval(id='interval', interval=1 * 1000),

], style={'align': 'center', 'background-image': './back_ground_img.png'})


@app.callback(Output(component_id='remaining-selection-div', component_property='children'),
              [Input(component_id='remaining-selection-dropdown', component_property='value'), ],
              [State(component_id='remaining_pop_parameter', component_property='value'), ])
def remaining_selection_drop_down(input, param_val):
    if input is None:
        return [
            html.Span('Remaining Selection Algorithms'),
            dcc.Dropdown(id='remaining-selection-dropdown',
                         options=remaining_selection_options,
                         value=None),
            html.Span('Parameter', style={'display': 'None'}),
            dcc.Input(id='remaining_pop_parameter', value='1', style={'display': 'None'}),
        ]
    elif input == 0:
        return [
            html.Span('Remaining Selection Algorithms'),
            dcc.Dropdown(id='remaining-selection-dropdown',
                         options=remaining_selection_options,
                         value=0),
            html.Span('Parameter', style={'display': 'None'}),
            dcc.Input(id='remaining_pop_parameter', value='1', style={'display': 'None'}),
        ]
    elif input == 1:
        return [
            html.Span('Remaining Selection Algorithms'),
            dcc.Dropdown(id='remaining-selection-dropdown',
                         options=remaining_selection_options,
                         value=1),
            html.Span('Parameter', style={'display': 'None'}),
            dcc.Input(id='remaining_pop_parameter', value='1', style={'display': 'None'}),
        ]
    elif input == 2:
        return [
            html.Span('Remaining Selection Algorithms'),
            dcc.Dropdown(id='remaining-selection-dropdown',
                         options=remaining_selection_options,
                         value=2),
            html.Span('Parameter'),
            dcc.Input(id='remaining_pop_parameter', value=param_val),
        ]
    elif input == 3:
        return [
            html.Span('Remaining Selection Algorithms'),
            dcc.Dropdown(id='remaining-selection-dropdown',
                         options=remaining_selection_options,
                         value=3),
            html.Span('Parameter', style={'display': 'None'}),
            dcc.Input(id='remaining_pop_parameter', value='1', style={'display': 'None'}),
        ]


@app.callback(Output(component_id='parent-selection-div', component_property='children'),
              [Input(component_id='parents-selection-dropdown', component_property='value')])
def parent_selection_drop_down(input):
    if input is None:
        return [
            html.Span('Parents Selection Algorithms'),
            dcc.Dropdown(id='parents-selection-dropdown',
                         options=parent_selection_options,
                         value=None),
        ]
    elif input == 0:
        return [
            html.Span('Parents Selection Algorithms'),
            dcc.Dropdown(id='parents-selection-dropdown',
                         options=parent_selection_options,
                         value=0),

        ]


@app.callback(Output(component_id='cross-over-div', component_property='children'),
              [Input(component_id='cross-over-dropdown', component_property='value'), ],
              [State(component_id='parents-probability', component_property='value'),
               State(component_id='cross-over-points-number', component_property='value'), ])
def cross_over_drop_down(input, parents_prob, cross_over_points):
    if input is None:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=None),
            html.Span('Probability of which parents', style={'display': 'None'}),
            dcc.Input(id='parents-probability', value=parents_prob, style={'display': 'None'}),
            html.Span('Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]
    elif input == 0:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=0),
            html.Span('Probability of which parents', style={'display': 'None'}),
            dcc.Input(id='parents-probability', value=parents_prob, style={'display': 'None'}),
            html.Span('Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]
    elif input == 1:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=1),
            html.Span('Probability of which parents'),
            dcc.Input(id='parents-probability', value=parents_prob),
            html.Span('Number of Points'),
            dcc.Input(id='cross-over-points-number', value=cross_over_points),
        ]
    elif input == 2:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=2),
            html.Span('Probability of which parents'),
            dcc.Input(id='parents-probability', value=parents_prob),
            html.Span('Number of Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]
    elif input == 3:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=3),
            html.Span('Probability of which parents', style={'display': 'None'}),
            dcc.Input(id='parents-probability', value=parents_prob, style={'display': 'None'}),
            html.Span('Number of Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]
    elif input == 4:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=4),
            html.Span('Probability of which parents', style={'display': 'None'}),
            dcc.Input(id='parents-probability', value=parents_prob, style={'display': 'None'}),
            html.Span('Number of Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]
    elif input == 5:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=5),
            html.Span('Probability of which parents', style={'display': 'None'}),
            dcc.Input(id='parents-probability', value=parents_prob, style={'display': 'None'}),
            html.Span('Number of Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]
    elif input == 6:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=6),
            html.Span('Probability of which parents', style={'display': 'None'}),
            dcc.Input(id='parents-probability', value=parents_prob, style={'display': 'None'}),
            html.Span('Number of Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]
    elif input == 7:
        return [
            html.Span('Cross over Algorithms'),
            dcc.Dropdown(id='cross-over-dropdown',
                         options=cross_over_options,
                         value=7),
            html.Span('Probability of which parents', style={'display': 'None'}),
            dcc.Input(id='parents-probability', value=parents_prob, style={'display': 'None'}),
            html.Span('Number of Points', style={'display': 'None'}),
            dcc.Input(id='cross-over-points-number', value=cross_over_points, style={'display': 'None'}),
        ]


@app.callback(Output(component_id='mutation-div', component_property='children'),
              [Input(component_id='mutation-dropdown', component_property='value'), ],
              [State(component_id='mutation-probability', component_property='value')])
def mutation_drop_down(input, mutation_prob):
    if input is None:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=None),
            html.Span('Probability', style={'display': 'None'}),
            dcc.Input(id='mutation-probability', value=mutation_prob, style={'display': 'None'}),
        ]
    elif input == 0:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=0),
            html.Span('Probability'),
            dcc.Input(id='mutation-probability', value=mutation_prob),
        ]
    elif input == 1:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=1),
            html.Span('Probability'),
            dcc.Input(id='mutation-probability', value=mutation_prob),
        ]
    elif input == 2:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=2),
            html.Span('Probability'),
            dcc.Input(id='mutation-probability', value=mutation_prob),
        ]
    elif input == 3:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=3),
            html.Span('Probability', style={'display': 'None'}),
            dcc.Input(id='mutation-probability', value=mutation_prob, style={'display': 'None'}),
        ]
    elif input == 4:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=4),
            html.Span('Probability', style={'display': 'None'}),
            dcc.Input(id='mutation-probability', value=mutation_prob, style={'display': 'None'}),
        ]
    elif input == 5:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=5),
            html.Span('Probability'),
            dcc.Input(id='mutation-probability', value=mutation_prob),
        ]
    elif input == 6:
        return [
            html.Span('Mutation Algorithms'),
            dcc.Dropdown(id='mutation-dropdown',
                         options=mutation_options,
                         value=6),
            html.Span('Probability'),
            dcc.Input(id='mutation-probability', value=mutation_prob),
        ]


@app.callback(Output(component_id='stop-condition-div', component_property='children'),
              [Input(component_id='stop-condition-dropdown', component_property='value')])
def stop_condition_drop_down(input):
    if input is None:
        return [
            html.Span('Stop condition'),
            dcc.Dropdown(id='stop-condition-dropdown',
                         options=stop_condition_options,
                         value=None),

        ]
    elif input == 0:
        return [
            html.Span('Stop condition'),
            dcc.Dropdown(id='stop-condition-dropdown',
                         options=stop_condition_options,
                         value=0),

        ]
    elif input == 1:
        return [
            html.Span('Stop condition'),
            dcc.Dropdown(id='stop-condition-dropdown',
                         options=stop_condition_options,
                         value=1),

        ]


@app.callback(
    Output(component_id='run-btn', component_property='children'),
    [Input(component_id='run-btn', component_property='n_clicks')],
    [State(component_id='remaining_pop_parameter', component_property='value'),
     State(component_id='name-input', component_property='value'),
     State(component_id='generation-input', component_property='value'),
     State(component_id='children-input', component_property='value'),
     State(component_id='population-input', component_property='value'),
     State(component_id='queen-number-input', component_property='value'),
     State(component_id='mutation-probability', component_property='value'),
     State(component_id='cross-over-points-number', component_property='value'),
     State(component_id='parents-probability', component_property='value'),
     State(component_id='mutation-dropdown', component_property='value'),
     State(component_id='cross-over-dropdown', component_property='value'),
     State(component_id='parents-selection-dropdown', component_property='value'),
     State(component_id='remaining-selection-dropdown', component_property='value'),
     State(component_id='stop-condition-dropdown', component_property='value'),
     ],
)
def run_btn(n_clicks,
            remai_pop_param,
            name,
            generation,
            children,
            population,
            queen_number,
            mutation_prob,
            cross_over_points,
            parents_prob,
            mutation_drop_down,
            cross_over_drop_down,
            parents_selection_drop_down,
            remaining_selection_drop_down,\
            stop_condition_dropdown):
    global avg_fitness_per_generation, variance_per_generation, best_chromosome, running
    if n_clicks > 0 and not running:
        mutation, cross_over, parents_selection, remaining_selection = None, None, None, None
        avg_fitness_per_generation = []
        variance_per_generation = []
        best_chromosome = [[0]]
        if name == '' or name is None:
            name = str(datetime.datetime.now())
        # mutation
        if mutation_drop_down == 0 or mutation_drop_down is None:
            mutation = (default_mutation, {'prob': float(mutation_prob)})
        elif mutation_drop_down == 1:
            mutation = (random_swap_mutation, {'prob': float(mutation_prob)})
        elif mutation_drop_down == 2:
            mutation = (shuffle_index_mutation, {'prob': float(mutation_prob)})
        elif mutation_drop_down == 3:
            mutation = (neighbour_based_mutation, None)
        elif mutation_drop_down == 4:
            mutation = (scramble_mutation, None)
        elif mutation_drop_down == 5:
            mutation = (insertion_swap_mutation, {'prob': float(mutation_prob)})
        elif mutation_drop_down == 6:
            mutation = (reverse_sequence_mutation, {'prob': float(mutation_prob)})

        # cross over
        if cross_over_drop_down == 0 or cross_over_drop_down is None:
            cross_over = (default_cross_over, {'prob': float(parents_prob)})
        elif cross_over_drop_down == 1:
            cross_over = (multi_points_crossover, {'prob': float(parents_prob), 'points_count': int(cross_over_points)})
        elif cross_over_drop_down == 2:
            cross_over = (upmx_crossover, {'prob': float(parents_prob)})
        elif cross_over_drop_down == 3:
            cross_over = (edge_crossover, None)
        elif cross_over_drop_down == 4:
            cross_over = (order_one_crossover, None)
        elif cross_over_drop_down == 5:
            cross_over = (masked_crossover, None)
        elif cross_over_drop_down == 6:
            cross_over = (maximal_preservation_crossover, None)
        elif cross_over_drop_down == 7:
            cross_over = (position_based_crossover, None)

        # parents selection
        if parents_selection_drop_down == 0 or parents_selection_drop_down is None:
            parents_selection = (default_parent_selection, None)

        # remaining selection
        if remaining_selection_drop_down == 0 or remaining_selection_drop_down is None:
            remaining_selection = (default_population_selection, None)
        elif remaining_selection_drop_down == 1:
            remaining_selection = (fitness_based_population_selection, None)
        elif remaining_selection_drop_down == 2:
            remaining_selection = (boltzmann_population_selection, {'T': remai_pop_param})
        elif remaining_selection_drop_down == 3:
            remaining_selection = (q_tornoment_based_population_selection, None)

        #stop Conditions
        stop_condition = ''
        if stop_condition_dropdown == 0 or stop_condition_dropdown == None:
            stop_condition = (default_stop_condition, {'max_generation':int(generation)})
        elif stop_condition_dropdown == 1:
            stop_condition = (evaluation_count_stop_condition, {'max_evaluation_count':int(generation)})

        ea = evolutionary_algorithms.EvolutionaryAlgorithm(
            mutation=mutation,
            cross_over=cross_over,
            parent_selection=parents_selection,
            remaining_population_selection=remaining_selection,
            evaluator=default_evaluator,
            gene_generator=permutation_random_gene_generator,
            stop_condition=stop_condition,
            max_generation=int(generation),
            n=int(queen_number),
            m=int(population),
            y=int(children),
        )
        print([('mutation', mutation),
               ('cross over', cross_over),
               ('remaining selection', remaining_selection),
               ('parent selection', parents_selection),
               ])
        running = True
        ea.run(name,
               variance_per_generation,
               avg_fitness_per_generation,
               best_chromosome)
        running = False
        # print(running)
    elif running:
        print('wait until current EA algorithm be finished!')
    return 'RUN'


@app.callback(
    Output(component_id='best-solution-graph', component_property='children'),
    [Input(component_id='interval', component_property='n_intervals'),
     Input(component_id='run-btn', component_property='n_clicks')],
)
def update_best_solution_graph(interval, n_clicks):
    global best_chromosome
    global interval_counter
    # print(n_clicks)
    if n_clicks == 0:
        return
    # print(best_chromosome)
    # print(np.array(best_chromosome[0]))
    fig = go.Figure(
        data=go.Heatmap(
            z=np.array(best_chromosome[0]),
            colorscale=[
                # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
                [0, "rgb(180, 200, 200)"],
                [0.6, "rgb(255, 0, 0)"],
                [1, "rgb(70, 255, 70)"]
            ],
            showscale=False,
        ))
    fig.update_layout(title='--------Best Chromosome',
                      )
    return dcc.Graph(
        id='fitness-graph-plot',
        figure=fig,
    )


@app.callback(
    Output(component_id='avg_graph', component_property='children'),
    [Input(component_id='interval', component_property='n_intervals'), ],
    [State(component_id='log-dropdown', component_property='value'),
     State(component_id='name-input', component_property='value'), ]
)
def update_fitness_graph(_, log_dropdown_value, name):
    global avg_fitness_per_generation
    global variance_per_generation
    global log_value
    global data_fit
    global data_var
    global y_var_max_value, y_avg_max_value, y_var_min_value, y_avg_min_value, x_max_value
    # print(len(data_fit))
    if name is None:
        name = 'Current'
    new_log = False
    if log_dropdown_value is None:
        log_dropdown_value = []
    # print(log_dropdown_value, log_value)
    if log_dropdown_value != log_value:
        log_value = log_dropdown_value
        # print(log_dropdown_value, log_value)
        new_log = True
    # print(running, new_log)
    # print(running)
    if not running and not new_log:
        # print('pass')
        pass
    else:
        # print('update')
        # print(avg_fitness_per_generation)
        data_fit = []
        data_var = []
        data_fit = [{'x': np.arange(0, len(avg_fitness_per_generation)),
                     'y': avg_fitness_per_generation,
                     'type': 'line',
                     'name': name + ' Avg. fitness'}]
        data_var = [{'x': np.arange(0, len(variance_per_generation)),
                     'y': variance_per_generation,
                     'type': 'line',
                     'name': name + ' Avg. variance'}]
        x_max_value = len(avg_fitness_per_generation) + 1

        if len(avg_fitness_per_generation) > 0:
            y_avg_max_value = np.max(avg_fitness_per_generation)
            y_var_max_value = np.max(variance_per_generation)
            y_avg_min_value = np.min(avg_fitness_per_generation)
            y_var_min_value = np.min(variance_per_generation)
        else:
            y_avg_max_value = 0
            y_var_max_value = 0
            y_avg_min_value = 0
            y_var_min_value = 0

        for i in log_dropdown_value:
            data_fit.append(
                {'x': np.arange(0, len(logs[i]['avg_fit'])),
                 'y': logs[i]['avg_fit'],
                 'type': 'line',
                 'name': logs[i]['name']}
            )
            data_var.append(
                {'x': np.arange(0, len(logs[i]['var_fit'])),
                 'y': logs[i]['var_fit'],
                 'type': 'line',
                 'name': logs[i]['name']}
            )
            if len(logs[i]['avg_fit']) > x_max_value:
                x_max_value = len(logs[i]['avg_fit']) + 1
            if y_var_max_value < np.max(logs[i]['var_fit']):
                y_var_max_value = np.max(logs[i]['var_fit'])
            if y_avg_max_value < np.max(logs[i]['avg_fit']):
                y_avg_max_value = np.max(logs[i]['avg_fit'])
            if y_var_min_value > np.min(logs[i]['var_fit']):
                y_var_min_value = np.min(logs[i]['var_fit'])
            if y_avg_min_value > np.min(logs[i]['avg_fit']):
                y_avg_min_value = np.min(logs[i]['avg_fit'])
    return [
        dcc.Graph(
            id='fitness-graph-plot',
            figure={
                'data': data_fit,
                'layout': {
                    'title': ' Average Fitness (AF) plot',
                    'xaxis': {
                        'title': 'Generation',
                        'range': [0, x_max_value],
                    },
                    'yaxis': {
                        'title': 'Average Fitness',
                        'range': [y_avg_min_value - 0.1 * y_avg_min_value, y_avg_max_value + 0.1 * y_avg_max_value],
                    }
                }
            },
        ),
        dcc.Graph(
            id='variance-graph-plot',
            figure={
                'data': data_var,
                'layout': {
                    'title': ' Variance Fitness (VF) plot',
                    'xaxis': {
                        'title': 'Generation',
                        'range': [0, x_max_value],
                    },
                    'yaxis': {
                        'title': 'Variance Fitness',
                        'range': [y_var_min_value - 0.1 * y_var_min_value, y_var_max_value + 0.1 * y_var_max_value],
                    }

                }
            },
        ),
    ]


@app.callback(
    Output(component_id='log-div', component_property='children'),
    [Input(component_id='interval', component_property='n_intervals'),
     Input(component_id='log-dropdown', component_property='value'), ]
)
def update_logs_data(_, value):
    read_logs()
    return [
        html.Span('select logs for show'),
        dcc.Dropdown(
            id='log-dropdown',
            options=drop_down_logs,
            value=value,
            multi=True
        ),
        html.Div(id='output-container')
        ,
    ]


if __name__ == '__main__':
    app.run_server(debug=True)
