import json
import os
from datetime import datetime

import dash
import numpy as np
import plotly.graph_objs as go
import torch
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots


class Visualizer:
    def __init__(self, config):
        """Initialize the SAGIN visualization system"""
        self.config = config
        self.app = Dash(__name__)

        # Data storage
        self.env_data = {}
        self.trajectories = {
            'satellites': [],
            'uavs': [],
            'ground_stations': []
        }
        self.communication_history = []
        self.current_episode = 0

        # Performance metrics storage
        self.performance_data = {
            'episodes': [],
            'rewards': [],
            'coverages': [],
            'energies': [],
            'collision_counts': []
        }

        # Create save directory
        self.save_dir = os.path.join(config.base_dir, "visualizations")
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize visualization
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the Dash application layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1('Space-Air-Ground Intelligent Network Visualization',
                        style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div([
                    html.Button('Switch to 2D View', id='view-toggle', n_clicks=0),
                    html.Button('Export Data', id='export-button', n_clicks=0),
                ], style={'textAlign': 'center', 'marginBottom': '20px'})
            ], style={'marginBottom': '30px'}),

            # Main content container
            html.Div([
                # Left panel - Environment visualization
                html.Div([
                    # Environment map
                    html.Div([
                        dcc.Graph(id='environment-plot', style={'height': '60vh'}),
                        html.Div([
                            # Visualization controls
                            html.Div([
                                html.Div([
                                    html.Label('Show Trajectories:',
                                               style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                    dcc.Checklist(
                                        id='trajectory-toggle',
                                        options=[
                                            {'label': ' Satellites', 'value': 'satellites'},
                                            {'label': ' UAVs', 'value': 'uavs'},
                                            {'label': ' Ground Stations', 'value': 'ground_stations'}
                                        ],
                                        value=['satellites', 'uavs', 'ground_stations'],
                                        style={'lineHeight': '1.5'}
                                    )
                                ], style={'display': 'inline-block', 'marginRight': '40px'}),

                                html.Div([
                                    html.Label('Show Elements:',
                                               style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                    dcc.Checklist(
                                        id='element-toggle',
                                        options=[
                                            {'label': ' Coverage Map', 'value': 'coverage'},
                                            {'label': ' Communication Links', 'value': 'communication'},
                                            {'label': ' POIs', 'value': 'pois'}
                                        ],
                                        value=['coverage', 'communication', 'pois'],
                                        style={'lineHeight': '1.5'}
                                    )
                                ], style={'display': 'inline-block'})
                            ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                        ], style={'marginBottom': '20px'})
                    ], style={'marginBottom': '20px'}),

                    # Time controls
                    html.Div([
                        html.Div([
                            html.Label('Episode:', style={'fontWeight': 'bold'}),
                            dcc.Slider(
                                id='episode-slider',
                                min=0,
                                max=self.config.num_episodes - 1,
                                value=0,
                                marks={i: str(i) for i in range(0, self.config.num_episodes, 10)},
                                step=1,
                                updatemode='drag'
                            )
                        ], style={'marginBottom': '20px'}),

                        html.Div([
                            html.Label('Timestep:', style={'fontWeight': 'bold'}),
                            dcc.Slider(
                                id='timestep-slider',
                                min=0,
                                max=self.config.max_time_steps - 1,
                                value=0,
                                marks={i: str(i) for i in range(0, self.config.max_time_steps, 10)},
                                step=1,
                                updatemode='drag'
                            )
                        ], style={'marginBottom': '15px'}),

                        # Playback controls
                        html.Div([
                            html.Button('⏪', id='rewind-button', n_clicks=0,
                                        style={'marginRight': '10px'}),
                            html.Button('▶️', id='play-button', n_clicks=0,
                                        style={'marginRight': '10px'}),
                            html.Button('⏸️', id='pause-button', n_clicks=0,
                                        style={'marginRight': '10px'}),
                            html.Button('⏩', id='forward-button', n_clicks=0),
                        ], style={'textAlign': 'center', 'marginBottom': '15px'}),

                        dcc.Interval(id='animation-interval', interval=500, disabled=True)
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px',
                              'marginBottom': '20px'}),

                    # Episode information
                    html.Div(id='episode-info',
                             style={'padding': '20px', 'backgroundColor': '#f8f9fa',
                                    'borderRadius': '5px'})
                ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'paddingRight': '20px'}),

                # Right panel - Performance metrics
                html.Div([
                    dcc.Graph(id='reward-plot', style={'height': '22vh'}),
                    dcc.Graph(id='coverage-plot', style={'height': '22vh'}),
                    dcc.Graph(id='energy-plot', style={'height': '22vh'}),
                    dcc.Graph(id='collision-plot', style={'height': '22vh'})
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'margin': '0 20px'}),

            # Update interval
            dcc.Interval(id='interval-component', interval=2000, n_intervals=0),

            # Store for 3D/2D view state
            dcc.Store(id='view-state', data={'is_3d': False})
        ])

    def setup_callbacks(self):
        """Setup Dash callbacks for interactive visualization"""

        @self.app.callback(
            [Output('environment-plot', 'figure'),
             Output('episode-info', 'children')],
            [Input('episode-slider', 'value'),
             Input('timestep-slider', 'value'),
             Input('trajectory-toggle', 'value'),
             Input('element-toggle', 'value'),
             Input('view-state', 'data')]
        )
        def update_environment_plot(episode, timestep, show_trajectories, show_elements, view_state):
            if view_state.get('is_3d', False):
                fig = self.create_3d_visualization(episode, timestep, show_trajectories, show_elements)
            else:
                fig = self.create_environment_figure(episode, timestep, show_trajectories, show_elements)
            return fig, self.create_episode_info(episode, timestep)

        @self.app.callback(
            [Output('reward-plot', 'figure'),
             Output('coverage-plot', 'figure'),
             Output('energy-plot', 'figure'),
             Output('collision-plot', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metric_plots(_):
            return (
                self.create_metric_figure('rewards', 'Training Rewards'),
                self.create_metric_figure('coverages', 'Coverage Rate'),
                self.create_metric_figure('energies', 'Average Energy'),
                self.create_metric_figure('collision_counts', 'Collision Count')
            )

        @self.app.callback(
            Output('animation-interval', 'disabled'),
            [Input('play-button', 'n_clicks'),
             Input('pause-button', 'n_clicks')]
        )
        def toggle_animation(play_clicks, pause_clicks):
            if play_clicks > pause_clicks:
                return False
            return True

        @self.app.callback(
            Output('timestep-slider', 'value'),
            [Input('animation-interval', 'n_intervals'),
             Input('timestep-slider', 'value'),
             Input('rewind-button', 'n_clicks'),
             Input('forward-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def update_timestep(_, current_timestep, rewind_clicks, forward_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_timestep

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'rewind-button':
                return max(0, current_timestep - 5)
            elif trigger_id == 'forward-button':
                return min(self.config.max_time_steps - 1, current_timestep + 5)
            elif trigger_id == 'animation-interval':
                if current_timestep < self.config.max_time_steps - 1:
                    return current_timestep + 1
                return 0

            return current_timestep

        @self.app.callback(
            Output('view-state', 'data'),
            [Input('view-toggle', 'n_clicks')],
            prevent_initial_call=True
        )
        def toggle_view(n_clicks):
            if n_clicks % 2 == 0:
                return {'is_3d': False}
            return {'is_3d': True}

    def create_environment_figure(self, episode, timestep, show_trajectories, show_elements):
        """Create detailed environment visualization with trajectories and coverage"""
        try:
            if episode not in self.env_data or timestep not in self.env_data[episode]:
                return go.Figure()

            data = self.env_data[episode][timestep]
            fig = go.Figure()

            # Add coverage heatmap if enabled
            if 'coverage' in show_elements:
                fig.add_trace(go.Heatmap(
                    z=data['coverage'],
                    colorscale='RdYlBu',
                    showscale=True,
                    opacity=0.5,
                    name='Coverage',
                    hoverongaps=False,
                    colorbar=dict(
                        title='Coverage Intensity',
                        titleside='right',
                        titlefont=dict(size=12)
                    )
                ))

            # Add POIs if enabled
            if 'pois' in show_elements:
                fig.add_trace(go.Scatter(
                    x=data['pois'][:, 0],
                    y=data['pois'][:, 1],
                    mode='markers',
                    marker=dict(
                        size=data['poi_priorities'] * 5,
                        color=data['poi_priorities'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title='POI Priority',
                            titleside='right',
                            titlefont=dict(size=12),
                            x=1.1
                        )
                    ),
                    name='Points of Interest',
                    hovertemplate='POI<br>Priority: %{marker.color:.1f}<br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
                ))

            # Add agents and their trajectories
            agent_types = {
                'satellites': {'color': 'red', 'symbol': 'circle', 'size': 12, 'name': 'Satellite'},
                'uavs': {'color': 'blue', 'symbol': 'triangle-up', 'size': 10, 'name': 'UAV'},
                'ground_stations': {'color': 'green', 'symbol': 'square', 'size': 8, 'name': 'Ground Station'}
            }

            for agent_type, style in agent_types.items():
                if agent_type in show_trajectories:
                    # Add coverage radius
                    for i, pos in enumerate(data[agent_type]):
                        range_val = getattr(self.config, f'{agent_type[:-1]}_range')
                        fig.add_shape(
                            type="circle",
                            xref="x", yref="y",
                            x0=pos[0] - range_val,
                            y0=pos[1] - range_val,
                            x1=pos[0] + range_val,
                            y1=pos[1] + range_val,
                            line_color=style['color'],
                            line_dash="dash",
                            opacity=0.3,
                            layer="below"
                        )

                    # Current positions
                    fig.add_trace(go.Scatter(
                        x=data[agent_type][:, 0],
                        y=data[agent_type][:, 1],
                        mode='markers+text',
                        marker=dict(
                            size=style['size'],
                            color=style['color'],
                            symbol=style['symbol'],
                            line=dict(width=1, color='white')
                        ),
                        text=[f'{style["name"]} {i}' for i in range(len(data[agent_type]))],
                        name=style['name'],
                        hovertemplate=style['name'] + ' %{text}<br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
                    ))

                    # Trajectories
                    if len(self.trajectories[agent_type]) > 1:
                        for i in range(len(data[agent_type])):
                            trajectory = np.array([
                                positions[i]
                                for positions in self.trajectories[agent_type][-20:]
                            ])
                            fig.add_trace(go.Scatter(
                                x=trajectory[:, 0],
                                y=trajectory[:, 1],
                                mode='lines',
                                line=dict(
                                    color=style['color'],
                                    width=1,
                                    dash='dot'
                                ),
                                opacity=0.5,
                                showlegend=False,
                                hoverinfo='skip'
                            ))

            # Add communication links if enabled
            if 'communication' in show_elements:
                for link in data['communication_links']:
                    pos1, pos2 = link['points']
                    signal_strength = link['signal_strength']
                    fig.add_trace(go.Scatter(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        mode='lines',
                        line=dict(
                            color=f'rgba(100, 100, 100, {signal_strength})',
                            width=2
                        ),
                        name='Communication Link',
                        hovertemplate='Signal Strength: %{text:.2f}<extra></extra>',
                        text=[signal_strength],
                        showlegend=False
                    ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'SAGIN Environment - Episode {episode}, Timestep {timestep}',
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title='X coordinate',
                    range=[0, self.config.area_size],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)'
                ),
                yaxis=dict(
                    title='Y coordinate',
                    range=[0, self.config.area_size],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    scaleanchor='x',
                    scaleratio=1
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                hovermode='closest',
                margin=dict(l=60, r=60, t=60, b=60),
                plot_bgcolor='white'
            )

            return fig

        except Exception as e:
            print(f"Warning: Error in creating environment figure: {str(e)}")
            return go.Figure()
    def create_3d_visualization(self, episode, timestep, show_trajectories, show_elements):
        """Create detailed 3D environment visualization"""
        try:
            if episode not in self.env_data or timestep not in self.env_data[episode]:
                return go.Figure()

            data = self.env_data[episode][timestep]
            fig = go.Figure()

            # Add ground plane with coverage heatmap if enabled
            if 'coverage' in show_elements:
                fig.add_trace(go.Surface(
                    z=np.zeros_like(data['coverage']),
                    surfacecolor=data['coverage'],
                    colorscale='RdYlBu',
                    showscale=True,
                    opacity=0.8,
                    name='Coverage',
                    colorbar=dict(title='Coverage Intensity')
                ))

            # Add POIs if enabled
            if 'pois' in show_elements:
                fig.add_trace(go.Scatter3d(
                    x=data['pois'][:, 0],
                    y=data['pois'][:, 1],
                    z=np.zeros(len(data['pois'])),
                    mode='markers',
                    marker=dict(
                        size=data['poi_priorities'] * 5,
                        color=data['poi_priorities'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='POI Priority', x=1.1)
                    ),
                    name='Points of Interest'
                ))

            # Define agent heights and styles
            agent_heights = {
                'satellites': 100,
                'uavs': 50,
                'ground_stations': 0
            }

            agent_styles = {
                'satellites': {'color': 'red', 'size': 8, 'name': 'Satellite'},
                'uavs': {'color': 'blue', 'size': 6, 'name': 'UAV'},
                'ground_stations': {'color': 'green', 'size': 4, 'name': 'Ground Station'}
            }

            # Add agents and their trajectories
            for agent_type in agent_styles:
                if agent_type in show_trajectories:
                    style = agent_styles[agent_type]
                    height = agent_heights[agent_type]
                    positions = data[agent_type]

                    # Current positions
                    fig.add_trace(go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=np.full(len(positions), height),
                        mode='markers',
                        marker=dict(
                            size=style['size'],
                            color=style['color'],
                            symbol='circle'
                        ),
                        name=style['name'],
                        text=[f'{style["name"]} {i}' for i in range(len(positions))],
                        hovertemplate='%{text}<br>Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>'
                    ))

                    # Trajectories
                    if len(self.trajectories[agent_type]) > 1:
                        for i in range(len(positions)):
                            trajectory = np.array([
                                positions[i]
                                for positions in self.trajectories[agent_type][-20:]
                            ])
                            fig.add_trace(go.Scatter3d(
                                x=trajectory[:, 0],
                                y=trajectory[:, 1],
                                z=np.full(len(trajectory), height),
                                mode='lines',
                                line=dict(
                                    color=style['color'],
                                    width=2,
                                    dash='dot'
                                ),
                                opacity=0.5,
                                showlegend=False
                            ))

            # Add communication links if enabled
            if 'communication' in show_elements:
                for link in data['communication_links']:
                    pos1, pos2 = link['points']
                    agent1, agent2 = link['agents']
                    h1 = agent_heights[agent1['type']]
                    h2 = agent_heights[agent2['type']]
                    signal_strength = link['signal_strength']

                    fig.add_trace(go.Scatter3d(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        z=[h1, h2],
                        mode='lines',
                        line=dict(
                            color=f'rgba(100, 100, 100, {signal_strength})',
                            width=2
                        ),
                        opacity=0.7,
                        showlegend=False,
                        hovertemplate=f'Signal Strength: {signal_strength:.2f}<extra></extra>'
                    ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'3D SAGIN Environment - Episode {episode}, Timestep {timestep}',
                    font=dict(size=16)
                ),
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Altitude',
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2),
                        center=dict(x=0, y=0, z=0)
                    ),
                    xaxis=dict(range=[0, self.config.area_size]),
                    yaxis=dict(range=[0, self.config.area_size]),
                    zaxis=dict(range=[0, 120])
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            return fig

        except Exception as e:
            print(f"Warning: Error in creating 3D visualization: {str(e)}")
            return go.Figure()

    def create_metric_figure(self, metric_name, title):
        """Create performance metric figure with trend analysis"""
        try:
            fig = go.Figure()

            if self.performance_data['episodes']:
                # Add main metric line
                fig.add_trace(go.Scatter(
                    x=self.performance_data['episodes'],
                    y=self.performance_data[metric_name],
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(
                        color='blue',
                        width=2
                    ),
                    marker=dict(
                        size=6,
                        color='blue'
                    )
                ))

                # Add moving average
                window = min(10, len(self.performance_data[metric_name]))
                if len(self.performance_data[metric_name]) >= window:
                    moving_avg = np.convolve(
                        self.performance_data[metric_name],
                        np.ones(window)/window,
                        mode='valid'
                    )
                    fig.add_trace(go.Scatter(
                        x=self.performance_data['episodes'][window-1:],
                        y=moving_avg,
                        mode='lines',
                        name=f'{window}-Episode Average',
                        line=dict(
                            color='red',
                            width=2,
                            dash='dash'
                        )
                    ))

                # Add trend line
                z = np.polyfit(self.performance_data['episodes'],
                             self.performance_data[metric_name], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=self.performance_data['episodes'],
                    y=p(self.performance_data['episodes']),
                    mode='lines',
                    name='Trend',
                    line=dict(
                        color='green',
                        width=1,
                        dash='dot'
                    )
                ))

            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=14)
                ),
                xaxis=dict(
                    title='Episode',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    zeroline=False
                ),
                yaxis=dict(
                    title=metric_name.capitalize(),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    zeroline=False
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                margin=dict(l=60, r=20, t=40, b=40)
            )

            return fig

        except Exception as e:
            print(f"Warning: Error in creating metric figure: {str(e)}")
            return go.Figure()

    def create_episode_info(self, episode, timestep):
        """Create detailed episode information display"""
        try:
            if episode not in self.env_data or timestep not in self.env_data[episode]:
                return html.Div("No data available")

            data = self.env_data[episode][timestep]

            # Calculate additional metrics
            active_links = len(data['communication_links'])
            coverage_percentage = data['coverage_percentage']
            energy_levels = np.array([link['signal_strength']
                                    for link in data['communication_links']])
            avg_signal_strength = np.mean(energy_levels) if len(energy_levels) > 0 else 0

            info_style = {
                'padding': '10px',
                'backgroundColor': 'white',
                'borderRadius': '5px',
                'marginBottom': '10px'
            }

            return html.Div([
                html.H4(f'Episode {episode}, Timestep {timestep}',
                       style={'marginBottom': '15px', 'color': '#2c3e50'}),

                # Performance metrics
                html.Div([
                    html.H5('Performance Metrics',
                           style={'color': '#2980b9', 'marginBottom': '10px'}),
                    html.Div([
                        html.P([
                            html.Strong('Total Reward: '),
                            f'{data["total_reward"]:.2f}'
                        ]),
                        html.P([
                            html.Strong('Coverage: '),
                            f'{coverage_percentage:.2f}%'
                        ]),
                        html.P([
                            html.Strong('Average UAV Energy: '),
                            f'{data["avg_uav_energy"]:.2f}'
                        ])
                    ])
                ], style=info_style),

                # Network metrics
                html.Div([
                    html.H5('Network Statistics',
                           style={'color': '#27ae60', 'marginBottom': '10px'}),
                    html.Div([
                        html.P([
                            html.Strong('Active Links: '),
                            str(active_links)
                        ]),
                        html.P([
                            html.Strong('Avg Signal Strength: '),
                            f'{avg_signal_strength:.2f}'
                        ]),
                        html.P([
                            html.Strong('Collisions: '),
                            str(data["collision_count"])
                        ])
                    ])
                ], style=info_style),

                # Agent status
                html.Div([
                    html.H5('Agent Status',
                           style={'color': '#c0392b', 'marginBottom': '10px'}),
                    html.Div([
                        html.P([
                            html.Strong('Satellites: '),
                            str(len(data["satellites"]))
                        ]),
                        html.P([
                            html.Strong('UAVs: '),
                            str(len(data["uavs"]))
                        ]),
                        html.P([
                            html.Strong('Ground Stations: '),
                            str(len(data["ground_stations"]))
                        ])
                    ])
                ], style=info_style)
            ])

        except Exception as e:
            print(f"Warning: Error in creating episode info: {str(e)}")
            return html.Div("Error displaying information")

    def update_env_data(self, env, episode, timestep):
        """Update environment data including trajectories"""
        try:
            if episode not in self.env_data:
                self.env_data[episode] = {}

            # Calculate coverage
            coverage_map = self.calculate_coverage(env)

            # Update current positions and data
            current_data = {
                'pois': env.pois.astype(np.float32),
                'poi_priorities': env.poi_priorities.astype(np.float32),
                'satellites': env.satellites.astype(np.float32),
                'uavs': env.uavs.astype(np.float32),
                'ground_stations': env.ground_stations.astype(np.float32),
                'coverage': coverage_map,
                'coverage_percentage': float(np.mean(coverage_map) * 100),
                'communication_links': self.get_communication_links(env),
                'total_reward': float(env.total_reward),
                'avg_uav_energy': float(np.mean(
                    env.agent_energy[env.num_satellites:env.num_satellites + env.num_uavs]
                )),
                'collision_count': int(env.collision_count)
            }

            self.env_data[episode][timestep] = current_data

            # Update trajectories
            for agent_type in ['satellites', 'uavs', 'ground_stations']:
                positions = getattr(env, agent_type)
                self.trajectories[agent_type].append(positions.copy())

            # Limit trajectory history
            max_history = 100
            for agent_type in self.trajectories:
                if len(self.trajectories[agent_type]) > max_history:
                    self.trajectories[agent_type] = self.trajectories[agent_type][-max_history:]

            self.current_episode = max(self.current_episode, episode)

            # Update performance metrics
            if timestep == 0:  # Only update at the start of each episode
                self.performance_data['episodes'].append(episode)
                self.performance_data['rewards'].append(float(env.total_reward))
                self.performance_data['coverages'].append(float(np.mean(coverage_map) * 100))
                self.performance_data['energies'].append(float(np.mean(
                    env.agent_energy[env.num_satellites:env.num_satellites + env.num_uavs]
                )))
                self.performance_data['collision_counts'].append(int(env.collision_count))

            # Cleanup old episodes
            self._cleanup_old_data(episode)

        except Exception as e:
            print(f"Warning: Error in update_env_data: {str(e)}")

    def calculate_coverage(self, env):
        """Calculate coverage map efficiently using chunked processing"""
        try:
            coverage = np.zeros((self.config.area_size, self.config.area_size),
                              dtype=np.float32)

            x = np.arange(self.config.area_size, dtype=np.float32)
            y = np.arange(self.config.area_size, dtype=np.float32)

            chunk_size = 50
            for i in range(0, self.config.area_size, chunk_size):
                i_end = min(i + chunk_size, self.config.area_size)
                for j in range(0, self.config.area_size, chunk_size):
                    j_end = min(j + chunk_size, self.config.area_size)

                    xx, yy = np.meshgrid(x[i:i_end], y[j:j_end])

                    # Process each agent type with their respective ranges
                    agent_ranges = {
                        'satellites': (env.satellites, env.satellite_range),
                        'uavs': (env.uavs, env.uav_range),
                        'ground_stations': (env.ground_stations, env.ground_station_range)
                    }

                    for positions, range_val in agent_ranges.values():
                        for pos in positions:
                            dist = np.sqrt((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2)
                            coverage[j:j_end, i:i_end] += (dist <= range_val).astype(np.float32)

                    del xx, yy

            return coverage

        except Exception as e:
            print(f"Warning: Error in coverage calculation: {str(e)}")
            return np.zeros((self.config.area_size, self.config.area_size), dtype=np.float32)

    def get_communication_links(self, env):
        """Calculate communication links between agents with signal strength"""
        try:
            links = []
            comm_range = self.config.communication_range

            # Pre-calculate agent types and positions
            agents = []
            for agent_type in ['satellites', 'uavs', 'ground_stations']:
                positions = getattr(env, agent_type)
                for i, pos in enumerate(positions):
                    agents.append({
                        'type': agent_type,
                        'id': i,
                        'position': pos,
                        'range': getattr(env, f'{agent_type[:-1]}_range')
                    })

            # Calculate links between agents using vectorized operations
            positions = np.array([agent['position'] for agent in agents])
            distances = np.linalg.norm(positions[:, None] - positions, axis=2)

            # Process valid links
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    distance = distances[i, j]
                    if distance <= comm_range:
                        # Calculate signal strength with distance and agent types
                        base_strength = 1.0 - (distance / comm_range)

                        # Apply type-specific modifiers
                        type_modifiers = {
                            'satellites': 1.2,  # Better signal for satellite links
                            'uavs': 1.0,  # Standard signal for UAVs
                            'ground_stations': 0.8  # Slightly weaker for ground stations
                        }

                        modifier = np.mean([
                            type_modifiers[agents[i]['type']],
                            type_modifiers[agents[j]['type']]
                        ])

                        signal_strength = min(1.0, base_strength * modifier)

                        links.append({
                            'points': (agents[i]['position'], agents[j]['position']),
                            'agents': (agents[i], agents[j]),
                            'distance': distance,
                            'signal_strength': signal_strength,
                            'type': f"{agents[i]['type']}-{agents[j]['type']}"
                        })

            return links

        except Exception as e:
            print(f"Warning: Error in communication links calculation: {str(e)}")
            return []

    def _cleanup_old_data(self, current_episode, keep_episodes=5):
        """Clean up old episode data to manage memory"""
        try:
            if len(self.env_data) > keep_episodes * 2:  # Allow some buffer
                episodes_to_remove = []
                sorted_episodes = sorted(self.env_data.keys())

                for episode in sorted_episodes[:-keep_episodes]:
                    if episode < current_episode - keep_episodes:
                        episodes_to_remove.append(episode)

                # Remove old episodes
                for episode in episodes_to_remove:
                    del self.env_data[episode]

                # Clear trajectories if they're getting too long
                max_trajectory_length = self.config.max_time_steps * 2
                for agent_type in self.trajectories:
                    if len(self.trajectories[agent_type]) > max_trajectory_length:
                        self.trajectories[agent_type] = \
                            self.trajectories[agent_type][-max_trajectory_length:]

                # Force garbage collection after major cleanup
                import gc
                gc.collect()

                # Clear GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return True

        except Exception as e:
            print(f"Warning: Error in cleaning up old data: {str(e)}")
            return False

    def save_visualization_state(self, output_path=None):
        """Save current visualization state to file"""
        try:
            if output_path is None:
                output_path = os.path.join(
                    self.save_dir,
                    f"visualization_state_ep_{self.current_episode}.json"
                )

            # Prepare data for saving
            save_data = {
                'current_episode': self.current_episode,
                'performance_data': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.performance_data.items()
                },
                'config': {
                    'area_size': self.config.area_size,
                    'num_agents': self.config.num_agents,
                    'max_time_steps': self.config.max_time_steps,
                    'communication_range': self.config.communication_range
                },
                'timestamp': datetime.now().isoformat()
            }

            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=4)

            return output_path

        except Exception as e:
            print(f"Warning: Error in saving visualization state: {str(e)}")
            return None

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            report = {
                'overall_metrics': {},
                'agent_performance': {},
                'network_metrics': {},
                'mission_success': {}
            }

            # Calculate overall metrics
            for metric in ['rewards', 'coverages', 'energies', 'collision_counts']:
                if self.performance_data[metric]:
                    values = np.array(self.performance_data[metric])
                    report['overall_metrics'][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'latest': float(values[-1]),
                        'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'decreasing'
                    }

            # Calculate agent-specific metrics if available
            if self.trajectories['satellites']:
                for agent_type in ['satellites', 'uavs', 'ground_stations']:
                    movement_data = np.array(self.trajectories[agent_type])
                    distances = np.linalg.norm(
                        movement_data[1:] - movement_data[:-1],
                        axis=2
                    )
                    report['agent_performance'][agent_type] = {
                        'total_distance': float(np.sum(distances)),
                        'avg_speed': float(np.mean(distances)),
                        'movement_pattern': self._analyze_movement_pattern(distances)
                    }

            # Calculate network metrics
            if self.env_data and self.current_episode in self.env_data:
                latest_data = self.env_data[self.current_episode]
                latest_timestep = max(latest_data.keys())
                links = latest_data[latest_timestep]['communication_links']

                report['network_metrics'] = {
                    'active_links': len(links),
                    'avg_signal_strength': float(np.mean([
                        link['signal_strength'] for link in links
                    ])) if links else 0,
                    'network_density': len(links) / (self.config.num_agents * (self.config.num_agents - 1) / 2)
                }

            # Calculate mission success metrics
            report['mission_success'] = {
                'completion_rate': float(np.mean(
                    np.array(self.performance_data['coverages']) > 80
                )) if self.performance_data['coverages'] else 0,
                'efficiency': self._calculate_mission_efficiency(),
                'stability': self._calculate_system_stability()
            }

            return report

        except Exception as e:
            print(f"Warning: Error in generating performance report: {str(e)}")
            return None

    def _analyze_movement_pattern(self, distances):
        """Analyze agent movement patterns"""
        try:
            pattern = {
                'regular': bool(np.std(distances) < np.mean(distances) * 0.2),
                'bursty': bool(np.max(distances) > np.mean(distances) * 3),
                'stationary_periods': bool(np.sum(distances < 0.1) > len(distances) * 0.3)
            }
            return pattern
        except:
            return {'regular': False, 'bursty': False, 'stationary_periods': False}

    def _calculate_mission_efficiency(self):
        """Calculate overall mission efficiency"""
        try:
            if not self.performance_data['rewards'] or not self.performance_data['energies']:
                return 0.0

            recent_rewards = np.array(self.performance_data['rewards'][-10:])
            recent_energy = np.array(self.performance_data['energies'][-10:])

            return float(np.mean(recent_rewards) / (np.mean(recent_energy) + 1e-6))
        except:
            return 0.0

    def _calculate_system_stability(self):
        """Calculate system stability metrics"""
        try:
            if not self.performance_data['coverages']:
                return 0.0

            coverage_values = np.array(self.performance_data['coverages'])
            stability = 1.0 - (np.std(coverage_values) / (np.mean(coverage_values) + 1e-6))
            return float(max(0.0, min(1.0, stability)))
        except:
            return 0.0

    def run(self, debug=False, port=8050):
        """Run the visualization server with complete setup"""
        try:
            print("\nStarting SAGIN Visualization Server")
            print("===================================")
            print(f"Configuration:")
            print(f"- Area Size: {self.config.area_size}")
            print(f"- Number of Agents: {self.config.num_agents}")
            print(f"- Max Time Steps: {self.config.max_time_steps}")
            print(f"- Port: {port}")
            print("\nVisualization Features:")
            print("- 2D and 3D environment views")
            print("- Real-time performance metrics")
            print("- Agent trajectory visualization")
            print("- Network connectivity analysis")
            print("- Interactive controls")
            print("\nAccess the visualization at:")
            print(f"http://localhost:{port}")
            print("===================================\n")

            self.app.run_server(debug=debug, port=port)

        except Exception as e:
            print(f"Error starting visualization server: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup visualization resources"""
        try:
            # Clear stored data
            self.env_data.clear()
            self.trajectories.clear()
            for key in self.performance_data:
                self.performance_data[key].clear()

            # Force garbage collection
            import gc
            gc.collect()

            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Error in cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

