import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.express as px


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        self.env_data = {}
        self.performance_data = {
            'episodes': [],
            'rewards': [],
            'coverages': [],
            'energies': [],
            'collision_counts': []
        }
        self.current_episode = 0

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Space-Air-Ground Intelligent Network (SAGIN) Visualization', style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='environment-graph'),
                    html.Div([
                        html.Label('Episode:'),
                        dcc.Slider(
                            id='episode-slider',
                            min=0,
                            max=self.config.num_episodes - 1,
                            value=0,
                            marks={i: str(i) for i in range(0, self.config.num_episodes, 10)},
                            step=1
                        ),
                        html.Label('Time Step:'),
                        dcc.Slider(
                            id='time-step-slider',
                            min=0,
                            max=self.config.max_time_steps - 1,
                            value=0,
                            marks={i: str(i) for i in range(0, self.config.max_time_steps, 20)},
                            step=1
                        )
                    ]),
                    html.Div(id='episode-info')
                ], style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'}),
                html.Div([
                    dcc.Graph(id='performance-graph'),
                    dcc.Graph(id='energy-graph'),
                    dcc.Graph(id='collision-graph')
                ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '0 20'})
            ]),
            dcc.Interval(
                id='interval-component',
                interval=1 * 1000,  # in milliseconds
                n_intervals=0
            )
        ])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('environment-graph', 'figure'),
             Output('episode-info', 'children')],
            [Input('episode-slider', 'value'),
             Input('time-step-slider', 'value')]
        )
        def update_environment_graph(episode, time_step):
            fig = self.create_environment_figure(episode, time_step)
            info = self.create_episode_info(episode, time_step)
            return fig, info

        @self.app.callback(
            [Output('performance-graph', 'figure'),
             Output('energy-graph', 'figure'),
             Output('collision-graph', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_graphs(n):
            return (self.create_performance_figure(),
                    self.create_energy_figure(),
                    self.create_collision_figure())

    def create_environment_figure(self, episode, time_step):
        if episode not in self.env_data or time_step not in self.env_data[episode]:
            return go.Figure()

        data = self.env_data[episode][time_step]

        fig = go.Figure()

        # Add PoIs
        fig.add_trace(go.Scatter(
            x=data['pois'][:, 0],
            y=data['pois'][:, 1],
            mode='markers',
            marker=dict(
                size=data['poi_priorities'] * 5,
                color=data['poi_priorities'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='PoI Priority')
            ),
            name='Points of Interest'
        ))

        # Add agents
        agent_colors = {'satellites': 'red', 'uavs': 'blue', 'ground_stations': 'green'}
        for agent_type in ['satellites', 'uavs', 'ground_stations']:
            fig.add_trace(go.Scatter(
                x=data[agent_type][:, 0],
                y=data[agent_type][:, 1],
                mode='markers+text',
                marker=dict(size=10, color=agent_colors[agent_type]),
                text=[f'{agent_type[:-1].capitalize()} {i}' for i in range(len(data[agent_type]))],
                name=agent_type.capitalize()
            ))

        # Add coverage
        fig.add_trace(go.Heatmap(
            z=data['coverage'],
            colorscale='RdYlBu',
            showscale=True,
            opacity=0.5,
            name='Coverage',
            colorbar=dict(title='Coverage Intensity')
        ))

        # Add communication links
        for link in data['communication_links']:
            fig.add_trace(go.Scatter(
                x=[link[0][0], link[1][0]],
                y=[link[0][1], link[1][1]],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=1),
                showlegend=False
            ))

        fig.update_layout(
            title=f'SAGIN Environment - Episode {episode}, Time Step {time_step}',
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            height=700,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def create_episode_info(self, episode, time_step):
        if episode not in self.env_data or time_step not in self.env_data[episode]:
            return html.Div("No data available")

        data = self.env_data[episode][time_step]

        return html.Div([
            html.H4(f'Episode {episode}, Time Step {time_step}'),
            html.P(f'Total Reward: {data["total_reward"]:.2f}'),
            html.P(f'Coverage: {data["coverage_percentage"]:.2f}%'),
            html.P(f'Average UAV Energy: {data["avg_uav_energy"]:.2f}'),
            html.P(f'Collisions: {data["collision_count"]}')
        ])

    def create_performance_figure(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.performance_data['episodes'],
            y=self.performance_data['rewards'],
            mode='lines',
            name='Reward'
        ))

        fig.add_trace(go.Scatter(
            x=self.performance_data['episodes'],
            y=self.performance_data['coverages'],
            mode='lines',
            name='Coverage'
        ))

        fig.update_layout(
            title='Training Performance',
            xaxis_title='Episode',
            yaxis_title='Value',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def create_energy_figure(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.performance_data['episodes'],
            y=self.performance_data['energies'],
            mode='lines',
            name='Avg UAV Energy'
        ))

        fig.update_layout(
            title='UAV Energy Over Time',
            xaxis_title='Episode',
            yaxis_title='Energy',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def create_collision_figure(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.performance_data['episodes'],
            y=self.performance_data['collision_counts'],
            mode='lines',
            name='Collisions'
        ))

        fig.update_layout(
            title='Collision Count Over Time',
            xaxis_title='Episode',
            yaxis_title='Number of Collisions',
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def update_env_data(self, env, episode, time_step):
        if episode not in self.env_data:
            self.env_data[episode] = {}

        coverage_map = self.calculate_coverage(env)

        self.env_data[episode][time_step] = {
            'pois': env.pois,
            'poi_priorities': env.poi_priorities,
            'satellites': env.satellites,
            'uavs': env.uavs,
            'ground_stations': env.ground_stations,
            'coverage': coverage_map,
            'coverage_percentage': np.mean(coverage_map) * 100,
            'communication_links': self.get_communication_links(env),
            'total_reward': env.total_reward,
            'avg_uav_energy': np.mean(env.agent_energy[env.num_satellites:env.num_satellites + env.num_uavs]),
            'collision_count': env.collision_count
        }

        self.current_episode = max(self.current_episode, episode)

    def calculate_coverage(self, env):
        coverage = np.zeros((self.config.area_size, self.config.area_size))
        x = np.arange(self.config.area_size)
        y = np.arange(self.config.area_size)
        xx, yy = np.meshgrid(x, y)

        for agent_type in ['satellites', 'uavs', 'ground_stations']:
            positions = getattr(env, agent_type)
            ranges = getattr(env, f'{agent_type[:-1]}_range')

            for pos in positions:
                dist = np.sqrt((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2)
                coverage += (dist <= ranges).astype(float)

        return coverage

    def get_communication_links(self, env):
        links = []
        all_agents = np.vstack((env.satellites, env.uavs, env.ground_stations))
        for i, agent1 in enumerate(all_agents):
            for j, agent2 in enumerate(all_agents[i + 1:], start=i + 1):
                dist = np.linalg.norm(agent1 - agent2)
                if dist <= max(env.satellite_range, env.uav_range, env.ground_station_range):
                    links.append((agent1, agent2))
        return links

    def update_performance_metrics(self, episode, reward, coverage, energy, collisions):
        self.performance_data['episodes'].append(episode)
        self.performance_data['rewards'].append(reward)
        self.performance_data['coverages'].append(coverage)
        self.performance_data['energies'].append(energy)
        self.performance_data['collision_counts'].append(collisions)

    def run(self, debug=False, port=8050):
        self.app.run_server(debug=debug, port=port)