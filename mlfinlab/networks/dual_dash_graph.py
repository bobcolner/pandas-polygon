# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
This class takes in a Graph object and creates interactive visualisations using Plotly's Dash.
The DualDashGraph class contains private functions used to generate the frontend components needed to create the UI.

Running run_server() will produce the warning "Warning: This is a development server. Do not use app.run_server
in production, use a production WSGI server like gunicorn instead.".
However, this is okay and the Dash server will run without a problem.
"""

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash

class DualDashGraph:
    """
    The DualDashGraph class is the inerface for comparing and highlighting the difference between two graphs.
    Two Graph class objects should be supplied - such as MST and ALMST graphs.
    """

    def __init__(self, graph_one, graph_two, app_display='default'):
        """
        Initialises the dual graph interface and generates the interface layout.

        :param graph_one: (Graph) The first graph for the comparison interface.
        :param graph_two: (Graph) The second graph for the comparison interface.
        :param app_display: (str) 'default' by default and 'jupyter notebook' for running Dash inside Jupyter Notebook.
        """

        # Dash app styling with Bootstrap
        if app_display == 'jupyter notebook':
            self.app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        else:
            self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Setting input graphs as objest variables
        cyto.load_extra_layouts()
        self.graph_one = graph_one
        self.graph_two = graph_two

        # Getting a list of tuples with differnet edge connections
        difference = graph_one.get_difference(graph_two)

        # Updating the elements needed for the Dash Cytoscape Graph object
        self.one_components = None
        self.two_components = None
        self._update_elements_dual(self.graph_one, difference, 1)
        self._update_elements_dual(self.graph_two, difference, 2)

        self.cyto_one = None
        self.cyto_two = None

        # Callback functions to allow simultaneous node selection when clicked
        self.app.callback(Output('cytoscape_two', 'elements'),
                          [Input('cytoscape', 'tapNode')],
                          [State('cytoscape_two', 'elements')]
                          )(DualDashGraph._select_other_graph_node)
        self.app.callback(Output('cytoscape', 'elements'),
                          [Input('cytoscape_two', 'tapNode')],
                          [State('cytoscape', 'elements')]
                          )(DualDashGraph._select_other_graph_node)

    @staticmethod
    def _select_other_graph_node(data, elements):
        """
        Callback function to select the other graph node when a graph node
        is selected by setting selected to True.

        :param data: (Dict) Dictionary of "tapped" or selected node.
        :param elements: (Dict) Dictionary of elements.
        :return: (Dict) Returns updates dictionary of elements.
        """
        if data:
            for element in elements:
                element['selected'] = (data['data']['id'] == element.get('data').get('id'))

        return elements

    def _generate_comparison_layout(self, graph_one, graph_two):
        """
        Returns and generates a dual comparison layout.

        :param graph_one: (Graph) The first graph object for the dual interface.
        :param graph_two: (Graph) Comparison graph object for the dual interface.
        :return: (html.Div) Returns a Div containing the interface.
        """
        # Set Graph names
        graph_one_name = type(graph_one).__name__
        graph_two_name = type(graph_two).__name__

        # Set the cyto graphs
        self._set_cyto_graph()

        # Get different edges between two graphs
        difference = graph_one.get_difference(graph_two)

        # Layout components
        padding = {'padding': '10px 10px 10px 10px'}
        cards = dbc.CardDeck(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(graph_one_name),
                        dbc.CardBody(self.cyto_one),
                    ],
                ),
                dbc.Card(
                    [
                        dbc.CardHeader(graph_two_name),
                        dbc.CardBody(self.cyto_two)
                    ],
                )
            ],
            style=padding
        )
        summary = dbc.Card(
            [
                html.H5("Summary", className="card-title"),
                html.P("{} nodes in each graph and {} different edge(s) per graph.".format(
                    graph_one.get_graph().number_of_nodes(), int(len(difference)/2)), className="card-text")
            ],
            className="w-50",
            style={'margin': '0 auto', 'padding': '10px 10px 10px 10px'}
        )
        layout = html.Div(
            [
                dbc.Row(dbc.Col(cards, width=12, align='center')),
                summary,
            ],
            style={'padding-bottom': '10px'}
        )

        return layout

    @staticmethod
    def _get_default_stylesheet(weights):
        """
        Returns the default stylesheet for initialisation.

        :param weights: (List) A list of weights of the edges.
        :return: (List) A List of definitions used for Dash styling.
        """
        stylesheet = \
            [
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'background-color': '#4cc9f0',
                        'font-family': 'sans-serif',
                        'font-size': '12',
                        'font-weight': 'bold',
                        'border-width': 1.5,
                        'border-color': '#161615',
                    }
                },
                {
                    "selector": 'edge',
                    "style": {
                        'label': 'data(weight)',
                        "line-color": "#4cc9f0",
                        'font-size': '8',
                    }
                },
                {
                    "selector": '[weight => 0]',
                    "style": {
                        "width": "mapData(weight, 0, {}, 1, 8)".format(max(weights)),
                    }
                },
                {
                    "selector": '[weight < 0]',
                    "style": {
                        "width": "mapData(weight, 0, {}, 1, 8)".format(min(weights)),
                    }
                },
                {
                    "selector": '.central',
                    "style": {
                        "background-color": "#80b918"
                    }
                },
                {
                    'selector': ':selected',
                    "style": {
                        "border-width": 2,
                        'background-color': '#f72585',
                        "border-color": "black",
                        "border-opacity": 1,
                        "opacity": 1,
                        "label": "data(label)",
                        "color": "black",
                        "font-size": 12,
                        'z-index': 9999
                    }
                },
                {
                    "selector": '.different',
                    "style": {
                        "line-color": "#f72585",
                        }
                }
            ]
        return stylesheet

    def _set_cyto_graph(self):
        """
        Updates and sets the two cytoscape graphs using the corresponding components.
        """
        layout = {'name': 'cose-bilkent'}
        style = {'width': '100%', 'height': '600px', 'padding': '5px 3px 5px 3px'}
        self.cyto_one = cyto.Cytoscape(
            id="cytoscape",
            layout=layout,
            style=style,
            elements=self.one_components[1],
            stylesheet=DualDashGraph._get_default_stylesheet(self.one_components[0])
        )
        self.cyto_two = cyto.Cytoscape(
            id="cytoscape_two",
            layout=layout,
            style=style,
            elements=self.two_components[1],
            stylesheet=DualDashGraph._get_default_stylesheet(self.two_components[0])
        )

    def _update_elements_dual(self, graph, difference, graph_number):
        """
        Updates the elements needed for the Dash Cytoscape Graph object.

        :param graph: (Graph) Graph object such as MST or ALMST.
        :param difference: (List) List of edges where the two graphs differ.
        :param graph_number: (Int) Graph number to update the correct graph.
        """
        weights = []
        elements = []

        for node in graph.get_pos():
            # If a node is "central", add the central label as a class
            if graph.get_graph().degree(node) >= 5:
                elements.append({
                    'data': {'id': node, 'label': node},
                    'selectable': 'true',
                    'classes': 'central'
                })
            else:
                elements.append({
                    'data': {'id': node, 'label': node},
                    'selectable': 'true',
                })

        for node1, node2, weight in graph.get_graph().edges(data=True):
            element = {'data': {'source': node1, 'target': node2, 'weight': round(weight['weight'], 4)}}

            # If the edge is a "different" edge, label with class "different" to highlight this edge
            if (node1, node2) in difference:
                element = {'data': {'source': node1, 'target': node2, 'weight': round(weight['weight'], 4)},
                           'classes': 'different'}

            weights.append(round(weight['weight'], 4))
            elements.append(element)

        # Update correct graph components
        if graph_number == 1:
            self.one_components = (weights, elements)
        if graph_number == 2:
            self.two_components = (weights, elements)

    def get_server(self):
        """
        Returns the comparison interface server

        :return: (Dash) Returns the Dash app object, which can be run using run_server.
            Returns a Jupyter Dash object if DashGraph has been initialised for Jupyter Notebook.
        """
        # Create an app from a comparison layout
        self.app.layout = self._generate_comparison_layout(self.graph_one, self.graph_two)
        # Return the app
        return self.app
