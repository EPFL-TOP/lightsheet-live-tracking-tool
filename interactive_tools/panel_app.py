from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import panel as pn
from bokeh_visualisation import make_layout as bokeh_visualisation
from bokeh_selection import make_layout as bokeh_selection

pn.extension()

# Wrap the Bokeh Application in Panel
visu_panel = pn.panel(bokeh_visualisation, sizing_mode="stretch_both")
sel_panel = pn.panel(bokeh_selection, sizing_mode="stretch_both")

# Example layout with tabs
tabs = pn.Tabs(
    ("Selection", sel_panel),
    ("Visualisation", visu_panel),
)

tabs.servable()
