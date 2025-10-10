from typing import NamedTuple, Type, Dict, Sequence, List, Callable, Any
import ipywidgets as widgets
from IPython.display import  display, HTML

def cprint(s, color = 'black', background = 'white'):
    display(HTML(f'<span style="color:{color};background:{background};">{s}</span>'))

class AttributeForm():
    
    class InputGroupWidgets(NamedTuple):
        name: widgets.Label
        selector: widgets.Dropdown
        value: widgets.FloatText
        units: widgets.Label
        desc: widgets.Label

        def on_dropdown(self, change):
            attr = self.selector.value
            if attr is not None:
                self.value.disabled = False
                self.units.value = attr.units or ''
                self.desc.value = attr.desc or ''
            else:
                self.value.disabled = True
                self.units.value = ''
                self.desc.value = ''

        def get_value(self) -> Dict[str, float]:
            if self.selector.value is not None:
                return { self.selector.label: self.value.value }
            else:
                return {}
    
    def __init__(self, cls: type, input_groups: Sequence[str], default_model: Any = None):
        ''' Display form for specifing DeductorBase objects paramters.

        Result model will be available via `model` property.

        Paramters
        ---------
        cls : type
            Subclass of DeductorBase.
        input_groups : List[str]
            List of DeductorBase attributes groups. For each group input fields will be created in result form.
        default_model: DeductorBase
            Model to fill input fields. 
        '''
        self._groups = []
        self._cls = cls
        self._model = None
        if not isinstance(default_model, cls):
            raise TypeError(f'default_model must be instance of cls class ({cls})')
        # create layout
        #grid = widgets.GridspecLayout(len(input_groups), 5, width_ratios=[5,5,5,2,15])
        grid_widget_layout = widgets.Layout(width='auto')
        grid_widgets = [ widgets.Label(value=text, style={'font_weight': 'bold'}) for text in ('Group', 'Parameter', 'Value', 'Units', 'Description') ] 
        for k, group in enumerate(input_groups):
            row = AttributeForm.InputGroupWidgets(
                    widgets.Label(value=group, layout=grid_widget_layout),
                    widgets.Dropdown(options=[(attr.name, attr) for attr in cls.get_attributes_by_group(group)] + [('None', None)], layout=grid_widget_layout),
                    widgets.FloatText(value=0.0, layout=grid_widget_layout),
                    widgets.Label(value='', layout=grid_widget_layout),
                    widgets.Label(value='', layout=grid_widget_layout),
                )
            # set values
            attr = row.selector.value
            if attr is not None:
                row.units.value = attr.units or ''
                row.desc.value = attr.desc or ''
            else:
                row.value.disabled = True
            if default_model is not None:
                row.value.value = getattr(default_model, attr.name)
            # configure events
            row.selector.observe(row.on_dropdown, names='value')
            # add to grid
            #for l, w in enumerate(row):
            #    grid[k,l] = w
            grid_widgets.extend(row)
            # save to group list
            self._groups.append(row)
        # grid
        grid = widgets.GridBox(children=grid_widgets, layout=widgets.Layout(grid_template_columns='15% 20% 10% 5% auto'))
        # overall display
        separator = widgets.HTML(value="<hr>")
        title = widgets.Label(value=f'{cls.__name__}', style={'font_weight': 'bold', 'font_size': '16'})
        name = default_model.name if default_model is not None else ''
        self._name = widgets.Text(value=name)
        name =  widgets.HBox([widgets.Label(value='Model name'), self._name])
        button = widgets.Button(description='Calculate')
        button.on_click(self.on_calculate)
        self._output = widgets.Output()
        self._vbox = widgets.VBox([separator, title, separator, name, separator, grid, separator, button, self._output])
        # display
        display(self._vbox)

    def get_values(self) -> Dict[str, float]:
        values = {}
        for group in self._groups:
            values.update(group.get_value())
        return values

    def on_calculate(self, b) -> None:
        self._output.clear_output()
        with self._output:
            # init model
            values = self.get_values()
            try:
                self._model = self._cls(self._name.value, **values)
            except (ValueError, TypeError) as e:
                cprint('MODEL DEDUCTION FAILED', color='red')
                print(e)
                return
            # print 
            if self._model.is_fully_defined():
                cprint('MODEL IS FULLY DEFINED.\n', color='green')
            else:
                cprint('MODEL IS NOT FULLY DEFINED.\n', color='red')
            display(self._model.to_string(tablefmt='html'))

    @property
    def model(self): 
        return self._model
    
class ParameterWidgets:
    def __init__(self, name: str, value: float = 0.0, max_value: float = 1.0, units: str = '', desc: str = ''):
        grid_widget_layout = widgets.Layout(width='auto')
        self._value_widget =  widgets.FloatText(value=value, layout=grid_widget_layout)
        self._max_value = max_value
        self._max_value_widget = widgets.Label(value=str(max_value), layout=grid_widget_layout)
        self._widgets = [ 
            widgets.Label(value=name, layout=grid_widget_layout),
            self._value_widget,
            widgets.Label(value=units, layout=grid_widget_layout),
            self._max_value_widget,
            widgets.Label(value=desc, layout=grid_widget_layout)
        ]   
        self._value_widget.observe(lambda change: self._set_color(), names='value')

    @property
    def widgets(self) -> List[widgets.Widget]:
        return self._widgets

    @property 
    def value(self) -> float:
        return self._value_widget.value

    @value.setter
    def value(self, value: float) -> None:
        self._value_widget.value = value
        self._set_color()

    @property 
    def max_value(self) -> float:
        return self._max_value

    @value.setter
    def max_value(self, value: float) -> None:
        self._max_value = value
        self._max_value_widget.value = str(value)
        self._set_color()

    def _set_color(self) -> None:
        if self._value_widget.value > self._max_value:
            self._max_value_widget.style.text_color = 'red'
        else:
            self._max_value_widget.style.text_color = 'black'

    def register_value_callback(self, cb: Callable[[Any], None]) -> None:
        self._value_widget.observe(cb, names='value')

class ParameterForm():                    
    def __init__(self, title: str, param_widgets: Sequence[ParameterWidgets], custom_widget = None):
        # create layout
        grid_widget_layout = widgets.Layout(width='auto')
        # header
        grid_widgets = [ widgets.Label(value=text, style={'font_weight': 'bold'}) for text in ('Parameter', 'Value', 'Units', 'Absolute Maximum', 'Description') ] 
        # add to parent gid widget
        for param_widget in param_widgets:
            grid_widgets.extend(param_widget.widgets)
            param_widget.register_value_callback(self.on_update)
        grid = widgets.GridBox(children=grid_widgets, layout=widgets.Layout(grid_template_columns='15% 20% 10% 15% auto'))
        # overall display
        separator = widgets.HTML(value="<hr>")
        title = widgets.Label(value=title, style={'font_weight': 'bold', 'font_size': '20'})
        button = widgets.Button(description='Calculate')
        button.on_click(self.on_calculate)
        self._output = widgets.Output()
        if custom_widget is None:
            self._vbox = widgets.VBox([title, separator, grid, separator, button, self._output])
        else:
            self._vbox = widgets.VBox([title, separator, grid, separator, custom_widget, separator, button, self._output])
        # update widgets
        self.update()
        # display
        display(self._vbox)

    def on_calculate(self, state):
        self._output.clear_output()
        with self._output:
            self.calculate()

    def on_update(self, change):
        self.update()

    def update(self):
        pass

