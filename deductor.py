from typing import Tuple, Callable, Dict, Union, Optional, Sequence, Set, List
from inspect import signature
from copy import copy, deepcopy
import numpy as np
import regex as re
from tabulate import tabulate

class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)
    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)
    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))

class DerivateRule:
    def __init__(self, output_names: Union[Tuple[str,...], str], func: Callable, input_names: Optional[Tuple[str,...]] = None):
        self._out_vars_names = output_names if isinstance(output_names, tuple) else (output_names,)
        func_arg_names = signature(func).parameters.keys()
        self._func = func
        if input_names is not None:
            if len(input_names) != len(func_arg_names):
                raise ValueError('Derivate rule {input_names} -> {output_names}: incompatible fuction signature with args {func_arg_names}')
            self._in_vars_names = input_names
        else:
            self._in_vars_names = tuple(func_arg_names)

    @property
    def input_variables_names(self) -> Tuple[str]:
        return self._in_vars_names

    @property
    def output_variables_names(self) -> Tuple[str]:
        return self._out_vars_names

    def __call__(self, variables: Dict[str, float], variables_name_set: Optional[Set[str]] = None) -> Dict[str, float]:
        # get known variables names
        if variables_name_set is None:
            variables_name_set = set(variables.keys())
        # check if input variables are known
        if variables_name_set.issuperset(self._in_vars_names):
            # apply rule
            out_vars_values = self._func(*[ variables[name] for name in self._in_vars_names ])
            # parse result depening on number of ouput variables
            if len(self._out_vars_names) == 1:
                return { self._out_vars_names[0]: out_vars_values }
            elif len(self._out_vars_names) == len(out_vars_values):
                return dict(zip(self._out_vars_names, out_vars_values))
            else:
                raise TypeError('Derivative rule {self._in_vars_names} -> {self._out_vars_names}: incorrect function signature')
        else:
            # nothing is deduced
            return {}

    def __repr__(self) -> str:
        return f'DerivateRule({self._in_vars_names} -> {self._out_vars_names})'

class AnnotatedFloat(float):
    def __new__(cls, value: float, units: Optional[str], desc: Optional[str]):
        instance = super(AnnotatedFloat, cls).__new__(cls, value)
        instance.units = units
        instance.desc = desc
        return instance

    def __repr__(self):
        return f'{float(self)} {self.units} ({self.desc})'

class GroupItem:
    def __init__(self, groups: Sequence[str] = []):
        self.groups = copy(groups)

class Attribute(GroupItem):
    set_value = None
    get_value = None
    
    def __init__(self, name: str, units: Optional[str], desc: Optional[str], **kwargs):
        super(Attribute, self).__init__(**kwargs)
        if not self.name_is_valid(name):
            raise ValueError(f'Variable name {name} must contain only alpha-numeric characters and underscores.') 
        self._name = name
        self._units = units
        self._desc = desc

    @staticmethod
    def name_is_valid(name: str) -> bool:
        return re.match(r'^[A-Za-z][A-Za-z0-9_]*$', name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def units(self) -> str:
        return self._units
    
    @property
    def desc(self) -> str:
        return self._desc

    def add_to_class(self, cls: type) -> None:
        if hasattr(cls, self._name):
            raise TypeError(f'Dublicate attribute {self._name}')
        setattr(cls, self._name, property(self.get_value, self.set_value))

class BaseAttribute(Attribute):
    def __init__(self, *args, **kwargs):
        super(BaseAttribute, self).__init__(*args, **kwargs)
        if 'base' not in self.groups:
            self.groups.append('base')
            
    @property
    def derivate_rules(self) -> List[DerivateRule]:
        return []
        
    def get_value(self, obj) -> AnnotatedFloat:
        return getattr(obj, '_' + self._name)

    def set_value(self, obj, value: float) -> None:
        setattr(obj, '_' + self._name, AnnotatedFloat(value, self._units, self._desc))

    def __repr__(self) -> str:
        return f'BaseAttribute "{self._name}" ({self._units}) --- "{self._desc}"'

class DerivedAttribute(Attribute):
    def __init__(self, name: str, units: Optional[str], desc: Optional[str], 
                 func: Callable[[float,...], float], **kwargs):
        super(DerivedAttribute, self).__init__(name, units, desc, **kwargs)
        self._func = func

    @property
    def derivate_rules(self) -> List[DerivateRule]:
        return [ DerivateRule((self._name,), self._func) ]

    @property
    def input_attrs_names(self) -> Sequence[str]:
        return signature(self._func).parameters.keys()
    
    def get_value(self, obj) -> AnnotatedFloat:
        input_vars_names = self.input_attrs_names
        input_vars = [ getattr(obj, name) for name in input_vars_names ]
        if any(np.isnan(input_vars)):
            return AnnotatedFloat(np.nan, self._units, self._desc)
        else:
            value = self._func(*input_vars)
            return AnnotatedFloat(value, self._units, self._desc)

    def add_to_class(self, cls: type) -> None: 
        if not all( hasattr(cls, name) for name in self.input_attrs_names ):
            raise TypeError(f'Derived atribute {self._name} requires attributes {self.input_attrs_names} to exists.')
        super(DerivedAttribute, self).add_to_class(cls)

    def __repr__(self) -> str:
        return f'DerivedAttribute "{self._name}" ({self._units}) from {list(self.input_attrs_names)} --- "{self._desc}"'

class AliasAttribute(Attribute):
    def __init__(self, name: str, original_name: str, **kwargs):
        super(AliasAttribute, self).__init__(name, None, None, **kwargs)
        self._original_name = original_name

    def get_value(self, obj) -> AnnotatedFloat:
        return getattr(obj, self._original_name)

    def set_value(self, obj, value):
        setattr(obj, self._original_name, value)

    @property
    def derivate_rules(self) -> List[DerivateRule]:
        return [ DerivateRule( (self._name,), lambda x: x, (self._original_name,) ),
                 DerivateRule( (self._original_name,), lambda x: x, (self._name,) ) ]

    def add_to_class(self, cls: type) -> None:
        if not hasattr(cls, self._original_name):
            raise TypeError(f'Alias atribute {self._name} requires attributes {self._original_name} to exist.')
        # disable setter if origina attr is not assignable
        if getattr(cls, self._original_name).setter is None:
            self.set_value = None
        # extract original attribute metainformation
        if hasattr(cls, '_ATTRIBUTES'):
            for attr in cls._ATTRIBUTES:
                if attr.name == self._original_name:
                    if self._units is None:
                        self._units = attr._units
                    if self._desc is None:
                        self._desc = attr._desc
        super(AliasAttribute, self).add_to_class(cls)

    def __repr__(self) -> str:
        return f'AliasAttribute "{self._name}" for "{self._original_name}"'

class ScaledAliasAttribute(Attribute):
    def __init__(self, name: str, scale: float, original_name: str, *args, **kwargs):
        super(ScaledAliasAttribute, self).__init__(name, *args, **kwargs)
        self._scale = scale
        self._original_name = original_name
  
    @property
    def derivate_rules(self) -> List[DerivateRule]:
        return [ DerivateRule( (self._name,), lambda x: x * self._scale, (self._original_name,) ),
                 DerivateRule( (self._original_name,), lambda x: x / self._scale, (self._name,) ) ]
    
    def get_value(self, obj) -> AnnotatedFloat:
        original_value = getattr(obj, self._original_name)
        return AnnotatedFloat(self._scale * original_value, self._units, self._desc)

    def set_value(self, obj, value) -> None:
        original_value = value / self._scale
        setattr(obj, self._original_name, original_value)

    def add_to_class(self, cls: type) -> None:
        if not hasattr(cls, self._original_name):
            raise TypeError(f'Alias atribute {self._name} requires attributes {self._original_name} to exist.')
        # disable setter if origina attr is not assignable
        if getattr(cls, self._original_name).setter is None:
            self.set_value = None
        super(ScaledAliasAttribute, self).add_to_class(cls)

    def __repr__(self) -> str:
        return f'ScaledAliasAttribute "{self._name}" ({self._units}) equals to  "{self._scale}*{self._original_name}" --- "{self._desc}"'

class Validator:
    def __init__(self, func: Callable[[float,...], bool], desc: str, strict: bool = False):
        self._func = func
        self._desc = desc
        self._strict = strict

    def __call__(self, m) -> bool:
        # collect arguments 
        args_names = signature(self._func).parameters.keys()
        args = { name : getattr(m, name) for name in args_names }
        # check that all arguments are defined
        if np.any(np.isnan(list(args.values()))):
            # model is not fully defined, but this is not error
            if self._strict:
                raise ValueError(f'{type(m).__name__}: {self._desc}: attributes {str.join(", ", args_names)} must be defined.')
        else:
            # validate
            if not self._func(**args):
                raise ValueError(f'{type(m).__name__}: {self._desc}')

    @property
    def desc(self) -> str:
        return self._desc

    def __repr__(self) -> str:
        return f'ModelValidator({self._desc})'

class DeductorBase:
    _ATTRIBUTES: List[Attribute] = []
    _DERIVATE_RULES: List[DerivateRule] = []
    _VALIDATORS: List[Validator] = []
    
    def __init_subclass__(cls, **kwargs):
        super(DeductorBase, cls).__init_subclass__(**kwargs)
        # create class atrributes
        cls._ALL_ATTRIBUTES = []
        cls._ALL_DERIVATE_RULES = []
        cls._ALL_VALIDATORS = []
        cls._BASE_ATTRIBUTES_NAMES = []
        # populate them with superclass attributes and rules
        for c in reversed(cls.__mro__):
            if issubclass(c, DeductorBase):
                cls._ALL_DERIVATE_RULES.extend(c._DERIVATE_RULES)
                cls._ALL_VALIDATORS.extend(c._VALIDATORS)
                for attr in c._ATTRIBUTES:
                    # form list of base attributes
                    if isinstance(attr, BaseAttribute):
                        cls._BASE_ATTRIBUTES_NAMES.append(attr.name)
                    # get derivative rules
                    cls._ALL_DERIVATE_RULES.extend(attr.derivate_rules)
                    cls._ALL_ATTRIBUTES.append(attr)
        # add properties (only for current class because superclass properties are inherited)
        for attr in cls._ATTRIBUTES:
            attr.add_to_class(cls)

    def __init__(self, rel_tolerance: float = 0.05, validate = True, **kwargs):
        # get tolerance
        self._rel_tolerance = rel_tolerance
        # vaiables
        variables = kwargs
        # check input variables names:
        valid_input_names = set(self._BASE_ATTRIBUTES_NAMES)
        for rule in self._ALL_DERIVATE_RULES:
            valid_input_names.update(rule.input_variables_names) 
        for name in variables.keys():
            if name not in valid_input_names:
                raise ValueError(f'DeductorBase: unknown input variable {name}, valid are {valid_input_names}')
        # deduce variables
        n_variables_deduced = len(variables)
        variables_names = set(variables.keys())
        while n_variables_deduced > 0:
            n_variables_deduced = 0
            for rule in self._ALL_DERIVATE_RULES:
                # derive variables
                new_variables = rule(variables, variables_names)
                # check them
                for name, value in new_variables.items():
                    # check value
                    if np.isnan(value):
                        raise ValueError(f'DeductorBase: variable {name} derived from {rule} is NAN.')
                    # check if variable exists
                    old_value = variables.get(name)
                    if old_value is None:
                        # new value is deduced
                        n_variables_deduced += 1
                        variables[name] = value
                        variables_names.add(name)
                    else:
                        # variable already exists
                        if abs(value - old_value) > self._rel_tolerance*max(abs(old_value), abs(value)):
                              raise ValueError(f'DeductorBase: value of {name} is {old_value}, but it contradicts to value {value} deudced from {rule}')
        # set base attributes
        for name in self._BASE_ATTRIBUTES_NAMES:
            value = variables.get(name)
            if value is None:
                value = np.nan
            setattr(self, name, value)
        # validate
        if validate:
            self.validate()

    @classproperty 
    def attributes(cls) -> List[Attribute]:
        return cls._ALL_ATTRIBUTES

    @classmethod
    def get_attributes_by_group(cls, group: str) -> List[Attribute]:
        return [ attr for attr in cls._ALL_ATTRIBUTES if group in attr.groups ]

    @classmethod
    def get_attribute_by_name(cls, name: str) -> Optional[Attribute]:
        for attr in cls._ALL_ATTRIBUTES:
            if name == attr.name:
                return attr
        return None

    @classproperty 
    def validators(cls) -> List[Validator]:
        return cls._ALL_VALIDATORS

    def is_fully_defined(self) -> bool:
        for name in self._BASE_ATTRIBUTES_NAMES:
            if np.isnan(getattr(self, name)):
                return False
        return True

    def validate(self):
        for validator in self._VALIDATORS:
            validator(self)
        
    def to_string(self, group: str = 'base', tablefmt="simple") -> str:
        display_all = group == 'all'
        # form table
        table = []
        for attr in self._ALL_ATTRIBUTES:
            if display_all or group in attr.groups:
                value = getattr(self, attr.name)
                table.append([attr.name, value if not np.isnan(value) else None, value.units, value.desc])
        return tabulate(table, headers=['Name', 'Value', 'Units', 'Description'], tablefmt = tablefmt)
            
    def __repr__(self) -> str:
        return self.to_string()

class DeductorBaseNamed(DeductorBase):
    def __init__(self, name: str, desc: str = '', **kwargs):
        super(DeductorBaseNamed, self).__init__(**kwargs)
        self._name = name
        self._desc = desc
        
    @property 
    def name(self) -> str:
        return self._name

    @property 
    def desc(self) -> str:
        return self._desc
        