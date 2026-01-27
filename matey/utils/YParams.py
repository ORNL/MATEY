# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

from ruamel.yaml import YAML
import logging
import ast
from collections import OrderedDict

class YParams():
  """ Yaml file parser """
  def __init__(self, yaml_filename, config_name=None, print_params=False):
    self._yaml_filename = yaml_filename
    self._config_name = config_name
    self.params = {}

    if print_params:
      print("------------------ Configuration ------------------")

    with open(yaml_filename) as _file:
      if config_name is None:
        for key, val in YAML().load(_file).items():
          try:
            val=ast.literal_eval(val)
          except:
            pass
          if True: print(key, val)
          if val =='None': val = None
          if key=="tokenizer_heads" and isinstance(val, str):
            #e.g.,  "[ordereddict([('head_name', 'tk-3D'), ('patch_size', [[8, 8, 8]])])]"
            val = val.replace("ordereddict", "OrderedDict")
            python_list = eval(val, {"OrderedDict": OrderedDict}, {})
            val = [dict(item) for item in python_list]
          if key=="hierarchical" and isinstance(val, str):
            #e.g.,  "ordereddict([('filtersize', [1, 4, 8]), ('cubsize', [128, 512, 1024])])"
            val = val.replace("ordereddict", "OrderedDict")
            val = dict(eval(val, {"OrderedDict": OrderedDict}, {})) 

          self.params[key] = val
          self.__setattr__(key, val)
      else:
        for key, val in YAML().load(_file)[config_name].items():
          if print_params: print(key, val)
          if val =='None': val = None
          if key=="tokenizer_heads" and isinstance(val, str):
            #e.g.,  "[ordereddict([('head_name', 'tk-3D'), ('patch_size', [[8, 8, 8]])])]"
            val = val.replace("ordereddict", "OrderedDict")
            python_list = eval(val, {"OrderedDict": OrderedDict}, {})
            val = [dict(item) for item in python_list]

          self.params[key] = val
          self.__setattr__(key, val)

    if print_params:
      print("---------------------------------------------------")

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val
    self.__setattr__(key, val)

  def __contains__(self, key):
    return (key in self.params)

  def update_params(self, config):
    for key, val in config.items():
      self.params[key] = val
      self.__setattr__(key, val)

  def log(self):
    logging.info("------------------ Configuration ------------------")
    logging.info("Configuration file: "+str(self._yaml_filename))
    logging.info("Configuration name: "+str(self._config_name))
    for key, val in self.params.items():
        logging.info(str(key) + ' ' + str(val))
    logging.info("---------------------------------------------------")
