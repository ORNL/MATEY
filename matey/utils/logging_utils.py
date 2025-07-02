import os
import logging
import torch
import time
from functools import wraps
import torch.profiler
from contextlib import contextmanager

_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def config_logger(log_level=logging.INFO):
  logging.basicConfig(format=_format, level=log_level)

def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='tensorflow.log'):

  if not os.path.exists(os.path.dirname(log_filename)):
    os.makedirs(os.path.dirname(log_filename))

  if logger_name is not None:
    log = logging.getLogger(logger_name)
  else:
    log = logging.getLogger()

  fh = logging.FileHandler(log_filename)
  fh.setLevel(log_level)
  fh.setFormatter(logging.Formatter(_format))
  log.addHandler(fh)

def log_versions():
  import torch
  import subprocess

  logging.info('--------------- Versions ---------------')
  logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
  logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
  logging.info('Torch: ' + str(torch.__version__))
  logging.info('----------------------------------------')



@contextmanager
def profile_function(enabled=True, logdir="log_profiler"):
    if enabled:
         with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
            record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            yield prof
    else:
        # Dummy profiler with a no-op `.step()` method
        class DummyProfiler:
            def step(self): pass
        yield DummyProfiler()

@contextmanager
def record_function_opt(name, enabled=True):
    if enabled:
        with torch.profiler.record_function(name):
            yield
    else:
        yield  # No-op context

class Timer:
    def __init__(self, enable_sync=False):
        self.enable_sync = enable_sync
        self.start_time = None
        self.end_time = None
    def get_time(self):
        if self.enable_sync:
            torch.cuda.synchronize()
        return time.time()

    def __enter__(self):
        if self.enable_sync:
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable_sync:
            torch.cuda.synchronize()
        self.end_time = time.time()

    @property
    def elapsed_time(self):
        if self.start_time is not None and self.end_time is not None:
            return (self.end_time - self.start_time) * 1000  # ms
        else:
            return None