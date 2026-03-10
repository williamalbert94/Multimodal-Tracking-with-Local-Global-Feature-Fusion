import yaml
import os
import logging
import sys

def create_log_file(exp_name,mode):
    log_dir = os.path.join('./logs/'+mode+'/', exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'log.txt')
    return log_path

def setup_logger(log_path, name='train'):
    """
    Configura un logger eficiente y compatible con Docker.

    Args:
        log_path: Ruta al archivo .txt donde se guardarán los logs
        name: Nombre del logger

    Returns:
        logger configurado
    """
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remover handlers existentes para evitar duplicados
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formato detallado con timestamp
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para archivo .txt (con buffer moderado para eficiencia)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para consola (stdout, compatible con Docker)
    # Sin buffering para captura inmediata en Docker logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.flush = lambda: sys.stdout.flush()  # Flush explícito
    logger.addHandler(console_handler)

    return logger

def parse_args_from_yaml(yaml_path):
    
    with open(yaml_path, 'r') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
        
    return args
   
class EasyDict(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__
    
    


