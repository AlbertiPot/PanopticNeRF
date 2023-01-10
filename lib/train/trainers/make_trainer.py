from .trainer import Trainer
import imp

def _wrapper_factory(cfg, network):
    module = cfg.loss_module
    path = cfg.loss_path
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network) # Networkwrapper负责计算loss
    return network_wrapper

def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return Trainer(network) # train val iter    # wrapper之后再送给trainer初始化