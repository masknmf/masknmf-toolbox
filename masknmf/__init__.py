import logging
import warnings

import lazy_loader

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("cmap").setLevel(logging.ERROR)
# torch's notices about its own sparse support, nothing the user can act on
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

# NOTE: every name in __init__.pyi is imported on first access (scientific-python SPEC 1), so `masknmf --help` and
# the viewer load only the modules they use. A name exported here goes in __init__.pyi.
__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
