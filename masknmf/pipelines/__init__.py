import lazy_loader

# NOTE: names load on first use, so reading one config dataclass does not import every pipeline
__getattr__, __dir__, _ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "TwoPhotonCalciumPipeline",
    "WidefieldSinglechannelPipeline",
    "OnePhotonCulturePipeline",
    "GlutamateCalciumSpinePipeline",
    "OnePhotonCalciumPipeline"
]
