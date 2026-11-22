import lazy_loader

# NOTE: names load on first use, so the viewer's CompressionArray does not import the denoiser's pytorch_lightning
__getattr__, __dir__, _ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "TrendArray",
    "CompressionTemporalDenoiser",
    "train_total_variance_denoiser",
    "save_denoiser",
    "load_denoiser",
    "pretrain_denoiser",
    "compression_routine",
    "CompressionArray",
    "CompressionResidualArray",
    "CompressStrategy",
    "CompressDenoiseStrategy",
    "SplineDetrend"
]
