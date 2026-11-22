import lazy_loader

# NOTE: names load on first use, so the demixing viewer does not import roicat, matplotlib and plotly
__getattr__, __dir__, _ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "CompressionVis",
    "visualize_superpixels_peaks",
    "plot_ith_roi",
    "construct_index",
    "MotionCorrectionVis",
    "SingleSessionDemixingVis",
    "MultiSessionDemixingVis",
    "ClassificationVis"
]
