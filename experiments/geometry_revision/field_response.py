"""Field response metrics from the manuscript implementation."""
import numpy as np

def response_metrics(ev, predictions, offsets):
    residual = np.asarray(predictions)-ev.raw_y
    return {"displacement_RMSE_mm": float(np.sqrt(np.mean(residual**2))),
        "weighted_SSE": float(np.sum((residual/ev.sigma)**2)),
        "channels": {c: {"offset_mm": float(offsets[c]),
            "RMSE_mm": float(np.sqrt(np.mean(residual[s]**2))),
            "residual_mm": residual[s].tolist(),
            "prediction_mm": np.asarray(predictions)[s].tolist()}
            for c, s in ev.channel_slices.items()}}
