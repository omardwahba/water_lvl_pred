"""Collection of simple models for time-series forecasting.

Each module exposes a `build(input_size, output_size, optimizer='ADAM', criterion='MSE', lr=0.001)`
function that returns `(model, optimizer_instance, criterion_instance)`.
"""

from .mlp import build as build_mlp
from .lstm import build as build_lstm
from .cnn import build as build_cnn
from .gru import build as build_gru
from .logistic import build as build_logistic

def get_model_describtion(model):
	"""Return a human-readable description of a PyTorch model's trainable parameters.

	The returned string contains per-parameter name, shape and count, total trainable
	parameters and an estimate of required memory (assuming float32 = 4 bytes).
	"""
	lines = []
	total_params = 0
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		count = param.numel()
		total_params += count
		shape_str = str(tuple(param.shape))
		lines.append(f"{name:40s} shape={shape_str:20s} params={count}")

	bytes_per_param = 4  # float32
	total_bytes = total_params * bytes_per_param

	def _hr(n):
		for unit in ['B', 'KB', 'MB', 'GB']:
			if n < 1024.0:
				return f"{n:.2f}{unit}"
			n /= 1024.0
		return f"{n:.2f}TB"

	header = [
		f"Total trainable parameters: {total_params}",
		f"Estimated memory (float32): {total_bytes} bytes ({_hr(total_bytes)})",
		"Parameter breakdown:"
	]

	return "\n".join(header + lines)

__all__ = [
	"build_mlp",
	"build_lstm",
	"build_cnn",
	"build_gru",
	"build_logistic",
    "get_model_describtion",
]

