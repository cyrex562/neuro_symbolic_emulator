"""Neural network architectures for ALU functional units."""

from .monolithic import MonolithicMLP
from .bitslice import BitSliceNetwork
from .hierarchical import HierarchicalNetwork

ARCHITECTURES = {
    "monolithic": MonolithicMLP,
    "bitslice": BitSliceNetwork,
    "hierarchical": HierarchicalNetwork,
}
