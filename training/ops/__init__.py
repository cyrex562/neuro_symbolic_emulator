"""ALU operation definitions and data generators for neural functional units."""

from .base import AluOp, DataGenerator
from .arithmetic import AddOp, SubOp
from .bitwise import AndOp, OrOp, XorOp
from .comparison import SltOp, SltuOp
from .shift import SllOp, SrlOp, SraOp

ALL_OPS = {
    "add": AddOp,
    "sub": SubOp,
    "and": AndOp,
    "or": OrOp,
    "xor": XorOp,
    "slt": SltOp,
    "sltu": SltuOp,
    "sll": SllOp,
    "srl": SrlOp,
    "sra": SraOp,
}
