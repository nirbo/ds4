#!/usr/bin/env python3
"""Dependency-free PolarQuant and QJL numerical authority for Ornith-35."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Sequence


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]
HEAD_DIM = 256


class TurboQuantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TurboQuantError(message)


# Generated from the exact spherical-coordinate Beta density in arXiv
# 2504.19874v1 with converged Lloyd-Max iteration. The d=128 values also match
# the independently pinned 0xSero reference files to displayed precision.
_CENTROIDS: dict[tuple[int, int], tuple[float, ...]] = {
    (128, 1): (-0.07066157273809426, 0.07066157273809426),
    (128, 2): (
        -0.13304019825336852,
        -0.03999094521535639,
        0.039990945215356344,
        0.13304019825336846,
    ),
    (128, 3): (
        -0.18839061380207797,
        -0.11813298369899358,
        -0.06658059531595682,
        -0.021602468667239208,
        0.021602468667239208,
        0.06658059531595682,
        0.11813298369899358,
        0.18839061380207797,
    ),
    (128, 4): (
        -0.2376271867309536,
        -0.18079372947217434,
        -0.14176165429673318,
        -0.11024706538276369,
        -0.08279256667309583,
        -0.057744535605257115,
        -0.03413402823112088,
        -0.011296498142743925,
        0.011296498142743847,
        0.03413402823112081,
        0.05774453560525704,
        0.08279256667309573,
        0.11024706538276359,
        0.14176165429673304,
        0.18079372947217423,
        0.23762718673095337,
    ),
    (256, 1): (-0.049916507721605746, 0.049916507721605746),
    (256, 2): (
        -0.09423680066889677,
        -0.02828811999467261,
        0.028288119994672584,
        0.09423680066889675,
    ),
    (256, 3): (
        -0.1338479440323176,
        -0.08375896775388561,
        -0.04716193453734492,
        -0.015295735736674062,
        0.015295735736674062,
        0.04716193453734492,
        0.08375896775388561,
        0.1338479440323176,
    ),
    (256, 4): (
        -0.1693834023763135,
        -0.1285573496586392,
        -0.10066636577797276,
        -0.07821939632405821,
        -0.05870616746679093,
        -0.04092915969184447,
        -0.024188228109530426,
        -0.008004050632916985,
        0.00800405063291698,
        0.024188228109530422,
        0.04092915969184447,
        0.05870616746679093,
        0.07821939632405821,
        0.10066636577797276,
        0.1285573496586392,
        0.1693834023763135,
    ),
    (256, 5): (
        -0.2014541141162268,
        -0.16671740361413484,
        -0.14381445117637195,
        -0.12601928606617685,
        -0.11110872818963682,
        -0.0980530816544512,
        -0.08628429931693198,
        -0.07545230211835824,
        -0.06532373979890999,
        -0.05573342809969586,
        -0.046558528501524185,
        -0.037703689602368466,
        -0.029091921820097136,
        -0.02065865937328187,
        -0.012347671043963252,
        -0.0041080655477713475,
        0.0041080655477713475,
        0.012347671043963252,
        0.02065865937328187,
        0.029091921820097136,
        0.037703689602368466,
        0.046558528501524185,
        0.05573342809969586,
        0.06532373979890999,
        0.07545230211835824,
        0.08628429931693198,
        0.0980530816544512,
        0.11110872818963682,
        0.12601928606617685,
        0.14381445117637195,
        0.16671740361413484,
        0.2014541141162268,
    ),
    (256, 6): (
        -0.23049485088523494,
        -0.20007108515782255,
        -0.18043804219616084,
        -0.1654781225849854,
        -0.1531757580384251,
        -0.1426012856257228,
        -0.133244820500129,
        -0.12479474681130201,
        -0.11704583855115059,
        -0.10985529011081516,
        -0.10311942899225857,
        -0.09676040685643382,
        -0.09071812727268802,
        -0.08494510963544205,
        -0.07940308980708534,
        -0.07406069444889905,
        -0.06889180426427105,
        -0.06387437349459753,
        -0.05898955994143774,
        -0.05422107140250057,
        -0.049554666086058826,
        -0.04497776458247914,
        -0.040479143945082706,
        -0.03604869303542452,
        -0.03167721410879684,
        -0.027356259624724584,
        -0.02307799607140486,
        -0.018835088580200648,
        -0.014620601528783068,
        -0.010427911356198334,
        -0.006250628551748713,
        -0.002082526307516963,
        0.002082526307516963,
        0.006250628551748713,
        0.010427911356198334,
        0.014620601528783068,
        0.018835088580200648,
        0.02307799607140486,
        0.027356259624724584,
        0.03167721410879684,
        0.03604869303542452,
        0.040479143945082706,
        0.04497776458247914,
        0.049554666086058826,
        0.05422107140250057,
        0.05898955994143774,
        0.06387437349459753,
        0.06889180426427105,
        0.07406069444889905,
        0.07940308980708534,
        0.08494510963544205,
        0.09071812727268802,
        0.09676040685643382,
        0.10311942899225857,
        0.10985529011081516,
        0.11704583855115059,
        0.12479474681130201,
        0.133244820500129,
        0.1426012856257228,
        0.1531757580384251,
        0.1654781225849854,
        0.18043804219616084,
        0.20007108515782255,
        0.23049485088523494,
    ),
    (256, 7): (
        -0.25699380022988546,
        -0.229799156481003,
        -0.21249641262234912,
        -0.19947146746546718,
        -0.18887877344909522,
        -0.17986883153156374,
        -0.17197666110585484,
        -0.16491867894361748,
        -0.15850853726635285,
        -0.15261688828534962,
        -0.14715009529898804,
        -0.14203807746474868,
        -0.13722694440087503,
        -0.13267431453993936,
        -0.12834621991347334,
        -0.12421499117788996,
        -0.12025777132554228,
        -0.11645544564937886,
        -0.11279185499866601,
        -0.10925320652452639,
        -0.10582762504754319,
        -0.10250480645577906,
        -0.09927574638782864,
        -0.09613252531300416,
        -0.09306813644141481,
        -0.0900763465674649,
        -0.08715158252606446,
        -0.08428883777639896,
        -0.0814835949548013,
        -0.07873176120972367,
        -0.07602961385172382,
        -0.07337375439084269,
        -0.07076106944217473,
        -0.06818869729262564,
        -0.0656539991626299,
        -0.06315453438385699,
        -0.06068803886071174,
        -0.058252406299329064,
        -0.05584567177990821,
        -0.05346599732197084,
        -0.05111165915149947,
        -0.04878103642699392,
        -0.04647260122063385,
        -0.044184909582777,
        -0.04191659354437475,
        -0.03966635393365707,
        -0.03743295390150242,
        -0.03521521306495492,
        -0.03301200219093228,
        -0.030822238352722886,
        -0.02864488050075304,
        -0.0264789253966012,
        -0.024323403865575908,
        -0.02217737732854678,
        -0.0200399345782767,
        -0.017910188769373774,
        -0.01578727459426726,
        -0.013670345620396943,
        -0.011558571766159987,
        -0.009451136895140752,
        -0.007347236509803729,
        -0.005246075527196465,
        -0.0031468661203185644,
        -0.0010488256096894468,
        0.0010488256096894468,
        0.0031468661203185644,
        0.005246075527196465,
        0.007347236509803729,
        0.009451136895140752,
        0.011558571766159987,
        0.013670345620396943,
        0.01578727459426726,
        0.017910188769373774,
        0.0200399345782767,
        0.02217737732854678,
        0.024323403865575908,
        0.0264789253966012,
        0.02864488050075304,
        0.030822238352722886,
        0.03301200219093228,
        0.03521521306495492,
        0.03743295390150242,
        0.03966635393365707,
        0.04191659354437475,
        0.044184909582777,
        0.04647260122063385,
        0.04878103642699392,
        0.05111165915149947,
        0.05346599732197084,
        0.05584567177990821,
        0.058252406299329064,
        0.06068803886071174,
        0.06315453438385699,
        0.0656539991626299,
        0.06818869729262564,
        0.07076106944217473,
        0.07337375439084269,
        0.07602961385172382,
        0.07873176120972367,
        0.0814835949548013,
        0.08428883777639896,
        0.08715158252606446,
        0.0900763465674649,
        0.09306813644141481,
        0.09613252531300416,
        0.09927574638782864,
        0.10250480645577906,
        0.10582762504754319,
        0.10925320652452639,
        0.11279185499866601,
        0.11645544564937886,
        0.12025777132554228,
        0.12421499117788996,
        0.12834621991347334,
        0.13267431453993936,
        0.13722694440087503,
        0.14203807746474868,
        0.14715009529898804,
        0.15261688828534962,
        0.15850853726635285,
        0.16491867894361748,
        0.17197666110585484,
        0.17986883153156374,
        0.18887877344909522,
        0.19947146746546718,
        0.21249641262234912,
        0.229799156481003,
        0.25699380022988546,
    ),
}

_K8_POSITIVE_CENTROIDS = (
    0.0005263794355647208,
    0.0015791875086739785,
    0.0026321432202492574,
    0.003685345104412376,
    0.004738891858197674,
    0.005792882407109309,
    0.006847415971072378,
    0.007902592130892592,
    0.008958510895340905,
    0.010015272768981557,
    0.01107297882086445,
    0.012131730754205655,
    0.013191630977183145,
    0.014252782674978615,
    0.015315289883200448,
    0.016379257562827615,
    0.01744479167681952,
    0.01851199926854257,
    0.01958098854217061,
    0.020651868945223354,
    0.021724751253414528,
    0.02279974765798982,
    0.023876971855743808,
    0.02495653914191487,
    0.026038566506167968,
    0.027123172731886736,
    0.02821047849900912,
    0.02930060649065459,
    0.03039368150380586,
    0.031489830564324385,
    0.03258918304659652,
    0.033691870798126314,
    0.03479802826941187,
    0.035907792649464525,
    0.037021304007355055,
    0.03813870544019726,
    0.03926014322800883,
    0.04038576699592058,
    0.04151572988423972,
    0.04265018872691025,
    0.04378930423895437,
    0.044933241213523364,
    0.04608216872923524,
    0.04723626036852937,
    0.048395694447826716,
    0.04956065426034796,
    0.050731328332511606,
    0.051907910694910785,
    0.05309060116895159,
    0.05427960567032838,
    0.05547513653061297,
    0.05667741283834725,
    0.057886660801151956,
    0.05910311413050132,
    0.06032701445096419,
    0.06155861173587945,
    0.06279816477161913,
    0.06404594165279814,
    0.0653022203110186,
    0.06656728907999121,
    0.06784144730016028,
    0.06912500596627569,
    0.07041828842170993,
    0.07172163110371521,
    0.07303538434426128,
    0.07435991323159499,
    0.07569559853822666,
    0.07704283772168366,
    0.07840204600509015,
    0.07977365754544517,
    0.08115812669839365,
    0.08255592938933382,
    0.08396756460189934,
    0.08539355599621755,
    0.08683445367090703,
    0.0882908360845669,
    0.08976331215456856,
    0.09125252355333345,
    0.09275914722502117,
    0.09428389814873084,
    0.09582753237801001,
    0.09739085039076972,
    0.0989747007887358,
    0.10057998439146776,
    0.10220765877692052,
    0.10385874332872445,
    0.10553432486007523,
    0.10723556389568458,
    0.10896370170704531,
    0.11072006821281656,
    0.11250609087606799,
    0.11432330475423699,
    0.11617336388696436,
    0.11805805424278122,
    0.119979308489592,
    0.12193922290819724,
    0.12394007683554269,
    0.1259843551086701,
    0.1280747740863783,
    0.13021431195991842,
    0.13240624423544708,
    0.1346541854914108,
    0.1369621387999495,
    0.13933455457556715,
    0.14177640110867978,
    0.14429324970159588,
    0.14689137821532172,
    0.14957789805261076,
    0.15236091128711297,
    0.1552497070130825,
    0.15825500936124223,
    0.1613892945147228,
    0.16466720128098067,
    0.1681060706684042,
    0.1717266667288157,
    0.17555415755556095,
    0.1796194787513182,
    0.1839612748874074,
    0.18862874270347715,
    0.1936859346063307,
    0.19921853416504426,
    0.20534504653876257,
    0.21223641377067867,
    0.22015311958287628,
    0.22952287498551183,
    0.2411282955076259,
    0.2566725422500891,
    0.28134691298346365,
)
_CENTROIDS[(256, 8)] = tuple(-value for value in reversed(_K8_POSITIVE_CENTROIDS)) + (
    _K8_POSITIVE_CENTROIDS
)

_MSE_TOTAL: dict[tuple[int, int], float] = {
    (128, 1): 0.36088859368690346,
    (128, 2): 0.11600007032012696,
    (128, 3): 0.0339659261608477,
    (128, 4): 0.009314791876879424,
    (256, 1): 0.362135617760949,
    (256, 2): 0.11674003883805695,
    (256, 3): 0.0342560042687041,
    (256, 4): 0.009407533529022872,
    (256, 5): 0.00247794223995182,
    (256, 6): 0.0006370650502317371,
    (256, 7): 0.00016161664229278422,
    (256, 8): 0.00004071067127539966,
}


@dataclass(frozen=True)
class Codebook:
    dimension: int
    bits: int
    centroids: tuple[float, ...]
    boundaries: tuple[float, ...]
    expected_total_mse: float


@dataclass(frozen=True)
class MSEEncoding:
    indices: tuple[int, ...]
    norm: float
    bits: int


@dataclass(frozen=True)
class ProductEncoding:
    mse: MSEEncoding
    residual_signs: tuple[int, ...]
    residual_norm: float
    total_bits: int


class SplitMixGaussian:
    """Version-stable SplitMix64 plus Box-Muller Gaussian stream."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        require(isinstance(seed, int), "Gaussian seed must be an integer")
        self._state = seed & self._MASK
        self._spare: float | None = None

    def _next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & self._MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return value ^ (value >> 31)

    def _uniform_open(self) -> float:
        return ((self._next_u64() >> 11) + 0.5) / float(1 << 53)

    def next(self) -> float:
        if self._spare is not None:
            value = self._spare
            self._spare = None
            return value
        radius = math.sqrt(-2.0 * math.log(self._uniform_open()))
        angle = 2.0 * math.pi * self._uniform_open()
        self._spare = radius * math.sin(angle)
        return radius * math.cos(angle)


def gaussian_matrix(rows: int, columns: int, seed: int) -> tuple[tuple[float, ...], ...]:
    require(rows > 0 and columns > 0, "Gaussian matrix dimensions must be positive")
    stream = SplitMixGaussian(seed)
    return tuple(tuple(stream.next() for _ in range(columns)) for _ in range(rows))


def codebook(dimension: int, bits: int) -> Codebook:
    centroids = _CENTROIDS.get((dimension, bits))
    require(centroids is not None, "unsupported TurboQuant codebook")
    boundaries = (-1.0,) + tuple(
        (left + right) * 0.5 for left, right in zip(centroids, centroids[1:])
    ) + (1.0,)
    return Codebook(
        dimension=dimension,
        bits=bits,
        centroids=centroids,
        boundaries=boundaries,
        expected_total_mse=_MSE_TOTAL[(dimension, bits)],
    )


def _validate_vector(vector: Vector, dimension: int, name: str) -> None:
    require(len(vector) == dimension, f"{name} dimension mismatch")
    require(all(math.isfinite(value) for value in vector), f"{name} is not finite")


def _validate_matrix(matrix: Matrix, dimension: int, name: str) -> None:
    require(len(matrix) == dimension, f"{name} row count mismatch")
    require(
        all(len(row) == dimension for row in matrix),
        f"{name} column count mismatch",
    )
    require(
        all(math.isfinite(value) for row in matrix for value in row),
        f"{name} is not finite",
    )


def matvec(matrix: Matrix, vector: Vector) -> tuple[float, ...]:
    dimension = len(vector)
    _validate_vector(vector, dimension, "matrix input")
    _validate_matrix(matrix, dimension, "matrix")
    return tuple(math.fsum(value * item for value, item in zip(row, vector)) for row in matrix)


def transpose_matvec(matrix: Matrix, vector: Vector) -> tuple[float, ...]:
    dimension = len(vector)
    _validate_vector(vector, dimension, "transpose input")
    _validate_matrix(matrix, dimension, "matrix")
    return tuple(
        math.fsum(matrix[row][column] * vector[row] for row in range(dimension))
        for column in range(dimension)
    )


def dot(left: Vector, right: Vector) -> float:
    require(len(left) == len(right), "dot-product dimension mismatch")
    return math.fsum(a * b for a, b in zip(left, right))


def l2_norm(vector: Vector) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def quantize_mse(vector: Vector, bits: int, rotation: Matrix) -> MSEEncoding:
    dimension = len(vector)
    selected = codebook(dimension, bits)
    _validate_vector(vector, dimension, "MSE input")
    _validate_matrix(rotation, dimension, "rotation")
    norm = l2_norm(vector)
    require(norm > 0.0, "TurboQuant cannot encode a zero vector")
    rotated = matvec(rotation, tuple(value / norm for value in vector))
    interior = selected.boundaries[1:-1]
    indices = tuple(bisect_right(interior, value) for value in rotated)
    return MSEEncoding(indices=indices, norm=norm, bits=bits)


def dequantize_mse(encoding: MSEEncoding, rotation: Matrix) -> tuple[float, ...]:
    selected = codebook(len(encoding.indices), encoding.bits)
    _validate_matrix(rotation, selected.dimension, "rotation")
    require(
        math.isfinite(encoding.norm) and encoding.norm >= 0.0,
        "MSE norm is invalid",
    )
    require(
        all(0 <= index < len(selected.centroids) for index in encoding.indices),
        "MSE codebook index is invalid",
    )
    rotated = tuple(selected.centroids[index] for index in encoding.indices)
    return tuple(value * encoding.norm for value in transpose_matvec(rotation, rotated))


def quantize_product(
    vector: Vector,
    total_bits: int,
    rotation: Matrix,
    projection: Matrix,
) -> ProductEncoding:
    require(2 <= total_bits <= 5, "product bit width must be in [2, 5]")
    mse = quantize_mse(vector, total_bits - 1, rotation)
    reconstructed = dequantize_mse(mse, rotation)
    residual = tuple(value - estimate for value, estimate in zip(vector, reconstructed))
    residual_norm = l2_norm(residual)
    _validate_matrix(projection, len(vector), "QJL projection")
    projected = matvec(projection, residual)
    signs = tuple(1 if value >= 0.0 else -1 for value in projected)
    return ProductEncoding(
        mse=mse,
        residual_signs=signs,
        residual_norm=residual_norm,
        total_bits=total_bits,
    )


def dequantize_product(
    encoding: ProductEncoding,
    rotation: Matrix,
    projection: Matrix,
) -> tuple[float, ...]:
    dimension = len(encoding.mse.indices)
    require(encoding.mse.bits + 1 == encoding.total_bits, "product bit width mismatch")
    require(
        len(encoding.residual_signs) == dimension
        and all(value in (-1, 1) for value in encoding.residual_signs),
        "QJL sign payload is invalid",
    )
    require(
        math.isfinite(encoding.residual_norm) and encoding.residual_norm >= 0.0,
        "QJL residual norm is invalid",
    )
    _validate_matrix(projection, dimension, "QJL projection")
    base = dequantize_mse(encoding.mse, rotation)
    scale = encoding.residual_norm * math.sqrt(math.pi / 2.0) / dimension
    correction = transpose_matvec(projection, encoding.residual_signs)
    return tuple(value + scale * residual for value, residual in zip(base, correction))


def product_inner_product(
    query: Vector,
    encoding: ProductEncoding,
    rotation: Matrix,
    projection: Matrix,
) -> float:
    dimension = len(query)
    require(len(encoding.mse.indices) == dimension, "product query dimension mismatch")
    base = dequantize_mse(encoding.mse, rotation)
    projected_query = matvec(projection, query)
    correction = (
        encoding.residual_norm
        * math.sqrt(math.pi / 2.0)
        / dimension
        * dot(projected_query, encoding.residual_signs)
    )
    return dot(query, base) + correction


def packed_bits_bytes(items: int, bits: int, alignment: int = 1) -> int:
    require(items > 0 and bits > 0, "packed dimensions must be positive")
    require(alignment > 0, "packed alignment must be positive")
    raw = (items * bits + 7) // 8
    return ((raw + alignment - 1) // alignment) * alignment


def mse_vector_bytes(
    dimension: int,
    bits: int,
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    codebook(dimension, bits)
    require(scalar_bytes in (2, 4), "norm scalar must be BF16/FP16 or FP32")
    return packed_bits_bytes(dimension, bits, alignment) + scalar_bytes


def product_vector_bytes(
    dimension: int,
    total_bits: int,
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    require(2 <= total_bits <= 5, "product bit width must be in [2, 5]")
    codebook(dimension, total_bits - 1)
    require(scalar_bytes in (2, 4), "norm scalar must be BF16/FP16 or FP32")
    indices = packed_bits_bytes(dimension, total_bits - 1, alignment)
    signs = packed_bits_bytes(dimension, 1, alignment)
    return indices + signs + 2 * scalar_bytes


def split_mse_vector_bytes(
    dimensions: Iterable[int],
    bits: Iterable[int],
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    groups = tuple(zip(dimensions, bits))
    require(groups, "mixed MSE profile must have at least one group")
    return sum(
        mse_vector_bytes(size, width, scalar_bytes=scalar_bytes, alignment=alignment)
        for size, width in groups
    )


def split_product_vector_bytes(
    dimensions: Iterable[int],
    total_bits: Iterable[int],
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    groups = tuple(zip(dimensions, total_bits))
    require(groups, "mixed product profile must have at least one group")
    return sum(
        product_vector_bytes(size, width, scalar_bytes=scalar_bytes, alignment=alignment)
        for size, width in groups
    )


def cache_payload_bytes(
    tokens: int,
    layers: int,
    kv_heads: int,
    key_bytes: int,
    value_bytes: int,
) -> int:
    require(tokens >= 0 and layers > 0 and kv_heads > 0, "invalid cache geometry")
    require(key_bytes > 0 and value_bytes > 0, "invalid cache vector size")
    return tokens * layers * kv_heads * (key_bytes + value_bytes)
