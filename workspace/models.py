"""
Общие модели данных для всех ролевых подпапок `workspace/`.


"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpectralAxis:
    wave: list[float]
    start: float
    stop: float
    bands_count: int


@dataclass
class SpectralImage:
    spectral_axis: SpectralAxis
    data: list[list[list[float]]]  # [y][x][band]


@dataclass
class SourceConfig:
    spectrum: list[float]
    position: list[float]  # [x, y, z]


@dataclass
class ObjectConfig:
    reflectance: list[float]
    width: int
    height: int
    point_size: float


@dataclass
class OpticsConfig:
    channel_count: int
    split_strategy: str
    mask_pattern: str
    transmission: list[float]  # [R, G, B]
    recombination_mode: str = "sum"
    rgb_ranges_nm: list[tuple[float, float]] | None = None
    # Геометрия камеры-обскуры (в тех же условных единицах, что point_size сцены).
    aperture_diameter: float = 50.0  # диаметр точечной диафрагмы
    object_distance: float = 50.0  # расстояние от плоскости объекта до диафрагмы (глубина по оси Z)
    image_distance: float = 50.0  # расстояние от диафрагмы до плоскости изображения
    # Произвольное положение и наклон диафрагмы/сенсора относительно центра сцены.
    aperture_offset_x: float = 0.0  # смещение диафрагмы по X относительно центра сцены
    aperture_offset_y: float = 0.0  # смещение диафрагмы по Y относительно центра сцены
    tilt_x_deg: float = 0.0  # наклон оптической оси вокруг X (градусы)
    tilt_y_deg: float = 0.0  # наклон оптической оси вокруг Y (градусы)
    description: str = "..."


@dataclass
class SpectralSensitivity:
    """
    Спектральная чувствительность регистрирующего элемента (фотодиода под
    цветными фильтрами Байера). Для каждой длины волны из `wave` заданы
    относительные отклики красного, зелёного и синего каналов (0..1).
    """

    wave: list[float]
    red: list[float]
    green: list[float]
    blue: list[float]
    description: str = "Кривые спектральной чувствительности R/G/B регистрирующего элемента"

    def as_matrix(self) -> list[list[float]]:
        """Возвращает чувствительность в виде [band][channel] (R, G, B)."""
        return [[self.red[i], self.green[i], self.blue[i]] for i in range(len(self.wave))]


@dataclass
class SensorExposure:
    spectral_axis: SpectralAxis
    channel_irradiance: list[list[list[float]]] | None = None  # (H, W, 3)
    description: str = "Распределение энергии на матрице до электроники"
    exposure_time_s: float = 0.01


@dataclass
class SensorConfig:
    resolution: tuple[int, int]
    pixel_size_um: float
    gain: float
    dark_offset: float
    quantum_efficiency: list[float] | None = None
    defect_pixels: list[tuple[int, int]] = field(default_factory=list)
    description: str = "Параметры сенсора и его физического отклика"


@dataclass
class ChargeMatrix:
    charge: list[list[float]]
    sensor_config: SensorConfig
    description: str = "Накопленный заряд на каждом пикселе матрицы"

    @property
    def height(self) -> int:
        return len(self.charge)

    @property
    def width(self) -> int:
        return len(self.charge[0]) if self.charge else 0


@dataclass
class AdcConfig:
    bit_depth: int
    full_scale: float
    reference_voltage_v: float | None = None
    amplification: float = 1.0
    saturation_mode: str = "clip"
    description: str = "Параметры квантования аналогового сигнала"


@dataclass
class DigitalFrame:
    data: list[list[int]]
    bit_depth: int
    saturated_mask: list[list[bool]] | None = None
    defect_mask: list[list[bool]] | None = None
    description: str = "Оцифрованный кадр после модели АЦП"

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0]) if self.data else 0


@dataclass
class ReconstructionConfig:
    normalize_to_u8: bool = True
    defect_correction: bool = True
    contrast_stretch: bool = True
    description: str = "Параметры постобработки и реконструкции"


@dataclass
class ExportConfig:
    output_dir: str
    image_format: str = "bmp"
    save_intermediate: bool = True
    report_name: str = "report.txt"
    description: str = "Настройки сохранения итогового кадра и артефактов"


@dataclass
class OpticalChannel:
    channel_id: str
    data: list[list[float]]
    transmission: float | list[float]
    mask_id: str | None = None
    prism_id: str | None = None
    description: str = "Промежуточное изображение после оптического преобразования"


@dataclass
class PipelineArtifacts:
    axis: SpectralAxis
    source: SourceConfig
    scene: SpectralImage
    exposure: SensorExposure
    charge: ChargeMatrix
    frame: DigitalFrame
    export: ExportConfig
    optical_channels: list[OpticalChannel] = field(default_factory=list)
    description: str = "Полный набор промежуточных и итоговых сущностей pipeline"


__all__ = [
    "AdcConfig",
    "ChargeMatrix",
    "DigitalFrame",
    "ExportConfig",
    "ObjectConfig",
    "OpticalChannel",
    "OpticsConfig",
    "PipelineArtifacts",
    "ReconstructionConfig",
    "SensorConfig",
    "SensorExposure",
    "SourceConfig",
    "SpectralAxis",
    "SpectralImage",
    "SpectralSensitivity",
]
