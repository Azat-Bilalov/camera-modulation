from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpectralAxis:
    """
    Дискретная спектральная ось, общая для всех этапов pipeline.

    Именно на этой сетке живут:
    - спектр источника;
    - коэффициенты отражения объекта;
    - коэффициенты пропускания оптики;
    - отклик сенсора;
    - измерения, импортированные со спектрометра.
    """

    wavelengths_nm: list[float]
    axis_name: str = "visible"
    unit: str = "nm"
    description: str = "Спектральная сетка симуляции"

    @property
    def band_count(self) -> int:
        """Количество спектральных каналов."""
        return len(self.wavelengths_nm)

    @property
    def min_wavelength_nm(self) -> float:
        """Левая граница спектрального диапазона."""
        return self.wavelengths_nm[0]

    @property
    def max_wavelength_nm(self) -> float:
        """Правая граница спектрального диапазона."""
        return self.wavelengths_nm[-1]


@dataclass
class SourceConfig:
    """
    Параметры источника света.

    `spectrum` хранится на той же спектральной оси, что и вся симуляция.
    Это позволяет в будущем напрямую подменять синтетический источник
    измеренным спектром от реального прибора.
    """

    source_type: str
    intensity: float
    spectrum: list[float]
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    is_collimated: bool = True
    description: str = "Удаленный источник с заданным спектральным распределением"


@dataclass
class ObjectConfig:
    """
    Параметры объекта или сцены.

    `reflectance_map` имеет форму `(height, width, bands)`.
    Для каждой точки сцены хранится спектральная отражательная способность.
    """

    object_name: str
    height: int
    width: int
    reflectance_map: list[list[list[float]]]
    absorption_map: list[list[list[float]]] | None = None
    texture_name: str | None = None
    description: str = "Сцена с пространственно-спектральной отражательной способностью"


@dataclass
class SpectralImage:
    """
    Центральная сущность проекта: спектральная карта сцены.

    `data[y][x][band]` описывает энергию или относительную интенсивность
    в точке сцены `(y, x)` на выбранной длине волны.
    """

    data: list[list[list[float]]]
    spectral_axis: SpectralAxis
    source_name: str
    description: str = "Спектральная карта сцены после взаимодействия света с объектом"

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0]) if self.data else 0


@dataclass
class OpticalChannel:
    """
    Промежуточный канал после части оптического тракта.

    Этот класс полезен, если команда хочет отдельно отслеживать ветви,
    которые возникают после разделения спектра призмами или масками.
    """

    channel_id: str
    data: list[list[float]]
    transmission: float | list[float]
    mask_id: str | None = None
    prism_id: str | None = None
    description: str = "Промежуточное изображение после оптического преобразования"


@dataclass
class OpticsConfig:
    """
    Конфигурация оптического тракта.

    Здесь удобно хранить описание логики:
    - как делится спектр;
    - какие есть маски;
    - какие коэффициенты потерь у элементов;
    - как каналы потом сводятся на матрицу.
    """

    channel_count: int
    split_strategy: str
    mask_pattern: str
    transmission_low: float
    transmission_high: float
    recombination_mode: str = "sum"
    description: str = "Параметры упрощенного оптического кодера"


@dataclass
class SensorExposure:
    """
    Энергия на поверхности сенсора перед накоплением заряда.

    В простом случае `irradiance` можно хранить как двумерную карту.
    В более строгой версии сюда можно добавить трехмерный спектральный слой.
    """

    irradiance: list[list[float]]
    exposure_time_s: float
    spectral_axis: SpectralAxis
    description: str = "Распределение энергии на матрице до электроники"

    @property
    def height(self) -> int:
        return len(self.irradiance)

    @property
    def width(self) -> int:
        return len(self.irradiance[0]) if self.irradiance else 0


@dataclass
class SensorConfig:
    """
    Параметры фотоматрицы.

    Сюда входят как геометрические параметры пикселей, так и физические:
    усиление, шумы, дефектные пиксели и спектральная чувствительность.
    """

    resolution: tuple[int, int]
    pixel_size_um: float
    gain: float
    dark_offset: float
    quantum_efficiency: list[float] | None = None
    defect_pixels: list[tuple[int, int]] = field(default_factory=list)
    description: str = "Параметры сенсора и его физического отклика"


@dataclass
class ChargeMatrix:
    """
    Заряд, накопленный на матрице после интегрирования экспозиции.

    Это переходная сущность между оптической моделью и цифровой электроникой.
    """

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
    """
    Конфигурация АЦП.

    Нужна для перевода аналогового сигнала в дискретный цифровой код.
    """

    bit_depth: int
    full_scale: float
    reference_voltage_v: float | None = None
    amplification: float = 1.0
    saturation_mode: str = "clip"
    description: str = "Параметры квантования аналогового сигнала"


@dataclass
class DigitalFrame:
    """
    Цифровой кадр после прохождения АЦП.

    Здесь уже можно применять коррекции, реконструкцию и экспорт в файл.
    """

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
    """
    Параметры восстановления итогового изображения.

    В первой версии тут обычно достаточно нормализации и базовой коррекции,
    но позже можно добавить денойзинг, восстановление спектральных каналов
    или компенсацию масок и оптических искажений.
    """

    normalize_to_u8: bool = True
    defect_correction: bool = True
    contrast_stretch: bool = True
    description: str = "Параметры постобработки и реконструкции"


@dataclass
class ExportConfig:
    """
    Параметры экспорта результата.

    Отделение экспорта от реконструкции помогает не смешивать физику модели
    с пользовательским форматом представления результата.
    """

    output_dir: str
    image_format: str = "bmp"
    save_intermediate: bool = True
    report_name: str = "report.txt"
    description: str = "Настройки сохранения итогового кадра и артефактов"


@dataclass
class PipelineArtifacts:
    """
    Собранный комплект результатов одного прогона pipeline.

    Удобен для интегратора и для демонстрации команде:
    в одном объекте видно, что каждая стадия производит на выходе.
    """

    axis: SpectralAxis
    source: SourceConfig
    scene: SpectralImage
    exposure: SensorExposure
    charge: ChargeMatrix
    frame: DigitalFrame
    export: ExportConfig
    optical_channels: list[OpticalChannel] = field(default_factory=list)
    description: str = "Полный набор промежуточных и итоговых сущностей pipeline"
