"""
Streamlit-обёртка над workspace pipeline.

Запуск:
    source .venv/bin/activate && streamlit run streamlit_app/app.py

Все расчёты выполняются через модули workspace/ и draft/models,
никакие исходные файлы проекта не модифицируются.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH, чтобы импортировать workspace
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from workspace.models import (
    AdcConfig,
    ChargeMatrix,
    DigitalFrame,
    ObjectConfig,
    OpticalChannel,
    OpticsConfig,
    PipelineArtifacts,
    ReconstructionConfig,
    SensorConfig,
    SensorExposure,
    SourceConfig,
    SpectralAxis,
    SpectralImage,
)
from workspace.shared import (
    build_default_reflectance_map,
    create_default_axis,
    gaussian_spectrum,
    integrate_sensor_charge,
    normalize_frame_to_u8,
    normalize_vector,
    quantize_frame,
    simulate_scene_matrix,
    split_optical_channels,
    summarize_matrix,
)
from workspace.visualization.verifier import (
    calculate_image_statistics,
    verify_digital_range,
    verify_no_clipping,
)

# ──────────────────────────────────────────────────────────────
# Конфигурация страницы
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Camera Modulation Simulator",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Симуляция камеры: спектральный пайплайн")
st.caption("Все расчёты — через `workspace/`. Исходные модули проекта не изменены.")

# ──────────────────────────────────────────────────────────────
# Боковая панель — параметры симуляции
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Параметры симуляции")

    with st.expander("Спектральная ось", expanded=False):
        start_nm = st.slider("Начало, нм", 300, 900, 400, 10)
        stop_nm = st.slider("Конец, нм", 400, 1000, 700, 10)
        step_nm = st.slider("Шаг, нм", 5, 100, 20, 5)
        if start_nm >= stop_nm:
            st.error("Начало диапазона должно быть меньше конца!")
            st.stop()

    with st.expander("Источник света", expanded=True):
        source_mode = st.radio(
            "Тип источника",
            ["Синтетический гаусс", "Загрузить спектр из CSV/TXT"],
        )
        intensity = st.slider("Интенсивность", 0.1, 10.0, 1.0, 0.1)
        center_nm = 560
        width_nm = 90
        uploaded_file = None
        if source_mode == "Синтетический гаусс":
            center_nm = st.slider("Центр спектра, нм", 400, 700, 560, 10)
            width_nm = st.slider("Ширина спектра, нм", 1, 300, 90, 5)
        else:
            uploaded_file = st.file_uploader("CSV/TXT со спектром", type=["csv", "txt"])
            sample_path = Path(__file__).parent / "sample_spectrum.csv"
            if sample_path.exists():
                st.download_button(
                    "📥 Скачать пример CSV",
                    sample_path.read_bytes(),
                    file_name="sample_spectrum.csv",
                    mime="text/csv",
                )

    with st.expander("Сцена", expanded=False):
        height = st.slider("Высота, px", 4, 256, 32, 4)
        width = st.slider("Ширина, px", 4, 256, 40, 4)

    with st.expander("Оптический тракт", expanded=False):
        transmission_low = st.slider("Пропускание LOW", 0.0, 1.0, 0.95, 0.01)
        transmission_high = st.slider("Пропускание HIGH", 0.0, 1.0, 0.90, 0.01)

    with st.expander("Сенсор", expanded=False):
        gain = st.slider("Gain", 100.0, 20000.0, 2000.0, 100.0)
        dark_offset = st.slider("Dark offset", 0.0, 1.0, 0.002, 0.001)
        exposure_time_s = st.slider("Время экспозиции, с", 0.001, 1.0, 0.01, 0.001)

    with st.expander("АЦП", expanded=False):
        bit_depth = st.slider("Разрядность, бит", 8, 16, 10, 1)
        full_scale = st.slider("Full scale", 100.0, 10000.0, 900.0, 50.0)

    with st.expander("📐 Геометрия источник ↔ приёмник", expanded=True):
        st.markdown("**💡 Источник света**")
        src_x = st.slider("X источника", -1000, 1000, 0, 10)
        src_y = st.slider("Y источника", -1000, 1000, 0, 10)
        src_z = st.slider("Z источника (высота)", 1, 1000, 200, 10)

        st.markdown("**📦 Приёмник (сенсор)**")
        rx = st.slider("X приёмника", -1000, 1000, 0, 10)
        ry = st.slider("Y приёмника", -1000, 1000, 0, 10)
        rz = st.slider("Z приёмника", 0, 100, 0, 1)

        # Расчёт геометрических параметров
        dx = rx - src_x
        dy = ry - src_y
        dz = rz - src_z
        distance_to_receiver = np.sqrt(dx * dx + dy * dy + dz * dz) if (dx * dx + dy * dy + dz * dz) > 0 else 0.001
        angle_of_incidence = np.degrees(np.arccos(abs(dz) / distance_to_receiver)) if distance_to_receiver > 0 else 0

        st.caption(f"📏 Расстояние: {distance_to_receiver:.0f} мм | 📐 Угол падения: {angle_of_incidence:.1f}°")

        if distance_to_receiver > 1000:
            st.warning("⚠️ Слишком большое расстояние — сигнал будет очень слабым!")

        # Чекбокс для показа 3D-сцены на главном экране
        show_3d = st.checkbox("✨ Показать 3D-сцену", key="show_3d_scene")

    run = st.button("▶ Запустить симуляцию", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# ОТОБРАЖЕНИЕ 3D-СЦЕНЫ НА ГЛАВНОМ ЭКРАНЕ
# ──────────────────────────────────────────────────────────────
if show_3d:
    st.subheader("🗺️ 3D-визуализация геометрии")

    # Расчёт параметров для отображения
    dx = rx - src_x
    dy = ry - src_y
    dz = rz - src_z
    distance = np.sqrt(dx * dx + dy * dy + dz * dz) if (dx * dx + dy * dy + dz * dz) > 0 else 0.001
    angle = np.degrees(np.arccos(abs(dz) / distance)) if distance > 0 else 0

    # Создание красивой 3D-сцены
    fig = go.Figure()

    # Источник света
    fig.add_trace(go.Scatter3d(
        x=[src_x], y=[src_y], z=[src_z],
        mode='markers+text',
        marker=dict(size=14, color='#FFD700', symbol='circle', line=dict(color='#FF8C00', width=2)),
        text=['💡'], textposition='top center',
        textfont=dict(size=16, color='#FFD700'),
        name='💡 Источник'
    ))

    # Приёмник
    fig.add_trace(go.Scatter3d(
        x=[rx], y=[ry], z=[rz],
        mode='markers+text',
        marker=dict(size=11, color='#FF4444', symbol='square', line=dict(color='#8B0000', width=2)),
        text=['📷'], textposition='top center',
        textfont=dict(size=14, color='#FF4444'),
        name='📷 Приёмник'
    ))

    # Луч света
    fig.add_trace(go.Scatter3d(
        x=[src_x, rx], y=[src_y, ry], z=[src_z, rz],
        mode='lines',
        line=dict(color='#FFA500', width=5, dash='solid'),
        name='✨ Световой луч'
    ))

    # Границы пространства
    size = 1000
    corners = [[-size, -size, 0], [size, -size, 0], [size, size, 0], [-size, size, 0], [-size, -size, 0]]
    fig.add_trace(go.Scatter3d(
        x=[c[0] for c in corners], y=[c[1] for c in corners], z=[c[2] for c in corners],
        mode='lines', line=dict(color='#4A90D9', width=2, dash='dash'),
        name='📐 Граница пространства'
    ))

    # Настройки внешнего вида
    fig.update_layout(
        title=dict(
            text=f"📏 Расстояние: {distance:.0f} мм | 📐 Угол падения: {angle:.1f}°",
            font=dict(size=14, color='#2C3E50'),
            x=0.5
        ),
        height=550,
        scene=dict(
            xaxis_title='<b>X (мм)</b>',
            yaxis_title='<b>Y (мм)</b>',
            zaxis_title='<b>Z (мм)</b>',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='#F5F5F5',
            xaxis=dict(gridcolor='#E0E0E0'),
            yaxis=dict(gridcolor='#E0E0E0'),
            zaxis=dict(gridcolor='#E0E0E0'),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
    )

    st.plotly_chart(fig, use_container_width=True)

    # Информационная панель
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("💡 Источник", f"({src_x}, {src_y}, {src_z}) мм")
    with col_info2:
        st.metric("📷 Приёмник", f"({rx}, {ry}, {rz}) мм")
    with col_info3:
        attenuation = 1.0 / ((distance/1000)**2 + 1e-6)
        st.metric("⚡ Ослабление", f"{attenuation:.3f}")

    st.divider()


# ──────────────────────────────────────────────────────────────
# Парсер загруженного спектра
# ──────────────────────────────────────────────────────────────
def _parse_uploaded_spectrum(uploaded_file) -> tuple[list[float], list[float]]:
    """Возвращает (wavelengths_nm, values) из CSV/TXT."""
    bytes_data = uploaded_file.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(bytes_data), sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.BytesIO(bytes_data), sep=r"\s+", engine="python")
    if len(df.columns) >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["wavelength_nm", "value"]
    elif len(df.columns) == 1:
        df.columns = ["value"]
        df["wavelength_nm"] = [380 + i * 10 for i in range(len(df))]
    else:
        raise ValueError("Не удалось распознать формат файла")
    return df["wavelength_nm"].astype(float).tolist(), df["value"].astype(
        float
    ).tolist()


# ──────────────────────────────────────────────────────────────
# Выполнение пайплайна
# ──────────────────────────────────────────────────────────────
def run_pipeline(
        src_x: float, src_y: float, src_z: float,
        rx: float, ry: float, rz: float,
) -> tuple:
    # 1. Спектральная ось
    axis = create_default_axis(start_nm=start_nm, stop_nm=stop_nm, step_nm=step_nm)

    # 2. Источник
    if source_mode == "Синтетический гаусс":
        source_spectrum = gaussian_spectrum(
            axis, center_nm=center_nm, width_nm=width_nm, amplitude=intensity
        )
        source_type = "synthetic-gaussian"
        source_description = f"Гаусс: λ₀={center_nm} нм, Δλ={width_nm} нм"
    else:
        if uploaded_file is None:
            st.error("Загрузите CSV/TXT со спектром!")
            st.stop()
        csv_wavelengths, csv_values = _parse_uploaded_spectrum(uploaded_file)
        source_spectrum = np.interp(
            axis.wavelengths_nm,
            csv_wavelengths,
            csv_values,
            left=csv_values[0],
            right=csv_values[-1],
        ).tolist()
        source_spectrum = normalize_vector(source_spectrum)
        source_type = "uploaded-csv"
        source_description = f"Загруженный спектр ({uploaded_file.name})"

    # ─── ГЕОМЕТРИЧЕСКОЕ ОСЛАБЛЕНИЕ ──────────────────────────
    dx = rx - src_x
    dy = ry - src_y
    dz = rz - src_z
    distance = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1.0)  # мм
    distance_m = distance / 1000.0
    attenuation_distance = 1.0 / (distance_m * distance_m + 1e-6)
    angle_rad = np.arccos(abs(dz) / distance) if distance > 0 else 0
    attenuation_angle = max(0.0, np.cos(angle_rad))
    total_attenuation = attenuation_distance * attenuation_angle

    # Применяем ослабление к интенсивности источника
    effective_intensity = intensity * total_attenuation
    # Применяем ослабление к спектру
    source_spectrum = [s * total_attenuation for s in source_spectrum]
    # ────────────────────────────────────────────────────────

    source = SourceConfig(
        spectrum=source_spectrum,
        position=[src_x, src_y, src_z],
    )

    # # 3. Объект / сцена
    # reflectance_map = build_default_reflectance_map(
    #     height=height, width=width, axis=axis
    # )
    # obj = ObjectConfig(
    #     object_name="test-target",
    #     height=height,
    #     width=width,
    #     reflectance_map=reflectance_map,
    #     description="Интерактивная сцена",
    # )
    # scene_data = simulate_scene_matrix(axis, source_spectrum, reflectance_map)
    # scene = SpectralImage(
    #     data=scene_data,
    #     spectral_axis=axis,
    #     description="Спектральная карта сцены",
    # )

    # 3. Объект / сцена с пятном, зависящим от геометрии
    # 3. Объект / сцена с пятном, зависящим от геометрии
    scene_data = []
    for y in range(height):
        row = []
        for x in range(width):
            # Нормализованные координаты [-1..1]
            nx = (2 * x - width) / width
            ny = (2 * y - height) / height

            # Смещение пятна в зависимости от положения источника
            spot_x = nx - src_x / 500
            spot_y = ny - src_y / 500

            # Интенсивность пятна (убывает от центра)
            distance_from_center = np.sqrt(spot_x ** 2 + spot_y ** 2)
            spot_intensity = max(0, 1 - distance_from_center)

            # Заполняем спектр
            pixel = [spot_intensity * s for s in source_spectrum]
            row.append(pixel)
        scene_data.append(row)

    scene = SpectralImage(
        data=scene_data,
        spectral_axis=axis,
        description="Сцена с пятном, зависящим от геометрии",
    )

    # 4. Оптика
    optics_config = OpticsConfig(
        channel_count=2,
        split_strategy="low-high spectral split",
        mask_pattern="checkerboard amplitude mask",
        transmission_low=transmission_low,
        transmission_high=transmission_high,
        recombination_mode="sum",
        description="Интерактивная оптика",
    )
    optical_data = split_optical_channels(scene_data, axis)
    channels = [
        OpticalChannel(
            channel_id="low",
            data=optical_data["channel_low"],
            transmission=transmission_low,
            mask_id=optics_config.mask_pattern,
            prism_id="P1",
            description="Коротковолновый канал",
        ),
        OpticalChannel(
            channel_id="high",
            data=optical_data["channel_high"],
            transmission=transmission_high,
            mask_id=optics_config.mask_pattern,
            prism_id="P2",
            description="Длинноволновый канал",
        ),
    ]

    exposure = SensorExposure(
        irradiance=optical_data["sensor_exposure"],
        exposure_time_s=exposure_time_s,
        spectral_axis=axis,
        description="Экспозиция на сенсоре",
    )

    # 5. Сенсор
    sensor_config = SensorConfig(
        resolution=(height, width),
        pixel_size_um=4.8,
        gain=gain,
        dark_offset=dark_offset,
        quantum_efficiency=[1.0 for _ in axis.wavelengths_nm],
        description="Интерактивный сенсор",
    )
    charge_map = integrate_sensor_charge(exposure.irradiance, gain, dark_offset)
    charge = ChargeMatrix(
        charge=charge_map,
        sensor_config=sensor_config,
        description="Накопленный заряд",
    )

    # 6. АЦП
    adc_config = AdcConfig(
        bit_depth=bit_depth,
        full_scale=full_scale,
        reference_voltage_v=3.3,
        amplification=1.0,
        saturation_mode="clip",
        description="Интерактивный АЦП",
    )
    frame_data = quantize_frame(charge_map, bit_depth, full_scale)
    frame = DigitalFrame(
        data=frame_data,
        bit_depth=bit_depth,
        description="Цифровой кадр",
    )

    # 7. Реконструкция и preview
    preview = normalize_frame_to_u8(frame_data)

    artifacts = PipelineArtifacts(
        axis=axis,
        source=source,
        scene=scene,
        exposure=exposure,
        charge=charge,
        frame=frame,
        export=None,  # type: ignore[arg-type]
        optical_channels=channels,
        description="Интерактивный прогон",
    )

    return (
        artifacts,
        np.array(preview, dtype=np.uint8),
        np.array(optical_data["channel_low"]),
        np.array(optical_data["channel_high"]),
        np.array(exposure.irradiance),
        np.array(charge_map),
        source_spectrum,
    )


if run:
    with st.spinner("Симуляция запущена…"):
        (
            artifacts,
            preview_img,
            ch_low,
            ch_high,
            exposure_arr,
            charge_arr,
            source_spectrum,
        ) = run_pipeline(
            src_x, src_y, src_z,
            rx, ry, rz,
        )

    # ── Результаты ──
    st.success("Симуляция завершена!")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🖼️ Итоговое изображение")
        st.image(preview_img, use_container_width=True)

        # Промежуточные карты
        tabs = st.tabs(["Экспозиция", "Заряд", "Канал LOW", "Канал HIGH"])

        def _to_img(arr):
            return np.array(normalize_frame_to_u8(arr.tolist()), dtype=np.uint8)

        with tabs[0]:
            st.image(
                _to_img(exposure_arr),
                caption="Экспозиция на сенсоре",
                use_container_width=True,
            )
        with tabs[1]:
            st.image(
                _to_img(charge_arr),
                caption="Накопленный заряд",
                use_container_width=True,
            )
        with tabs[2]:
            st.image(
                _to_img(ch_low),
                caption="Коротковолновый канал",
                use_container_width=True,
            )
        with tabs[3]:
            st.image(
                _to_img(ch_high),
                caption="Длинноволновый канал",
                use_container_width=True,
            )

    with col_right:
        st.subheader("📊 Статистика")

        frame_stats = calculate_image_statistics(
            artifacts.frame.data, artifacts.frame.bit_depth
        )
        range_check = verify_digital_range(
            artifacts.frame.data, artifacts.frame.bit_depth
        )
        clip_check = verify_no_clipping(artifacts.frame.data, artifacts.frame.bit_depth)

        st.metric("Разрешение", f"{frame_stats['height']}×{frame_stats['width']}")
        st.metric("Битность", f"{frame_stats['bit_depth']} бит")
        st.metric("Динамический диапазон", frame_stats["dynamic_range"])
        st.metric("Среднее значение", frame_stats["mean"])
        st.metric(
            "Клиппинг (max)",
            f"{clip_check['clipped_high']} px ({clip_check['clipped_high_percent']}%)",
        )
        st.metric(
            "Клиппинг (min)",
            f"{clip_check['clipped_low']} px ({clip_check['clipped_low_percent']}%)",
        )

        if not range_check["is_valid"]:
            st.error("Значения выходят за допустимый диапазон АЦП!")
        if not clip_check["is_acceptable"]:
            st.warning("Значительный клиппинг — уменьшите gain или интенсивность.")

        with st.expander("Сырые данные верификации"):
            st.json(
                {
                    "frame_stats": frame_stats,
                    "range_check": range_check,
                    "clip_check": clip_check,
                }
            )

    # ── Графики ──
    st.divider()
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("📈 Спектр источника")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(artifacts.axis.wavelengths_nm, source_spectrum, color="crimson", lw=2)
        ax.fill_between(
            artifacts.axis.wavelengths_nm, source_spectrum, alpha=0.2, color="crimson"
        )
        ax.set_xlabel("Длина волны, нм")
        ax.set_ylabel("Относительная интенсивность")
        ax.set_title("Спектр источника")
        ax.grid(True, ls="--", alpha=0.4)
        st.pyplot(fig, use_container_width=True)

    with g2:
        st.subheader("📊 Гистограмма кадра")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        flat_values = [v for row in artifacts.frame.data for v in row]
        ax2.hist(
            flat_values,
            bins=min(50, len(set(flat_values)) + 1),
            color="steelblue",
            edgecolor="white",
        )
        ax2.set_xlabel("Цифровое значение пикселя")
        ax2.set_ylabel("Количество")
        ax2.set_title("Распределение яркости")
        ax2.grid(True, ls="--", alpha=0.4)
        st.pyplot(fig2, use_container_width=True)

    # ── Текстовый отчёт ──
    st.divider()
    with st.expander("📝 Подробный текстовый отчёт"):
        report_lines = [
            "=" * 60,
            "ОТЧЁТ ПО РАБОТЕ ИНТЕРАКТИВНОЙ СИМУЛЯЦИИ",
            "=" * 60,
            "",
            "--- ОСНОВНАЯ ИНФОРМАЦИЯ ---",
            f"Спектральных диапазонов: {artifacts.axis.bands_count}",
            f"Оптические каналы: {[ch.channel_id for ch in artifacts.optical_channels]}",
             f"Размер сцены: {len(artifacts.scene.data)}×{len(artifacts.scene.data[0])}",
            "",
            "--- ПАРАМЕТРЫ СЕНСОРА ---",
            f"Gain: {artifacts.charge.sensor_config.gain}",
            f"Dark offset: {artifacts.charge.sensor_config.dark_offset}",
            f"Экспозиция: {summarize_matrix(artifacts.exposure.irradiance)}",
            f"Заряд: {summarize_matrix(artifacts.charge.charge)}",
            "",
            "--- ЦИФРОВОЙ КАДР ---",
            f"Размер: {frame_stats['height']}x{frame_stats['width']}",
            f"Битность: {frame_stats['bit_depth']} бит",
            f"Диапазон значений: {frame_stats['dynamic_range']}",
            f"Среднее значение: {frame_stats['mean']}",
            "",
            "--- ВЕРИФИКАЦИЯ ---",
            f"Диапазон корректен: {range_check['is_valid']}",
            f"Клиппинг (макс): {clip_check['clipped_high']} пикс. ({clip_check['clipped_high_percent']}%)",
            f"Клиппинг (мин): {clip_check['clipped_low']} пикс. ({clip_check['clipped_low_percent']}%)",
            f"Клиппинг допустим: {clip_check['is_acceptable']}",
            "",
            "=" * 60,
        ]
        st.text("\n".join(report_lines))

else:
    st.info(
        "Настройте параметры в боковой панели и нажмите **▶ Запустить симуляцию**.",
        icon="👈",
    )