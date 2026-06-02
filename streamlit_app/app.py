"""
Streamlit-обёртка над workspace pipeline.

Запуск:
    source .venv/bin/activate && streamlit run streamlit_app/app.py

Все расчёты выполняются через модули workspace/,
никакие исходные файлы проекта не модифицируются.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from workspace.models import (
    AdcConfig,
    ChargeMatrix,
    DigitalFrame,
    ExportConfig,
    OpticalChannel,
    OpticsConfig,
    PipelineArtifacts,
    SensorConfig,
)
from workspace.optics.optics_transformer import (
    convert_scene_to_exposure,
)
from workspace.scene_source.scene_models import (
    SceneSourceInput,
    build_axis_by_step,
    build_scene_source,
    read_spectrum_from_csv,
)
from workspace.sensor_adc.sensor_pipeline import (
    build_rgb_frame,
)
from workspace.shared import (
    normalize_frame_to_u8,
    normalize_rgb_to_u8,
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
    page_icon="",
    layout="wide",
)

st.title(" Симуляция камеры: спектральный пайплайн")
st.caption("Все расчёты — через `workspace/`. Исходные модули проекта не изменены.")


# ──────────────────────────────────────────────────────────────
# Боковая панель — параметры симуляции
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("️ Конфигурация пайплайна")

    with st.expander(" Сцена и источник", expanded=True):
        st.markdown("** Источник света**")
        src_x = st.slider("X источника", -500, 500, 10, 10)
        src_y = st.slider("Y источника", -500, 500, 10, 10)
        src_z = st.slider("Z источника (высота)", 1, 1000, 50, 10)

        st.markdown("** Объект (плоскость)**")
        obj_width = st.slider("Ширина, px", 4, 128, 32, 4)
        obj_height = st.slider("Высота, px", 4, 128, 32, 4)
        point_size = st.slider("Размер точки (pixel_pitch)", 1.0, 50.0, 10.0, 1.0)

        st.markdown("** Отражение (reflectance)**")
        reflectance_file = st.file_uploader(
            "CSV со спектром (колонка `value`)", type=["csv"]
        )
        if reflectance_file is None:
            default_csv = Path(__file__).parent / "sample_spectrum.csv"
            reflectance_values = read_spectrum_from_csv(str(default_csv))
        else:
            df = pd.read_csv(io.BytesIO(reflectance_file.getvalue()))
            if "value" not in df.columns:
                st.error("В CSV отсутствует колонка `value`!")
                st.stop()
            reflectance_values = df["value"].astype(float).tolist()

        max_val = max(reflectance_values)
        reflectance = [v / max_val for v in reflectance_values]

    with st.expander(" Оптический тракт (RGB)", expanded=False):
        transmission_r = st.slider("Пропускание R", 0.0, 1.0, 0.95, 0.01)
        transmission_g = st.slider("Пропускание G", 0.0, 1.0, 0.90, 0.01)
        transmission_b = st.slider("Пропускание B", 0.0, 1.0, 0.85, 0.01)

    with st.expander(" Сенсор", expanded=False):
        gain = st.slider("Gain", 100.0, 50000.0, 2000.0, 100.0)
        dark_offset = st.slider("Dark offset", 0.0, 1.0, 0.002, 0.001)

    with st.expander(" АЦП", expanded=False):
        bit_depth = st.slider("Разрядность, бит", 8, 16, 10, 1)
        full_scale = st.slider("Full scale", 1.0, 1000.0, 8.0, 1.0)

    st.markdown("**️ Отображение**")
    auto_normalize = st.checkbox(
        "Автонормализация яркости (min-max stretch)",
        value=True,
        help="Если включено — яркость растягивается на весь диапазон 0..255. "
        "Если выключено — показываются физические значения АЦП (будет темно при слабом сигнале).",
    )

    # Чекбокс для показа 3D-сцены
    show_3d = st.checkbox(" Показать 3D-сцену", key="show_3d_scene")

    run = st.button("▶ Запустить симуляцию", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# 3D-визуализация геометрии
# ──────────────────────────────────────────────────────────────
if show_3d:
    st.subheader("️ 3D-визуализация геометрии")

    # Размеры плоскости объекта
    plane_w = (obj_width - 1) * point_size
    plane_h = (obj_height - 1) * point_size
    plane_z = 0.0

    # Центр плоскости
    cx, cy = plane_w / 2, plane_h / 2

    dx = cx - src_x
    dy = cy - src_y
    dz = plane_z - src_z
    distance = max(np.sqrt(dx * dx + dy * dy + dz * dz), 0.001)
    angle = np.degrees(np.arccos(abs(dz) / distance)) if distance > 0 else 0

    fig = go.Figure()

    # Источник света
    fig.add_trace(
        go.Scatter3d(
            x=[src_x],
            y=[src_y],
            z=[src_z],
            mode="markers+text",
            marker=dict(
                size=14,
                color="#FFD700",
                symbol="circle",
                line=dict(color="#FF8C00", width=2),
            ),
            text=[""],
            textposition="top center",
            textfont=dict(size=16, color="#FFD700"),
            name=" Источник",
        )
    )

    # Плоскость объекта (полупрозрачная сетка)
    corners_x = [0, plane_w, plane_w, 0]
    corners_y = [0, 0, plane_h, plane_h]
    corners_z = [plane_z, plane_z, plane_z, plane_z]

    fig.add_trace(
        go.Mesh3d(
            x=corners_x,
            y=corners_y,
            z=corners_z,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#FF4444",
            opacity=0.25,
            name=" Объект (плоскость)",
            hovertemplate="Объект: %{x:.1f}, %{y:.1f}, %{z:.1f}<extra></extra>",
        )
    )

    # Контур плоскости
    fig.add_trace(
        go.Scatter3d(
            x=corners_x + [corners_x[0]],
            y=corners_y + [corners_y[0]],
            z=corners_z + [corners_z[0]],
            mode="lines",
            line=dict(color="#8B0000", width=3),
            name=" Граница объекта",
            hoverinfo="skip",
        )
    )

    # Лучи от источника к углам плоскости
    for i in range(4):
        fig.add_trace(
            go.Scatter3d(
                x=[src_x, corners_x[i]],
                y=[src_y, corners_y[i]],
                z=[src_z, corners_z[i]],
                mode="lines",
                line=dict(color="#FFA500", width=2, dash="dash"),
                name=" Луч" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="skip",
            )
        )

    size = max(abs(src_x), abs(src_y), abs(src_z), plane_w, plane_h, 100)
    fig.update_layout(
        title=dict(
            text=f" Расстояние до центра: {distance:.0f} |  Угол: {angle:.1f}°",
            font=dict(size=14, color="#2C3E50"),
            x=0.5,
        ),
        height=550,
        scene=dict(
            xaxis_title="<b>X</b>",
            yaxis_title="<b>Y</b>",
            zaxis_title="<b>Z</b>",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor="#F5F5F5",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric(" Источник", f"({src_x}, {src_y}, {src_z})")
    with col_info2:
        st.metric(" Объект", f"{plane_w:.0f}×{plane_h:.0f} мм")
    with col_info3:
        attenuation = 1.0 / (distance**2 + 1e-6)
        st.metric(" 1/r²", f"{attenuation:.6f}")

    st.divider()


# ──────────────────────────────────────────────────────────────
# Пайплайн
# ──────────────────────────────────────────────────────────────
def _normalize_2d(arr_2d: list[list[float]]) -> np.ndarray:
    """Нормализует 2D float-массив в 0..255 для отображения."""
    flat = [v for row in arr_2d for v in row]
    min_v, max_v = min(flat), max(flat)
    if max_v == min_v:
        return np.zeros((len(arr_2d), len(arr_2d[0])), dtype=np.uint8)
    return np.array(
        [[int(255 * (v - min_v) / (max_v - min_v)) for v in row] for row in arr_2d],
        dtype=np.uint8,
    )


def run_pipeline():
    """Собирает весь пайплайн с текущими параметрами sidebar."""

    # 1. Спектральная ось (по длине reflectance)
    axis = build_axis_by_step(
        start=380.0,
        step=10.0,
        count=len(reflectance),
    )

    # 2. Сцена и источник
    scene_input = SceneSourceInput(
        radiation=[1.0] * len(reflectance),
        source_xyz=[float(src_x), float(src_y), float(src_z)],
        reflectance=reflectance,
        object_width=obj_width,
        object_height=obj_height,
        point_size=point_size,
    )
    scene_source = build_scene_source(scene_input)

    # 3. Оптика
    optics_config = OpticsConfig(
        channel_count=3,
        split_strategy="rgb spectral split",
        mask_pattern="none",
        transmission=[transmission_r, transmission_g, transmission_b],
        recombination_mode="multi-channel",
        rgb_ranges_nm=[(380, 480), (480, 600), (600, 730)],
        description="Интерактивная RGB оптика",
    )
    exposure = convert_scene_to_exposure(scene_source.scene, optics_config)

    # 4. Сенсор
    sensor_config = SensorConfig(
        resolution=(obj_height, obj_width),
        pixel_size_um=4.8,
        gain=gain,
        dark_offset=dark_offset,
        quantum_efficiency=[1.0 for _ in range(axis.bands_count)],
        description="Интерактивный сенсор",
    )

    # 5. АЦП
    adc_config = AdcConfig(
        bit_depth=bit_depth,
        full_scale=full_scale,
        reference_voltage_v=3.3,
        amplification=1.0,
        saturation_mode="clip",
        description="Интерактивный АЦП",
    )

    # 6. Формирование RGB-кадра
    rgb_int = build_rgb_frame(exposure, sensor_config, adc_config)

    # 7. Grayscale-заглушки для отчёта
    grayscale_data = [[sum(pixel) // 3 for pixel in row] for row in rgb_int]
    frame = DigitalFrame(
        data=grayscale_data,
        bit_depth=bit_depth,
    )
    charge = ChargeMatrix(
        charge=[[float(v) for v in row] for row in grayscale_data],
        sensor_config=sensor_config,
    )

    artifacts = PipelineArtifacts(
        axis=axis,
        source=scene_source.source,
        scene=scene_source.scene,
        exposure=exposure,
        charge=charge,
        frame=frame,
        export=ExportConfig(output_dir="workspace/outputs"),
        optical_channels=[
            OpticalChannel(
                channel_id="R",
                data=[[p[0] for p in row] for row in exposure.channel_irradiance or []],
                transmission=transmission_r,
            ),
            OpticalChannel(
                channel_id="G",
                data=[[p[1] for p in row] for row in exposure.channel_irradiance or []],
                transmission=transmission_g,
            ),
            OpticalChannel(
                channel_id="B",
                data=[[p[2] for p in row] for row in exposure.channel_irradiance or []],
                transmission=transmission_b,
            ),
        ],
    )

    return artifacts, rgb_int, exposure, sensor_config, adc_config


if run:
    with st.spinner("Симуляция запущена…"):
        artifacts, rgb_int, exposure, sensor_config, adc_config = run_pipeline()

        if auto_normalize:
            image = normalize_rgb_to_u8(rgb_int)
        else:
            max_code = (1 << adc_config.bit_depth) - 1
            image = [
                [
                    [max(0, min(255, int(v * 255 / max_code))) for v in pixel]
                    for pixel in row
                ]
                for row in rgb_int
            ]
        image_np = np.array(image, dtype=np.uint8)

    st.success("Симуляция завершена!")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("️ Итоговое RGB-изображение")
        if auto_normalize:
            st.caption("Режим: автонормализация (min-max stretch + gamma)")
        else:
            st.caption("Режим: физические значения АЦП (без растягивания)")
        st.image(image_np, use_container_width=True)

        tabs = st.tabs(["Канал R", "Канал G", "Канал B", "Заряд", "Кадр АЦП"])

        exp = exposure.channel_irradiance

        with tabs[0]:
            if exp:
                st.image(
                    _normalize_2d([[p[0] for p in row] for row in exp]),
                    caption="Экспозиция R",
                    use_container_width=True,
                )
        with tabs[1]:
            if exp:
                st.image(
                    _normalize_2d([[p[1] for p in row] for row in exp]),
                    caption="Экспозиция G",
                    use_container_width=True,
                )
        with tabs[2]:
            if exp:
                st.image(
                    _normalize_2d([[p[2] for p in row] for row in exp]),
                    caption="Экспозиция B",
                    use_container_width=True,
                )
        with tabs[3]:
            charge_img = normalize_frame_to_u8(artifacts.frame.data)
            st.image(
                np.array(charge_img, dtype=np.uint8),
                caption="Усреднённый заряд (grayscale)",
                use_container_width=True,
            )
        with tabs[4]:
            st.image(
                np.array(normalize_frame_to_u8(artifacts.frame.data), dtype=np.uint8),
                caption="Цифровой кадр после АЦП",
                use_container_width=True,
            )

    with col_right:
        st.subheader(" Статистика")

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
            st.warning("Значительный клиппинг — уменьшите gain или full_scale.")

    # ── Графики ──
    st.divider()
    g1, g2 = st.columns(2)

    with g1:
        st.subheader(" Спектр отражения (reflectance)")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(artifacts.axis.wave, reflectance, color="seagreen", lw=2)
        ax.fill_between(artifacts.axis.wave, reflectance, alpha=0.2, color="seagreen")
        ax.set_xlabel("Длина волны, нм")
        ax.set_ylabel("Коэффициент отражения [0..1]")
        ax.set_title("Reflectance объекта")
        ax.grid(True, ls="--", alpha=0.4)
        st.pyplot(fig, use_container_width=True)

    with g2:
        st.subheader(" Гистограмма кадра")
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
    with st.expander(" Подробный текстовый отчёт"):
        report_lines = [
            "=" * 60,
            "ОТЧЁТ ПО ИНТЕРАКТИВНОЙ СИМУЛЯЦИИ",
            "=" * 60,
            "",
            "--- ОСНОВНАЯ ИНФОРМАЦИЯ ---",
            f"Спектральных диапазонов: {artifacts.axis.bands_count}",
            f"Оптические каналы: {[ch.channel_id for ch in artifacts.optical_channels]}",
            f"Размер сцены: {len(artifacts.scene.data)}×{len(artifacts.scene.data[0])}",
            "",
            "--- ПАРАМЕТРЫ ---",
            f"Источник: {artifacts.source.position}",
            f"Gain: {artifacts.charge.sensor_config.gain}",
            f"Dark offset: {artifacts.charge.sensor_config.dark_offset}",
            f"Full scale: {adc_config.full_scale}",
            f"Bit depth: {adc_config.bit_depth}",
            "",
            "--- ПРОМЕЖУТОЧНЫЕ ДАННЫЕ ---",
            f"Экспозиция: {len(artifacts.exposure.channel_irradiance or [])}x{len((artifacts.exposure.channel_irradiance or [[]])[0])}x3",
            f"Заряд: {summarize_matrix(artifacts.charge.charge)}",
            "",
            "--- ЦИФРОВОЙ КАДР ---",
            f"Размер: {frame_stats['height']}x{frame_stats['width']}",
            f"Битность: {frame_stats['bit_depth']} бит",
            f"Диапазон: {frame_stats['dynamic_range']}",
            f"Среднее: {frame_stats['mean']}",
            "",
            "--- ВЕРИФИКАЦИЯ ---",
            f"Диапазон корректен: {range_check['is_valid']}",
            f"Клиппинг допустим: {clip_check['is_acceptable']}",
            "",
            "=" * 60,
        ]
        st.text("\n".join(report_lines))

else:
    st.info(
        "Настройте параметры в боковой панели и нажмите **▶ Запустить симуляцию**.",
    )
