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

st.title("Симуляция камеры: спектральный пайплайн")


# ──────────────────────────────────────────────────────────────
# Боковая панель — параметры симуляции
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Конфигурация пайплайна")

    with st.expander(" Сцена и источник", expanded=True):
        st.markdown("**Источник света**")
        src_x = st.slider("X источника", -500, 500, 10, 10)
        src_y = st.slider("Y источника", -500, 500, 10, 10)
        src_z = st.slider("Z источника (высота)", 1, 1000, 50, 10)
        source_power = st.slider("Мощность источника, Вт", 0.1, 100.0, 1.0, 0.1)
        source_tilt_deg = st.slider("Наклон источника, °", 0.0, 89.0, 0.0, 1.0)
        source_file = st.file_uploader("CSV источника (колонка `value`)", type=["csv"])

        st.markdown("**Объект (плоскость)**")
        obj_width = st.slider("Ширина, px", 4, 128, 32, 4)
        obj_height = st.slider("Высота, px", 4, 128, 32, 4)
        point_size = st.slider("Размер точки (pixel_pitch)", 1.0, 50.0, 10.0, 1.0)

        st.markdown("**Отражение (reflectance)**")
        reflectance_file = st.file_uploader("CSV со спектром (колонка `value`)", type=["csv"])
        if reflectance_file is None:
            default_csv = PROJECT_ROOT / "workspace" / "input" / "scarlet_spectrum.csv"
            reflectance_values = read_spectrum_from_csv(str(default_csv))
        else:
            df = pd.read_csv(io.BytesIO(reflectance_file.getvalue()))
            if "value" not in df.columns:
                st.error("В CSV отсутствует колонка `value`!")
                st.stop()
            reflectance_values = df["value"].astype(float).tolist()

        max_val = max(reflectance_values)
        reflectance = [v / max_val for v in reflectance_values]

        if source_file is None:
            radiation = [1.0] * len(reflectance)
        else:
            source_df = pd.read_csv(io.BytesIO(source_file.getvalue()))
            if "value" not in source_df.columns:
                st.error("В CSV источника отсутствует колонка `value`!")
                st.stop()
            radiation = source_df["value"].astype(float).tolist()
            if len(radiation) != len(reflectance):
                st.error("Длины спектров источника и отражения должны совпадать.")
                st.stop()

    with st.expander(" Оптический тракт (RGB)", expanded=False):
        transmission_r = st.slider("Пропускание R", 0.0, 1.0, 0.95, 0.01)
        transmission_g = st.slider("Пропускание G", 0.0, 1.0, 0.90, 0.01)
        transmission_b = st.slider("Пропускание B", 0.0, 1.0, 0.85, 0.01)

        st.markdown("**Камера-обскура (диафрагма)**")
        aperture_diameter = st.slider("Диаметр диафрагмы", 1.0, 200.0, 50.0, 1.0)
        object_distance = st.slider("Расстояние объект → диафрагма", 1.0, 500.0, 50.0, 1.0)
        image_distance = st.slider("Расстояние диафрагма → изображение", 1.0, 500.0, 50.0, 1.0)
        aperture_offset_x = st.slider("Смещение диафрагмы по X", -100.0, 100.0, 0.0, 1.0)
        aperture_offset_y = st.slider("Смещение диафрагмы по Y", -100.0, 100.0, 0.0, 1.0)
        tilt_x_deg = st.slider("Наклон оптической оси по X, °", -60.0, 60.0, 0.0, 1.0)
        tilt_y_deg = st.slider("Наклон оптической оси по Y, °", -60.0, 60.0, 0.0, 1.0)

    with st.expander(" Сенсор", expanded=False):
        # Кривые чувствительности нормированы балансом белого (сумма каждой ≈ 1),
        # поэтому усиление по умолчанию поднято до 24000 — иначе RAW-сигнал слишком
        # слаб (max ~33/1023), синий канал схлопывается и цвета искажаются.
        gain = st.slider("Gain", 100.0, 100000.0, 24000.0, 100.0)
        dark_offset = st.slider("Dark offset", 0.0, 1.0, 0.002, 0.001)

    with st.expander(" АЦП", expanded=False):
        bit_depth = st.slider("Разрядность, бит", 8, 16, 10, 1)
        full_scale = st.slider("Full scale", 1.0, 1000.0, 8.0, 1.0)

    st.markdown("**Отображение**")
    auto_normalize = st.checkbox(
        "Автонормализация яркости (min-max stretch)",
        value=True,
        help="Если включено — яркость растягивается на весь диапазон 0..255. "
        "Если выключено — показываются физические значения АЦП (будет темно при слабом сигнале).",
    )

    # Чекбокс для показа 3D-сцены
    show_3d = st.checkbox(" Показать 3D-сцену", key="show_3d_scene")

    # Параметры плоскости для интерактивной 3D-визуализации
    plane_tilt_deg = st.slider("Наклон плоскости, °", -45.0, 45.0, 0.0, 1.0)
    plane_rot_deg = st.slider("Поворот плоскости вокруг Z, °", 0.0, 360.0, 0.0, 1.0)

    run = st.button("▶ Запустить симуляцию", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# 3D-визуализация геометрии
# ──────────────────────────────────────────────────────────────
if show_3d:
    st.subheader("3D-визуализация геометрии")

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

    # Плоскость объекта (полупрозрачная сетка) с учётом наклона и поворота
    half_w = plane_w / 2.0
    half_h = plane_h / 2.0

    # локальные координаты углов плоскости относительно центра
    local_corners = np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ]
    )

    # преобразования: наклон вокруг локальной оси X и поворот вокруг глобальной Z
    th = np.radians(plane_tilt_deg)
    rz = np.radians(plane_rot_deg)

    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    # применяем сначала наклон, затем поворот
    transformed = (Rz @ (Rx @ local_corners.T)).T

    # переносим в мировые координаты (центр плоскости в (cx, cy, plane_z))
    corners_world = transformed + np.array([cx, cy, plane_z])
    corners_x = corners_world[:, 0].tolist()
    corners_y = corners_world[:, 1].tolist()
    corners_z = corners_world[:, 2].tolist()

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

    # Нормаль плоскости (из двух соседних ребер)
    v1 = corners_world[1] - corners_world[0]
    v2 = corners_world[2] - corners_world[1]
    normal = np.cross(v1, v2)
    n_norm = np.linalg.norm(normal)
    if n_norm > 0:
        normal_unit = normal / n_norm
    else:
        normal_unit = np.array([0.0, 0.0, 1.0])

    # Центр плоскости в мировых координатах
    center_world = np.array([cx, cy, plane_z])

    # Вектор от источника к центру плоскости
    src_vec = center_world - np.array([src_x, src_y, src_z])
    src_dist = np.linalg.norm(src_vec)
    if src_dist > 0:
        src_unit = src_vec / src_dist
    else:
        src_unit = np.array([0.0, 0.0, 1.0])

    # Угол падения между нормалью и направлением от центра сцены
    angle_to_center = np.degrees(
        np.arccos(np.clip(np.abs(np.dot(normal_unit, src_unit)), -1.0, 1.0))
    )

    # Вычисление ориентации источника по заданному наклону (tilt) — направлено в сторону центра по азимуту
    th_src = np.radians(source_tilt_deg)
    hx = cx - src_x
    hy = cy - src_y
    hnorm = np.hypot(hx, hy)
    if hnorm > 1e-6:
        ux = hx / hnorm
        uy = hy / hnorm
    else:
        ux, uy = 1.0, 0.0

    # Направление, смотрящее вниз под углом th_src к вертикали, в сторону центра
    src_dir = np.array([ux * np.sin(th_src), uy * np.sin(th_src), -np.cos(th_src)])
    src_dir_unit = src_dir / np.linalg.norm(src_dir)

    # Угол между нормалью плоскости и направлением источника
    angle_source = np.degrees(
        np.arccos(np.clip(np.abs(np.dot(normal_unit, src_dir_unit)), -1.0, 1.0))
    )

    # Отрисовка нормали (стрелка) и вектора от источника
    arrow_scale = max(plane_w, plane_h, 100) * 0.25
    n_end = center_world + normal_unit * arrow_scale
    fig.add_trace(
        go.Scatter3d(
            x=[center_world[0], n_end[0]],
            y=[center_world[1], n_end[1]],
            z=[center_world[2], n_end[2]],
            mode="lines+markers",
            line=dict(color="#0000FF", width=4),
            marker=dict(size=2, color="#0000FF"),
            name=" Нормаль",
            hoverinfo="skip",
        )
    )

    # Вектор от источника к центру
    fig.add_trace(
        go.Scatter3d(
            x=[src_x, center_world[0]],
            y=[src_y, center_world[1]],
            z=[src_z, center_world[2]],
            mode="lines",
            line=dict(color="#00AA00", width=3, dash="dashdot"),
            name=" Вектор источника",
            hoverinfo="skip",
        )
    )

    # Визуализация ориентации источника (по заданному наклону)
    s_end = np.array([src_x, src_y, src_z]) + src_dir_unit * arrow_scale
    fig.add_trace(
        go.Scatter3d(
            x=[src_x, s_end[0]],
            y=[src_y, s_end[1]],
            z=[src_z, s_end[2]],
            mode="lines+markers",
            line=dict(color="#FF00FF", width=4),
            marker=dict(size=2, color="#FF00FF"),
            name=" Ориентация источника",
            hoverinfo="skip",
        )
    )

    size = max(abs(src_x), abs(src_y), abs(src_z), plane_w, plane_h, 100)
    fig.update_layout(
        title=dict(
            text=(
                f" Расстояние до центра: {distance:.0f} | "
                f"Угол (центр→норма): {angle_to_center:.1f}° | "
                f"Наклон источника: {source_tilt_deg:.1f}° | "
                f"Угол падения: {angle_source:.1f}°"
            ),
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


def _colorize_channel(arr_2d: list[list[float]], channel_index: int) -> np.ndarray:
    """
    Нормализует одноканальную карту и раскрашивает её в собственный цвет
    (0=R, 1=G, 2=B), чтобы канал отображался в своём цвете, а не в ЧБ.
    """
    gray = _normalize_2d(arr_2d)
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    rgb[:, :, channel_index] = gray
    return rgb


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
        radiation=radiation,
        source_xyz=[float(src_x), float(src_y), float(src_z)],
        reflectance=reflectance,
        object_width=obj_width,
        object_height=obj_height,
        point_size=point_size,
        power=source_power,
        tilt_deg=source_tilt_deg,
    )
    scene_source = build_scene_source(scene_input)

    # 3. Оптика
    optics_config = OpticsConfig(
        channel_count=3,
        split_strategy="pinhole projection (camera obscura)",
        mask_pattern="single circular aperture",
        transmission=[transmission_r, transmission_g, transmission_b],
        recombination_mode="multi-channel",
        rgb_ranges_nm=[(380, 480), (480, 600), (600, 730)],
        aperture_diameter=aperture_diameter,
        object_distance=object_distance,
        image_distance=image_distance,
        aperture_offset_x=aperture_offset_x,
        aperture_offset_y=aperture_offset_y,
        tilt_x_deg=tilt_x_deg,
        tilt_y_deg=tilt_y_deg,
        description="Интерактивная камера-обскура",
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
            image = [[[max(0, min(255, int(v * 255 / max_code))) for v in pixel] for pixel in row] for row in rgb_int]
        image_np = np.array(image, dtype=np.uint8)

    st.success("Симуляция завершена!")

    # Отображаем кадры без сглаживания: браузер по умолчанию растягивает
    # маленькое изображение 32x32 билинейно (размытие). Принудительно включаем
    # nearest-neighbor рендеринг — пиксели остаются чёткими, как в outputs/.
    st.markdown(
        "<style>[data-testid=\"stImage\"] img { image-rendering: pixelated; "
        "image-rendering: crisp-edges; }</style>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Итоговое RGB-изображение")
        if auto_normalize:
            st.caption("Режим: автонормализация (min-max stretch + gamma)")
        else:
            st.caption("Режим: физические значения АЦП (без растягивания)")
        st.image(image_np, use_container_width=True)

        tabs = st.tabs(["Канал R", "Канал G", "Канал B", "Заряд", "Кадр АЦП"])

        # Разбиение по каналам берём из восстановленного (демозаикой) RGB-кадра
        # и показываем каждый канал в его собственном цвете.
        with tabs[0]:
            st.image(
                _colorize_channel([[p[0] for p in row] for row in rgb_int], 0),
                caption="Канал R (после демозаики)",
                use_container_width=True,
            )
        with tabs[1]:
            st.image(
                _colorize_channel([[p[1] for p in row] for row in rgb_int], 1),
                caption="Канал G (после демозаики)",
                use_container_width=True,
            )
        with tabs[2]:
            st.image(
                _colorize_channel([[p[2] for p in row] for row in rgb_int], 2),
                caption="Канал B (после демозаики)",
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
        st.subheader("Статистика")

        frame_stats = calculate_image_statistics(artifacts.frame.data, artifacts.frame.bit_depth)
        range_check = verify_digital_range(artifacts.frame.data, artifacts.frame.bit_depth)
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
        st.subheader("Спектр отражения (reflectance)")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(artifacts.axis.wave, reflectance, color="seagreen", lw=2)
        ax.fill_between(artifacts.axis.wave, reflectance, alpha=0.2, color="seagreen")
        ax.set_xlabel("Длина волны, нм")
        ax.set_ylabel("Коэффициент отражения [0..1]")
        ax.set_title("Reflectance объекта")
        ax.grid(True, ls="--", alpha=0.4)
        st.pyplot(fig, use_container_width=True)

    with g2:
        st.subheader("Спектр источника")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(artifacts.axis.wave, radiation, color="darkorange", lw=2)
        ax2.fill_between(artifacts.axis.wave, radiation, alpha=0.2, color="orange")
        ax2.set_xlabel("Длина волны, нм")
        ax2.set_ylabel("Излучение, Вт/м²/нм")
        ax2.set_title("Спектр источника")
        ax2.grid(True, ls="--", alpha=0.4)
        st.pyplot(fig2, use_container_width=True)

    g3, g4 = st.columns(2)

    with g3:
        st.subheader("Гистограмма кадра")
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

    with g4:
        st.subheader("Параметры источника")
        st.metric("Мощность", f"{source_power:.1f} Вт")
        st.metric("Наклон", f"{source_tilt_deg:.1f}°")
        st.metric("Размер спектра", f"{len(radiation)} точек")

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
            f"Экспозиция: {len(artifacts.exposure.channel_irradiance or [])}x"
            f"{len((artifacts.exposure.channel_irradiance or [[]])[0])}x3",
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
