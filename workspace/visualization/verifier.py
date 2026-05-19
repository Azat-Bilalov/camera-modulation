"""
Модуль верификации качества изображения
"""

from typing import List, Tuple, Dict, Any
import math


def verify_digital_range(frame_data: List[List[int]], expected_bit_depth: int = 10) -> Dict[str, Any]:
    """
    Проверяет, что значения пикселей находятся в допустимом диапазоне.

    Args:
        frame_data: цифровой кадр (2D список int)
        expected_bit_depth: ожидаемая битовая глубина

    Returns:
        словарь с результатами проверки
    """
    max_valid = (1 << expected_bit_depth) - 1
    min_valid = 0

    flat_values = [val for row in frame_data for val in row]
    actual_min = min(flat_values)
    actual_max = max(flat_values)

    errors = []
    if actual_min < min_valid:
        errors.append(f"Минимальное значение {actual_min} ниже допустимого {min_valid}")
    if actual_max > max_valid:
        errors.append(f"Максимальное значение {actual_max} выше допустимого {max_valid}")

    return {
        "bit_depth": expected_bit_depth,
        "expected_range": (min_valid, max_valid),
        "actual_range": (actual_min, actual_max),
        "is_valid": len(errors) == 0,
        "errors": errors
    }


def verify_no_clipping(frame_data: List[List[int]], bit_depth: int = 10, threshold_percent: float = 10.0) -> Dict[
    str, Any]:
    """
    Проверяет, нет ли значительного клиппинга.

    Args:
        frame_data: цифровой кадр (2D список int)
        bit_depth: битовая глубина
        threshold_percent: порог клиппинга в процентах

    Returns:
        словарь с результатами проверки
    """
    max_code = (1 << bit_depth) - 1
    min_code = 0

    total_pixels = len(frame_data) * len(frame_data[0])
    flat_values = [val for row in frame_data for val in row]

    clipped_high = sum(1 for v in flat_values if v == max_code)
    clipped_low = sum(1 for v in flat_values if v == min_code)

    high_percent = 100.0 * clipped_high / total_pixels
    low_percent = 100.0 * clipped_low / total_pixels

    is_acceptable = high_percent <= threshold_percent and low_percent <= threshold_percent

    return {
        "total_pixels": total_pixels,
        "clipped_high": clipped_high,
        "clipped_high_percent": round(high_percent, 2),
        "clipped_low": clipped_low,
        "clipped_low_percent": round(low_percent, 2),
        "is_acceptable": is_acceptable,
        "warning": "Значительный клиппинг!" if not is_acceptable else "Клиппинг в допустимых пределах"
    }


def calculate_image_statistics(frame_data: List[List[int]], bit_depth: int = 10) -> Dict[str, Any]:
    """
    Рассчитывает статистику по изображению.

    Args:
        frame_data: цифровой кадр (2D список int)
        bit_depth: битовая глубина

    Returns:
        словарь со статистикой
    """
    flat_values = [val for row in frame_data for val in row]

    return {
        "height": len(frame_data),
        "width": len(frame_data[0]),
        "min": min(flat_values),
        "max": max(flat_values),
        "mean": round(sum(flat_values) / len(flat_values), 2),
        "bit_depth": bit_depth,
        "dynamic_range": f"{min(flat_values)}..{max(flat_values)}"
    }


def verify_against_reference(
        test_frame: List[List[int]],
        reference_frame: List[List[int]]
) -> Dict[str, Any]:
    """
    Сравнивает два кадра и вычисляет метрики ошибок.

    Args:
        test_frame: тестовый кадр
        reference_frame: эталонный кадр

    Returns:
        словарь с метриками
    """
    if len(test_frame) != len(reference_frame) or len(test_frame[0]) != len(reference_frame[0]):
        return {"error": "Размеры кадров не совпадают"}

    flat_test = [val for row in test_frame for val in row]
    flat_ref = [val for row in reference_frame for val in row]

    mse = sum((t - r) ** 2 for t, r in zip(flat_test, flat_ref)) / len(flat_test)
    mae = sum(abs(t - r) for t, r in zip(flat_test, flat_ref)) / len(flat_test)
    max_diff = max(abs(t - r) for t, r in zip(flat_test, flat_ref))

    # PSNR (если MSE > 0)
    if mse > 0:
        psnr = 10 * math.log10((255 ** 2) / mse)
    else:
        psnr = float('inf')

    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "max_diff": max_diff,
        "psnr": round(psnr, 2) if psnr != float('inf') else "бесконечность"
    }