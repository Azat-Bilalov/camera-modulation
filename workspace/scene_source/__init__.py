"""Рабочее место роли: модель сцены и источника.
Гостев М.А."""

# Тестовые данные
testW = [400, 420, 440, 460, 480, 500]
testR = [0.2, 0.35, 0.6, 0.8, 0.7, 0.5]
testC = [0.1, 0.12, 0.18, 0.3, 0.45, 0.5]



# Преобразование входных данных в список
def parse_list(text):
    values = text.replace(";", ",").split(",")
    result = []

    for value in values:
        value = value.strip()
        if value:
            result.append(float(value))
    return result


# Формирование данных для модуля оптики
def build_optic_input(wave, radiation, coef):
    scene_signal = []
    for i in range(len(wave)):
        value = radiation[i] * coef[i]
        scene_signal.append(round(value, 4))
    return {
        "wave": wave,
        "input_signal": scene_signal,
        "bands_count": len(wave),
        "start": wave[0],
        "stop": wave[-1]
    }


if input("Использовать тестовые данные? (y/n): ").lower() == "y":
    wave = testW
    radiation = testR
    coef = testC
else:
    wave = parse_list(input("Длины волн, нм: "))
    radiation = parse_list(input("Мощность излучения: "))
    coef = parse_list(input("Коэффициенты отражения объекта: "))


optic_input = build_optic_input(wave, radiation, coef)


print(f"Начальная длина волны: {optic_input['start']} нм")
print(f"Конечная длина волны: {optic_input['stop']} нм")
print(f"Количество спектральных каналов: {optic_input['bands_count']}")
print(f"Длины волн: {optic_input['wave']}")
print(f"Спектральная карта сцены: {optic_input['input_signal']}")
print(f"Данные для модуля оптики: {optic_input}")