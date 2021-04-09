# Описание
Здесь представлены примеры файлов с сопоставлениями устойчивых точек. Представлено 4 набора сопоставлений точек. 
Для нахождения сопоставлений между устойчивыми точками SIFT использовался метод вычисления дескрипторов (SIFT, SURF) и метод для их сопоставления (метод ближайшего соседа).
Метод ближайшего соседа сопоставляет пару точек на оптическом и радиолокационном изображении, если расстояние между их дескрипторами является наименьшим 
(т.е. не существует точки-кандидата на другом изображении, при сопоставлении с которым дескрипторное расстояние станет меньше) и меньше порога `t_des`. 
Вычисление дескрипторов (SIFT, SURF) производилось с помощью библиотеки OpenCV.

Для нахождения сопоставлений использовалось два способа: без использования информации о предварительном выравнивании изображений и с использованием такой информации. 
Во втором случае при формировании сопоставлений поиск точек происходил только в окне 21x21 пиксель вокруг ожидаемого положения. 
Сопоставления, найденные по второму случаю, находятся в папках со словом `aligned`.

Формат названия директории с найденными сопоставлениями следующий: `<дескриптор>-<детектор>_<size>_<t_des>`. `size` -- диаметр устойчивой точки (параметр дескриптора).

# Матрицы проективного преобразования
Матрицы проективного преобразования находятся в директории `homography` внутри дирректорий с сопоставлениями точек. 
Матрицы представлены в виде `.txt` файлов, содержащих матрицу 3х3. Для вычисления проективного преобразования использовался алгоритм RANSAC. 
В качестве сопоставлений, использовались сопоставления из соответствующей директории. Порог для RANSAC брался равным 5 пикселям.