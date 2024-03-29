# Описание датасета
Датасет состоит из 100 выравненных пар optical-SAR изображений размера 1024х1024 пикселя. Данные были скачены с сайта Copernicus Open Access Hub [1].
Используемые в датасете изображения являются публично доступными, а лицензия к ним позволяет их распространять и изменять [2].

В качестве SAR изображений использовались изображения со спутника Sentinel-1A со следующими параметрами: 
тип продукта Level-1 Ground Range Detected (GRD), режим сенсора Interferometric Wave (IW) swath mode, поляризация VH. 
В качестве оптических изображений использовались трехканальные RGB спутниковые снимки со спутника Sentinel-2A (тип продукта: S2MSI1C). 
Изображения обоих типов были приведены к пространственному разрешению 10 метров/пиксель. 
Каждая пара optical-SAR изображений выравнена друг относительно друга с использованием геопривязки. Для увеличения точности выравнивания было проведено ручное сопоставление изображений с использованием проективного преобразования. Полученные изображения выравнены преимущественно с субпиксельной точностью, но иногда встречаются небольшие области, где точность ниже (погрешность до двух пикселей).

Датасет сопровождается файлом с метаданными (`metadata.json`). Файл содержит список словарей в следующем формате:

{

"fragment_filename": <имя изображения-фрагмента>,

"opt_tile_name": <имя оптического тайла, из которого вырезался фрагмент>,

"opt_tile_sensing_date": <дата, когда оптический тайл был получен спутником>,

"sar_tile_name": <имя радиолокационного тайла, из которого вырезался фрагмент>,

"sar_tile_sensing_date": <дата, когда радиолокационный тайл был получен спутником>,

"corners_EPSG:3395": <список координат углов изображения-фрагмента в системе проекций EPSG:3395>

}

# Инструкция по скачиванию данных с Copernicus Open Access Hub
Описанный выше датасет можно напрямую скачать из данного репозитория.
В этом разделе представляется инструкция по скачиванию данных с Copernicus Open Access Hub (изображения из вышеупомянутого датасета получены отсюда).

Один из способов скачивания – с помощью скрипта на языке Python. Код и инструкцию к нему можно найти на сайте [3]. 
Данный способ позволяет скачивать сразу несколько сцен (тайлов) одновременно, согласно указанным критериям.

Другой способ – интерактивное скачивание с сайта-источника [4]. 
Здесь нужно выделить интересующую область на карте и указать нужные параметры в фильтре (дата снимка, название спутника, тип продукта, поляризация (Sentinel-1), 
режим сенсора (Sentinel-1), доля облачного покрытия (Sentinel-2)). 
Скачивание в этом случае происходит по одной сцене одновременно, но пользователь может наглядно увидеть положение снимка на карте.

# Инструкция по организации данных
В этом разделе представлено краткое описание алгоритма получения изображений из нашего датасета из скаченных тайлов.

Скаченные оптические и радиолокационные изображения переводились в единую картографическую проекцию EPSG:3395. 
На каждом канале оптических и радиолокационных изображений большинство пикселей имеют низкие значения интенсивности. 
Поэтому каждый канал предварительно делился на квантиль 99% и полученные значения больше 1 уменьшались до 1 (т.е. 1% наиболее ярких пикселей «отбрасывался»). 
Для увеличения точности сопоставления вырезаемых фрагментов дополнительно использовалось ручное сопоставление с помощью проективного преобразования радиолокационного изображения. 
Матрица проективного преобразования вычислялась с помощью алгоритма RANSAC по набору пар соответствующих точек, составленных вручную. 
Далее из оптических и радиолокационных изображений вырезались соответствующие фрагменты размера 1024х1024 пикселя, которые нормировались от 0 до 1. 
Оптические изображения представляются в виде RGB изображения, сформированного из красного, зеленого и синего каналов.

# Литература:
1.	Copernicus Open Access Hub. Available at: https://scihub.copernicus.eu/ (accessed December 29, 2020).
2.	Copernicus Open Access Hub, Terms and Conditions. Available at: https://scihub.copernicus.eu/twiki/do/view/SciHubWebPortal/TermsConditions (accessed January 22, 2021).
3.  Automated download of Sentinel data. Available at: https://github.com/olivierhagolle/Sentinel-download (accessed December 29, 2020).
4.  Download data from Copernicus Open Access Hub. Available at: https://scihub.copernicus.eu/dhus/#/home (accessed December 29, 2020).

