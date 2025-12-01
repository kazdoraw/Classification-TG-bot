# Анализ датасетов

## Исходные датасеты

### 1. DRIVE (Digital Retinal Images for Vessel Extraction)

**Расположение:** `data/DRIVE/`

**Структура:**
```
DRIVE/
├── training/
│   ├── images/         # 20 изображений сетчатки (.tif)
│   ├── 1st_manual/     # Маски сосудов (для обучения сегментации)
│   └── mask/           # Маски области сетчатки
└── test/
    ├── images/         # 20 изображений сетчатки (.tif)
    └── mask/           # Маски
```

**Характеристики:**
- Всего изображений: 40
- Формат: TIFF (.tif)
- Разрешение: 565x584 пикселей
- Назначение:
  - Первичная классификация: класс `retina`
  - Сегментация сосудов: обучение U-Net

### 2. BCCD (Blood Cell Count and Detection)

**Расположение:** `data/BCCD_Dataset-master/`

**Структура:**
```
BCCD_Dataset-master/
└── BCCD/
    ├── JPEGImages/      # 364 изображения мазков крови (.jpg)
    ├── Annotations/     # XML-аннотации в формате VOC
    └── ImageSets/       # Списки train/test/val
```

**Характеристики:**
- Всего изображений: 364
- Формат: JPEG (.jpg)
- Разрешение: 640x480 пикселей
- Классы клеток: RBC (эритроциты), WBC (лейкоциты), Platelets (тромбоциты)
- Назначение:
  - Первичная классификация: класс `blood`
  - Детекция клеток: обучение YOLO

### 3. CIFAR-10 PNGs in folders

**Расположение:** `data/cifar10/cifar10/`

**Структура:**
```
cifar10/cifar10/
├── train/
│   ├── airplane/      # 5000 изображений
│   ├── automobile/    # 5000 изображений
│   ├── bird/          # 5000 изображений
│   ├── cat/           # 5000 изображений
│   ├── deer/          # 5000 изображений
│   ├── dog/           # 5000 изображений
│   ├── frog/          # 5000 изображений
│   ├── horse/         # 5000 изображений
│   ├── ship/          # 5000 изображений
│   └── truck/         # 5000 изображений
└── test/
    └── [те же классы]
```

**Характеристики:**
- Всего изображений: 50,000 (train) + 10,000 (test)
- Формат: PNG (.png)
- Разрешение: 32x32 пикселей
- Классы: 10 повседневных объектов
- Назначение:
  - Первичная классификация: класс `scene` (используются 5 классов: airplane, automobile, bird, ship, truck)
  - Вторичная классификация: классификатор CIFAR-10

---

## Организованная структура данных

### data/raw/ - Нормализованные исходные данные

```
data/raw/
├── drive/
│   └── images/              # 40 изображений сетчатки (.tif)
├── bccd/
│   └── images/              # 364 изображения крови (.jpg)
└── cifar10/
    ├── airplane/            # 5000 изображений
    ├── automobile/          # 5000 изображений
    ├── bird/                # 5000 изображений
    ├── cat/                 # 5000 изображений
    ├── deer/                # 5000 изображений
    ├── dog/                 # 5000 изображений
    ├── frog/                # 5000 изображений
    ├── horse/               # 5000 изображений
    ├── ship/                # 5000 изображений
    └── truck/               # 5000 изображений
```

### data/primary/ - Датасет первичной классификации

```
data/primary/
├── train/
│   ├── retina/              # 32 изображения (80% от 40)
│   ├── blood/               # 160 изображений (80% от 200)
│   └── scene/               # 160 изображений (80% от 200)
└── val/
    ├── retina/              # 8 изображений (20% от 40)
    ├── blood/               # 40 изображений (20% от 200)
    └── scene/               # 40 изображений (20% от 200)
```

**Статистика:**
- **Train:** 352 изображения (32 retina + 160 blood + 160 scene)
- **Val:** 88 изображений (8 retina + 40 blood + 40 scene)
- **Всего:** 440 изображений

**Примечания:**
- Класс `retina` несбалансирован (только 40 изображений vs 200 в других классах)
- Для компенсации дисбаланса при обучении можно использовать:
  - Weighted loss (class weights)
  - Аугментации для класса retina
  - Стратифицированную выборку

---

## Дальнейшее использование

### 1. Первичный классификатор (3 класса)

**Датасет:** `data/primary/`

**Классы:**
- `retina` (32/8) - изображения сетчатки глаза
- `blood` (160/40) - мазки крови
- `scene` (160/40) - повседневные сцены

**Обучение:**
```bash
python src/train_primary_baseline_cnn.py
python src/train_primary_resnet18.py      # Основная модель
python src/train_primary_vit.py
```

### 2. Сегментация сетчатки (U-Net)

**Датасет:** `data/DRIVE/training/`

**Данные:**
- Изображения: `training/images/*.tif`
- Маски: `training/1st_manual/*.gif`

**Обучение:**
```bash
python src/train_retina_segmentation.py
```

### 3. Детекция клеток крови (YOLO)

**Датасет:** `data/BCCD_Dataset-master/BCCD/`

**Данные:**
- Изображения: `JPEGImages/*.jpg`
- Аннотации: `Annotations/*.xml` (формат VOC)

**Обучение:**
```bash
python src/train_blood_detector.py
```

### 4. Классификация CIFAR-10

**Датасет:** `data/cifar10/cifar10/train/`

**Классы:** 10 классов повседневных объектов

**Обучение:**
```bash
python src/train_cifar10_classifier.py
```

---

## Команды для проверки

```bash
# Проверить структуру primary dataset
ls -la data/primary/train/
ls -la data/primary/val/

# Посчитать изображения в каждом классе
find data/primary/train/retina -type f | wc -l
find data/primary/train/blood -type f | wc -l
find data/primary/train/scene -type f | wc -l

# Проверить raw datasets
ls -la data/raw/drive/images/
ls -la data/raw/bccd/images/
ls -la data/raw/cifar10/
```

---

## Статус: ✅ Датасеты готовы

Все датасеты организованы и готовы для обучения моделей.

**Следующий шаг:** Этап 3 - Обучение первичного классификатора
