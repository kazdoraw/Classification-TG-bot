# Computer Vision Telegram Bot - Final Project

Telegram-бот для автоматической классификации и обработки изображений с использованием компьютерного зрения.

## Описание

Бот принимает изображение и автоматически определяет его тип:
- **Сетчатка глаза** → сегментация сосудов
- **Мазок крови** → детекция и подсчет клеток крови
- **Повседневная сцена** → классификация CIFAR-10

## Структура проекта

```
.
├── data/
│   ├── primary/          # Датасет для первичной классификации
│   │   ├── train/        # Обучающая выборка
│   │   └── val/          # Валидационная выборка
│   └── raw/              # Исходные датасеты
│       ├── drive/        # DRIVE (сетчатка)
│       ├── bccd/         # BCCD (кровь)
│       └── cifar10/      # CIFAR-10 (сцены)
├── models/               # Обученные модели
├── src/                  # Исходный код
│   ├── prepare_primary_dataset.py
│   ├── train_primary_*.py
│   ├── train_*.py
│   ├── inference_*.py
│   └── bot.py
├── tmp/                  # Временные файлы
├── memory-bank/          # Документация проекта
└── requirements.txt
```

## Установка

### 1. Клонирование и настройка окружения

```bash
# Создать виртуальное окружение (Python 3.12)
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt
```

### 2. Настройка конфигурации

```bash
# Скопировать шаблон конфигурации
cp .env.example .env

# Отредактировать .env и добавить токен Telegram-бота
# TELEGRAM_BOT_TOKEN=your_actual_token
```

### 3. Настройка Kaggle API

Для автоматической загрузки датасетов см. инструкции в [KAGGLE_SETUP.md](KAGGLE_SETUP.md)

Краткая версия:
```bash
# 1. Получить API токен на https://www.kaggle.com/account
# 2. Сохранить kaggle.json в ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Автоматическая загрузка и подготовка датасетов

```bash
# Один скрипт для загрузки и подготовки всех датасетов
python setup_datasets.py
```

Этот скрипт автоматически:
- Скачает DRIVE, BCCD, CIFAR-10 из Kaggle
- Распакует в `data/raw/`
- Подготовит датасет первичной классификации в `data/primary/`

**Альтернатива (ручная загрузка):** см. [KAGGLE_SETUP.md](KAGGLE_SETUP.md)

## Обучение моделей

### Первичный классификатор (3 класса)

```bash
# Baseline CNN
python src/train_primary_baseline_cnn.py

# ResNet-18 (рекомендуется для продакшна)
python src/train_primary_resnet18.py

# Vision Transformer
python src/train_primary_vit.py
```

### Вспомогательные модели

```bash
# Сегментация сосудов сетчатки (U-Net)
python src/train_retina_segmentation.py

# Детекция клеток крови (YOLO)
python src/train_blood_detector.py

# Классификатор CIFAR-10
python src/train_cifar10_classifier.py
```

## Запуск бота

```bash
python src/bot.py
```

## Использование бота

1. Найти бота в Telegram по имени
2. Отправить команду `/start`
3. Отправить изображение
4. Получить результат обработки

## Технологический стек

- **Python**: 3.12
- **PyTorch**: 2.1.2
- **OpenCV**: 4.9.0
- **aiogram**: 3.3.0 (Telegram bot)
- **ultralytics**: 8.1.9 (YOLO)

## Датасеты

- **DRIVE**: Digital Retinal Images for Vessel Extraction
- **BCCD**: Blood Cell Count and Detection Dataset
- **CIFAR-10**: PNGs in folders

## Модели

1. **Первичный классификатор**: ResNet-18 (transfer learning)
2. **Сегментация сетчатки**: U-Net
3. **Детекция клеток**: YOLOv8
4. **Классификатор CIFAR-10**: CNN/ResNet

## Лицензия

Образовательный проект.
