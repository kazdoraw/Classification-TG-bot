# Инструкции по установке и запуску

## Шаг 1: Установка зависимостей

```bash
# Активировать виртуальное окружение (если еще не активировано)
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac

# Установить все зависимости
pip install -r requirements.txt
```

## Шаг 2: Настройка Kaggle API

Подробные инструкции см. в [KAGGLE_SETUP.md](KAGGLE_SETUP.md)

**Краткая версия:**

1. Перейдите на https://www.kaggle.com/account
2. Нажмите "Create New API Token"
3. Сохраните скачанный `kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

4. Проверьте настройку:
```bash
kaggle datasets list
```

## Шаг 3: Загрузка и подготовка датасетов

```bash
# Запустить единый скрипт подготовки
python setup_datasets.py
```

Этот скрипт автоматически:
- Скачает DRIVE, BCCD, CIFAR-10 из Kaggle
- Распакует их в `data/raw/`
- Создаст датасет первичной классификации в `data/primary/`
- Разделит на train/val (80/20)

**Время выполнения:** 10-15 минут (зависит от скорости интернета)

## Шаг 4: Проверка результата

```bash
# Проверить структуру данных
ls -la data/primary/train/
ls -la data/primary/val/

# Должны увидеть папки: retina, blood, scene
# В каждой папке должны быть изображения
```

## Альтернатива: Ручная загрузка

Если автоматическая загрузка не работает:

1. Скачайте датасеты вручную с Kaggle:
   - DRIVE: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction
   - BCCD: https://www.kaggle.com/datasets/surajiiitm/bccd-dataset
   - CIFAR-10: https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders

2. Распакуйте в соответствующие папки:
   - `data/raw/drive/`
   - `data/raw/bccd/`
   - `data/raw/cifar10/`

3. Запустите скрипт:
```bash
python setup_datasets.py
```

Скрипт обнаружит уже загруженные датасеты и только подготовит датасет первичной классификации.

## Что дальше?

После успешной подготовки датасетов переходите к Этапу 3 - обучению первичного классификатора.

Смотрите [NEXT_STEPS.md](NEXT_STEPS.md) для дальнейших действий.
