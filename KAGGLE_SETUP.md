# Настройка Kaggle API

Для автоматического скачивания датасетов необходимо настроить Kaggle API.

## Шаг 1: Создание API токена

1. Перейдите на сайт Kaggle: https://www.kaggle.com
2. Войдите в свой аккаунт (или создайте новый)
3. Перейдите в настройки аккаунта: https://www.kaggle.com/account
4. Прокрутите вниз до раздела "API"
5. Нажмите кнопку "Create New API Token"
6. Файл `kaggle.json` будет автоматически скачан

## Шаг 2: Установка токена

### macOS / Linux

```bash
# Создать директорию для конфигурации
mkdir -p ~/.kaggle

# Переместить скачанный файл
mv ~/Downloads/kaggle.json ~/.kaggle/

# Установить правильные права доступа
chmod 600 ~/.kaggle/kaggle.json
```

### Windows

```bash
# Создать директорию
mkdir %USERPROFILE%\.kaggle

# Переместить файл kaggle.json в созданную директорию
# %USERPROFILE%\.kaggle\kaggle.json
```

## Шаг 3: Проверка настройки

```bash
# Установить пакет kaggle
pip install kaggle

# Проверить, что API работает
kaggle datasets list
```

Если команда выполнилась успешно, вы увидите список датасетов.

## Шаг 4: Запуск скрипта загрузки

```bash
python setup_datasets.py
```

Скрипт автоматически:
1. Скачает DRIVE, BCCD, CIFAR-10 из Kaggle
2. Распакует их в `data/raw/`
3. Сформирует датасет для первичной классификации в `data/primary/`

## Альтернативный вариант (ручная загрузка)

Если автоматическая загрузка не работает, скачайте датасеты вручную:

### DRIVE
- URL: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction
- Распакуйте в: `data/raw/drive/`

### BCCD
- URL: https://www.kaggle.com/datasets/surajiiitm/bccd-dataset
- Распакуйте в: `data/raw/bccd/`

### CIFAR-10
- URL: https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders
- Распакуйте в: `data/raw/cifar10/`

После ручной загрузки запустите:
```bash
python setup_datasets.py
```

Скрипт обнаружит уже скачанные датасеты и только подготовит датасет первичной классификации.
