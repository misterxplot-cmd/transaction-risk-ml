# Данные и воспроизводимость

Исходный файл: `metaverse_transactions_dataset.csv` из [Metaverse Financial Transactions Dataset](https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset).

Положите скачанный CSV в локальную папку `data/`. Данные, ZIP, модели и локальные настройки исключены из Git. В репозитории опубликованы только агрегированные результаты.

Для воспроизведения используйте файл из того же источника. SHA-256 фактически использованного CSV записывается в `metrics.json`; он помогает отличить версии набора без публикации самого файла.

Числовые поля: `hour_of_day`, `amount`, `ip_prefix`, `login_frequency`, `session_duration`, `risk_score`.

Категориальные поля: `transaction_type`, `location_region`, `purchase_pattern`, `age_group`.

Цель: `anomaly`. Не используемые поля: `timestamp`, `sending_address`, `receiving_address`.

`amount` описан источником как сумма в моделируемой валюте, `session_duration` — длительность в минутах. Категории и диапазоны источника не следует переносить на реальные системы без отдельной проверки.

В использованном наборе диапазоны `risk_score` для трёх классов не пересекаются. В `transaction_type` встречаются уже интерпретированные типы операций, включая scam/phishing. Поэтому показаны отдельные эксперименты с исключением этих полей.
