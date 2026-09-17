# AniKot 2.3.0 — Hybrid Search + High Load

VK-бот для поиска аниме и дунхуа по кадру, фото и названию. Версия рассчитана на BotHost.ru и запускается через `main.py`.

## Поиск

Текущая архитектура:

```text
AniKot
→ Qwen3.5 Flash

AniKot Pro
→ AnimeTrace
→ DeepSeek V4 Flash проверяет и нормализует кандидатов

AniKot Pro+
→ AnimeTrace
+
Kimi K2.6 независимо анализирует исходный кадр
→ Kimi сверяет оба результата
```

- **AniKot** — `qwen3.5-flash`
- **AniKot Pro** — AnimeTrace + `deepseek-v4-flash`
- **AniKot Pro+** — AnimeTrace + `kimi-k2.6`
- Минимальные пороги по умолчанию: 55% / 65% / 75%
- Администраторы из `ADMIN_IDS` имеют безлимит поисков
- Vision-кэш хранит результаты 30 дней
- Одновременно допускается ограниченное число ожидающих поисков (`MAX_PENDING_SEARCHES`)

На 17.09.2026 каталог AIAI.BY содержит **DeepSeek V4 Flash**, но отдельного `DeepSeek V4.1 Flash` в каталоге нет. Поэтому рабочий default Pro — `deepseek-v4-flash`. `AIAI_PRO_MODEL` остаётся настраиваемым через окружение: когда провайдер добавит нужную V4.1-модель, её можно будет включить без изменения архитектуры.

## Hybrid Accuracy v4

### Обычный AniKot

Обычный поиск использует мультимодальную `Qwen3.5 Flash` через AIAI.BY. Модель получает исходное изображение и сама определяет тайтл, персонажа и метаданные.

Старое значение:

```env
AIAI_ANIKOT_MODEL=gpt-5.4-nano
```

автоматически считается прежним default и мигрирует на `qwen3.5-flash`.

### AniKot Pro

Для изображения нормальный путь такой:

1. AnimeTrace получает исходный кадр и возвращает кандидатов произведений/персонажей.
2. DeepSeek получает только структурированные кандидаты AnimeTrace, проверяет их логическую согласованность, алиасы, переводы и выбирает итог.
3. Если AnimeTrace помечает все боксы как `not_confident`, итоговая уверенность ограничивается консервативно.
4. Если AnimeTrace или DeepSeek временно недоступен, бот не списывает запрос при полном провале и имеет fallback-путь, чтобы не падать целиком.

Для текстового поиска AnimeTrace не используется, потому что он работает с изображениями; текстовый Pro идёт непосредственно в DeepSeek.

### AniKot Pro+

AnimeTrace и Kimi K2.6 запускаются независимо по одному исходному кадру. После этого Kimi получает свой независимый vision-результат и кандидатов AnimeTrace и выполняет финальную сверку.

Если источники совпадают, это усиливает результат. Если они конфликтуют, финальный арбитр снижает уверенность и выбирает наиболее обоснованный вариант. Если AnimeTrace временно недоступен, Pro+ продолжает работать через независимый Kimi vision-поиск.

### AnimeTrace

AniKot не фиксирует в коде конкретную модель AnimeTrace. При запуске клиент запрашивает:

```text
GET https://api.animetrace.com/v1/model/list
```

и выбирает текущую доступную/default модель. Поиск выполняется через:

```text
POST https://api.animetrace.com/v1/search
```

с `is_multi=1` и `ai_detect=1`.

Для публичного AnimeTrace включена защита от перегрузки:

```env
ANIMETRACE_ENABLED=true
ANIMETRACE_BASE_URL=https://api.animetrace.com
ANIMETRACE_MODEL=
ANIMETRACE_TIMEOUT=15
ANIMETRACE_MAX_CONCURRENCY=2
ANIMETRACE_MIN_INTERVAL_MS=500
```

Клиент обрабатывает 429/503 и известные коды AnimeTrace, включая `17702`, `17704`, `17706`, `17728` и `17731`. На лимитах/перегрузке открывается временный circuit breaker, чтобы бот не продолжал агрессивно обращаться к сервису.

### Direct vision verifier

Для прямых мультимодальных проходов Qwen/Kimi остаётся Accuracy verifier:

```env
AIAI_VISION_VERIFY=true
AIAI_VISION_VERIFY_BELOW=0.88
AIAI_VISION_VERIFY_MARGIN=0.12
AIAI_VISION_VERIFY_UNANCHORED=true
AIAI_VISION_VERIFIER_REASONING=medium
```

Hybrid-сверка Pro/Pro+ выполняется отдельно в `AnimeDetector`.

## High Load

Перед основными обработчиками работает middleware, который:

- ограничивает число одновременно выполняющихся обработчиков;
- ограничивает частоту сообщений от одного пользователя;
- во время глобальной перегрузки отбрасывает лишние события до обращения к SQLite и внешним API;
- ограничивает количество предупреждений о перегрузке;
- не применяет пользовательский rate-limit к администраторам.

```env
MAX_INFLIGHT_MESSAGES=120
MESSAGE_RATE_BURST=12
MESSAGE_RATE_WINDOW_SECONDS=10
RATE_LIMIT_WARNING_INTERVAL=8
OVERLOAD_WARNINGS_PER_SECOND=5
```

### SQLite pool

Используется небольшой пул постоянных SQLite-соединений с WAL:

```env
DB_POOL_SIZE=4
```

Все существующие методы `Database` продолжают работать через `async with db.connection()`.

### Раздельные AI-лимиты

Распознавание аниме и AI-поддержка используют разные semaphore:

```env
AIAI_MAX_CONCURRENCY=3
SUPPORT_AI_MAX_CONCURRENCY=1
```

AnimeTrace имеет собственный лимит:

```env
ANIMETRACE_MAX_CONCURRENCY=2
ANIMETRACE_MIN_INTERVAL_MS=500
```

## Метрики

`/health` показывает:

- RAM процесса;
- число активных поисков;
- текущую и пиковую нагрузку сообщений;
- число отклонений rate-limit / overload;
- состояние SQLite pool;
- выбранные модели поиска;
- состояние AnimeTrace, число успешных/неуспешных вызовов, rate-limit и оставшееся время circuit breaker.

Ожидаемая версия:

```text
2.3.0-bothost
```

## Поддержка

Команда `support`, `/support`, `поддержка` или deep-link с `ref=support` открывает поддержку.

Технические проблемы и вопросы оплаты идут живому администратору. Остальные темы сначала обрабатывает AI первой линии (`qwen3-32b` по умолчанию), который может передать диалог оператору.

```env
SUPPORT_ADMIN_IDS=
SUPPORT_AI_MODEL=qwen3-32b
SUPPORT_AI_MAX_TOKENS=700
SUPPORT_AI_TIMEOUT=90
SUPPORT_AI_HISTORY_LIMIT=6
SUPPORT_AI_HOURLY_LIMIT=20
SUPPORT_AI_MAX_CONCURRENCY=1
```

## Интерфейс поиска

Во время поиска показывается временное сообщение:

```text
🔍 AniKot анализирует кадр...
```

Кнопка отмены называется `Отмена`. После завершения поиска временное сообщение удаляется.

Формат результата:

```text
🎬 Название: Внук мудреца
👤 Персонаж: Сицилия фон Клод
🌍 Страна: Япония
📅 Год: 2019
📺 Количество эпизодов: 12
🎯 Уверенность: 95%
```

## Регистрация и бонусы

- регистрация: `+1` обычный AniKot;
- подписка на сообщество: `+2` обычных AniKot один раз;
- приглашённый друг: `+3` AniKot Pro после 3-дневной проверки;
- 3 страйка за отписку после получения бонуса блокируют аккаунт;
- явно не-аниме изображение не списывает поисковый запрос, но может дать предупреждение.

## Магазин

### AniKot
- 50 запросов — 59 ₽
- 150 запросов — 129 ₽

### Pro
- 10 — 79 ₽
- 30 — 199 ₽
- 100 — 599 ₽

### Pro+
- 10 — 119 ₽
- 30 — 299 ₽
- 100 — 849 ₽

Оплата идёт через LAVA webhook:

```text
https://anikot.bothost.tech/lava/webhook
```

## BotHost

- главный файл: `main.py`
- порт панели: `3000`
- `WEB_HOST=0.0.0.0`
- `PORT` вручную не создавать

Health:

```text
https://anikot.bothost.tech/health
```
