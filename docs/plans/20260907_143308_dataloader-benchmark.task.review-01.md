# Ревью: сравнение ридеров в готовом PyTorch DataLoader

| Field | Value |
|---|---|
| Artifact | `review` |
| Reviewed artifact | `/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.task.md` |
| Reviewed artifact type | `task` |
| Reviewed artifact ID | `20260907_143308_dataloader-benchmark` |
| Reviewed revision | `2` |
| Reviewed artifact SHA-256 | `e802daaad00a1f3105aff5f4f660f716a586fd5061e62a93123ed860d39a7517` |
| Bound artifact SHA-256 | `c4c316e2538ef2759418bf04fb238656dd02c3d51f2ffec7f80f61d7bc1dfd33` |
| Review round | `01` |
| Reviewer | `independent-agent` |
| Review context | `fresh` |
| Reviewer profile | `work_artifact_reviewer` |
| Model / reasoning | `gpt-6-astra / xhigh` |
| Agent/thread ID | `/root/dataloader_task_review_01` |
| Input fingerprint | `d9bac04351a0649b8b413bfee8fe0ae57c0126b802ba471c99405db57fd2c1b1` |
| Input fingerprint definition | SHA-256 точных UTF-8 байтов манифеста в приложении, включая завершающий LF |
| Created | `2026-09-07T15:22:44+04:00` |
| Verdict | `approved` |

## Summary

Ревизия 2 задаёт согласованный и проверяемый контракт самостоятельного DataLoader-бенчмарка. Определены пользователи, измеряемый результат, границы времени, семантика изображений, воспроизводимость выборки и совместимость существующего CLI. Мониторинг CPU/RSS ограничен наблюдаемыми величинами; неполнота, накладные расходы и изоляция конфигураций описаны явно.

Альтернативы рассмотрены с содержательными различиями и основаниями выбора. Решения, оставленные планировщику, отделены от изменений, требующих возврата к постановке. Blocker/major и иных замечаний не выявлено.

Манифест и точный SHA-256 задачи соответствуют заданию. Хэши всех 19 перечисленных файлов совпали. Рубрика, AGENTS.md и задача прочитаны полностью; исходники и тесты изучены в пределах необходимых проверок. Применены общая рубрика и рубрика задачи.

## Findings

No findings.

## Traceability checks

| Требование | Критерии приёмки | Результат проверки |
|---|---|---|
| **R1 — данные** | AC1, AC6 | Зафиксированы правила выбора N, неизменность списка, ошибка пустого ввода и восстановимость выбранных путей. Фактический порядок проверяется отдельно от seed. |
| **R2 — датасет и ридеры** | AC2, AC9 | Контракт `(image, 0)`, первоначальные backends и отдельные OpenCV-пути заданы явно. Проверяются ID, доступность и непосредственная передача совместимых ndarray/Tensor. |
| **R3 — общий контракт** | AC2 | Определены RGB, layout, CPU, dtype, диапазон и форма батча. Проверки охватывают оба представления, отсутствие масштабирования, безопасность буферов и эквивалентность общей обработки. |
| **R4 — трансформ** | AC3 | Проверяются resize короткой стороны, центральная область, округление и фиксированные параметры. Режим без геометрии сохраняет Tensor-контракт и явно обрабатывает несовместимые размеры. |
| **R5 — настройки** | AC4, AC8, AC9 | Перечислены обязательные настройки CLI/API. Проверяются batching, недопустимые сочетания и повторные запуски с изменёнными параметрами. |
| **R6 — одинаковая нагрузка** | AC1, AC4 | Проверяется фактическая последовательность индексов между ридерами, числом workers и режимами persistence. Учтены отброшенные изображения и различие подмножеств. |
| **R7 — границы времени** | AC5, AC12 | Начало перед `iter(loader)` и завершение после исчерпания определены однозначно. Подготовка процесса, discovery, byte warmup и отчёт вынесены за таймер. |
| **R8 — эпохи и кэш** | AC5, AC6 | Первая эпоха отделена от последующих; времена сохраняются, отсутствие поздних эпох имеет явное представление. Скрытый loader warmup исключён, смысл файлового прогрева ограничен. |
| **R9 — метрики** | AC4, AC6 | Формулы используют фактические выданные изображения. Проверяются неполный батч, нулевой результат и исключение незавершённой эпохи из успешной сводки. |
| **R10 — ошибки** | AC3, AC7 | Неуспех определён для ошибок чтения, неподдерживаемого входа, несовместимого батча и worker exception. Пропуски, подстановка изображений и скрытый fallback запрещены. |
| **R11 — результат и воспроизведение** | AC6, AC8, AC9, AC12 | Предусмотрены согласованные таблица/API, настройки, версии, список файлов, фактический pinning и ресурсные метаданные. Изоляция и baseline имеют отдельные проверки. |
| **R12 — наблюдение ресурсов** | AC10, AC11, AC12 | Заданы включение, интервал, группа процессов, идентичность PID+creation time, привязка к эпохам, исключение наблюдателя и cleanup. |
| **R13 — метрики ресурсов** | AC10, AC11 | Проверяются собственные CPU times, единица 100%=один CPU, временные веса, границы интервалов, RSS-суммы и максимум суммы одного обхода. Временное покрытие и неполнота процессов разделены. |
| **C1 — совместимость** | AC8, AC9 | Сохранены старые команды и значение `--nw 0`. Отсутствие torch/torchvision проверяется отдельным сценарием. |
| **C2 — жизненный цикл** | AC8, AC12 | Требуются spawn-совместимость, валидация параметров и освобождение workers после успеха и ошибки. Persistence ограничен своей конфигурацией. |
| **C3 — семантика изображений** | AC2, AC7 | Определены RGB, grayscale, alpha, EXIF и ICC. Ограничения относятся к исходным файлам; автоматическое RGB uint8-приведение не скрывает неподдерживаемые режимы. |
| **C4 — качество измерения** | AC3, AC5, AC9 | Заданы монотонный таймер, единое потребление и проверка геометрии независимыми ожидаемыми результатами. Порог ускорения не используется как критерий корректности. |
| **C5 — разработка** | AC8, AC9 | Указаны Python 3.12–3.13, `uv`, проверяемая пара torch/torchvision и сохранение существующих изменений. |
| **C6 — достоверность наблюдения** | AC10, AC11 | Проверяются отсутствие sampler при off, диагностирование недоступности, partial/N/A, cleanup и сохранение overhead в измеренном времени. |
| **C7 — сопоставимость памяти** | AC12 | Проверяются разные свежие процессы, baseline до итерации, обратный порядок конфигураций и хранение samples вне измеряемого процесса. |

Все AC1–AC12 имеют содержательные связи с требованиями. Потерянных обязательных направлений между требованиями и критериями приёмки не выявлено.

Проверка разрешённого контекста подтверждает следующие основания постановки:

- `get_img_filenames.py:get_img_filenames` действительно использует несортированный рекурсивный обход, срез выборки и общий набор расширений, включающий BMP/TIFF. Это согласуется с R1 и явным отказом без повторной фильтрации в C3/AC7.
- `benchmark.py` содержит byte warmup вне измеряемого чтения. `cli.py` и `test_cli_unified.py:test_benchmark_nw_zero_means_all_cpus` подтверждают старую семантику `--nw 0`; новая семантика ограничена новым режимом.
- `read_img.py:_GracefulReadCallable` возвращает `None` при части ошибок. Адаптеры `imgread_rs.py` и `local_rs.py` содержат fallback. Требования R10 и отдельный пакет ридеров учитывают эти различия.
- Текущие PIL/cv2/torchvision-адаптеры сами по себе не обеспечивают весь новый контракт C3. Задача требует отдельного обеспечения семантики и не считает прежние адаптеры готовым решением.
- `_type_conversion.py:to_image` подтверждает преобразование ndarray через `from_numpy`, HWC→CHW и contiguous, а также непосредственное принятие Tensor. Дополнительная безопасная граница R3 необходима и явно включена в контракт.
- `_geometry.py:resize_image` поддерживает сохранение выходного dtype, включая внутренний путь через float при необходимости. Исключение для внутренних вычислений kernel в R3 согласовано с этим поведением. `center_crop_image` задаёт округление центральных координат и padding.
- `_presets.py:ImageClassification` включает после геометрии преобразование в float и нормализацию. Задача явно исключает эти стадии и корректно ограничивает интерпретацию результата.
- `dataloader.py:DataLoader.__iter__`, `_BaseDataLoaderIter` и `_MultiProcessingDataLoaderIter` подтверждают создание и повторное использование итератора, запуск workers, ограничения prefetch/persistence, условное действие pinning и штатное завершение workers при исчерпании без persistence. R6/AC1 требуют проверять фактический порядок, что существенно при различиях жизненного цикла генератора и workers.
- `pyproject.toml` подтверждает диапазон Python и отсутствие torch/torchvision среди обязательных базовых зависимостей.

По общей рубрике проблема и область цельны; входные контракты и non-goals согласованы. Выбор D обоснован естественными представлениями ридеров и общей Tensor-геометрией; варианты N/P и повторное использование процесса рассмотрены с реальными затратами. Раздел передачи в планирование определяет допустимые технические решения и основания для пересмотра задачи.

Операционные риски учтены через ограничения памяти, обработку неуспеха, изоляцию и cleanup. Постановка не требует изменения исходных изображений, внешней передачи данных или необратимой миграции; отдельного обязательного сценария отката на уровне этой задачи не выявлено.

## Residual risks and minor notes

- Проверена постановка и её согласованность с разрешёнными исходниками. Тесты и бенчмарки не запускались; корректность будущей реализации должна подтверждаться AC1–AC12.
- Точная поддерживаемая пара torch/torchvision, механизм cleanup и сборщик ресурсов оставлены планировщику явно. Их практическая совместимость требует проверки при реализации.
- Внешние страницы и заявленные номера установленных версий отдельно не проверялись. Технические выводы ревью опираются на зафиксированные локальные исходники; история независимой критики не использовалась как доказательство корректности контракта.
- OS cache, изменение содержимого файлов, JPEG-различия, shared pages и пропущенные короткие процессы остаются ограничениями измерений. Задача отражает их в интерпретации результатов и признаках полноты.
- Экспортируемый manifest содержит точные пути выбранных файлов. Это предусмотренная локальная информация для воспроизведения; её внешняя публикация не входит в постановку.

## Verdict rationale

`approved`: по общей рубрике и рубрике задачи отсутствуют blocker/major. Существенные требования сформулированы проверяемо, альтернативы обоснованы, ограничения изображения и измерений согласованы. Оставленные решения являются явно обозначенной работой планировщика.

Статусы `draft`, `pending` и `Plan readiness: blocked` описывают ещё не завершённый формальный переход. Настоящий вердикт не утверждает, что принятие артефакта или публикация привязки уже выполнены.

## Correction handoff

- **Artifact revision reviewed:** `2`.
- **Reviewed artifact SHA-256:** `e802daaad00a1f3105aff5f4f660f716a586fd5061e62a93123ed860d39a7517`.
- **Bound artifact SHA-256:** `c4c316e2538ef2759418bf04fb238656dd02c3d51f2ffec7f80f61d7bc1dfd33`.
- **Binding state:** `complete` только когда текущий SHA-256 артефакта равен указанному Bound artifact SHA-256; иначе `incomplete`.
- **Created:** `2026-09-07T15:22:44+04:00`.
- **Final sidecar path:** `/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.task.review-01.md`.
- **Blocking finding IDs:** отсутствуют.
- **Suggested next action:** содержательные исправления по этому ревью не требуются. Wrapper может опубликовать ревью и заполнить принадлежащие ему метаданные привязки, сохранив выводы. Переход к планированию допускается после выполнения условий принятия из раздела «Передача в планирование».

## Манифест входного набора (метаданные записи)

<details>
<summary>Зафиксированные файлы независимого ревью</summary>

```json
{
  "artifact_revision": 2,
  "assignment": "20260907_143308_dataloader-benchmark.task.review-01",
  "files": [
    {
      "path": "/home/aya/.codex/plugins/cache/albu-workflows/codex-workflows/1.0.0-dev.0+codex.20260907085239/skills/review-work-artifact/references/review-artifact.md",
      "sha256": "57fff0cc9ecc14eabedb3687eefd8698468a1085d83f4d87e524cae39defbce3"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torch/utils/data/dataloader.py",
      "sha256": "cb8d82b0de999f21349b7dd261d30919fcf692f0b1d7504ca2594e88e8fd0e96"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision/transforms/_presets.py",
      "sha256": "5e5a7b80b2e73d00a3a7c798ef0f6e44ab90295eb02ecef5659308bc28219670"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision/transforms/v2/functional/_geometry.py",
      "sha256": "1e4f4d2ff22a6d0b7f2b8a09f28e57a327ed8f067c6f17e5d1b359af6b440325"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision/transforms/v2/functional/_type_conversion.py",
      "sha256": "efcc25d1d34fc17d3c8ce096e8a719486cbc44042acb132d76b4d44154255133"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/AGENTS.md",
      "sha256": "ce3e9ada50dd3bef35344136bb3f963e195558ff636bff05874e2674de6e784f"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.task.md",
      "sha256": "e802daaad00a1f3105aff5f4f660f716a586fd5061e62a93123ed860d39a7517"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/pyproject.toml",
      "sha256": "01a7ba3542c89a59ba84c809d9a3106c248b487c77a59fe2c7e41baa7e720e93"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/benchmark.py",
      "sha256": "f617a140554f60343ed53c301a55e697dc1ae94feb1c6ef1aebba586a354ca6d"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/cli.py",
      "sha256": "e16a046d8d54d7bd885149dfa808a33fa2ede88854f779947dd6562d4a7ea34b"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/get_img_filenames.py",
      "sha256": "3c140523a793c573dd6c87bfecd5a2a7d656bae7da4eea03c752584b6038ccac"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/img_libs/PIL.py",
      "sha256": "3cc0f7e02e7b0ec88269193f5dce2539496eccc01632b055c412587b3cacda70"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/img_libs/cv2.py",
      "sha256": "0dd342f13cd289c3afa00eff695d7dec1cbe6a4908a043e32902680b2d206ca5"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/img_libs/imgread_rs.py",
      "sha256": "c350a5ec67bfa2de43e8e06627d05d4ade2b998ec22c8c87b66accaf9e92f262"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/img_libs/local_rs.py",
      "sha256": "3c3ff7b4e273a0b5d615d91e31cc4bc86841c7ea6c1e2d06b836903bfff4cd12"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/img_libs/torchvision.py",
      "sha256": "a9e4a2447e5ecdd3d31d6bcbae43d43fde8fb1e07ca157ce365fb25f9c1fb5f2"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/src/imgread_benchmark/read_img.py",
      "sha256": "71657ea4ab3cd195f257dd59391c63e2cb071214b321c8d4d51a2e14bb345729"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/tests/test_benchmark_file_order.py",
      "sha256": "5338d431cf47f0c7b1568b1197ecdca9326f8eb46e3b4b4f43a25a75f908402e"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/tests/test_cli_unified.py",
      "sha256": "d3aed36e86f7fe9d20a365c11255a68958c9552e0df4498da24877bd055505f0"
    }
  ]
}
```

</details>
