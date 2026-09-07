# Implementation plan: сравнение ридеров в PyTorch DataLoader

| Field | Value |
|---|---|
| Artifact | `plan` |
| ID | `20260907_143308_dataloader-benchmark` |
| Revision | `1` |
| Status | `accepted` |
| Review | `approved` |
| Review reference | `/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.plan.review-01.md` |
| Source task | `/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.task.md` |
| Source task revision | `2` |
| Task approval | `/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.task.review-01.md`, round `01`, revision `2`, `approved` |
| Planning base SHA | `c0f450674620cc8119a73fc4222daf3bc84eb7e3` |
| Canonical repository | `/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark` |
| Implementation branch | `aya/dataloader-benchmark` |
| Workflow record root | `/home/aya/.codex/workflow-records/imgread-benchmark/20260907_143308_dataloader-benchmark` |
| Implementation review record root | `/home/aya/.codex/workflow-records/imgread-benchmark/20260907_143308_dataloader-benchmark/implementation-reviews` |
| Created | `2026-09-07T15:33:16+04:00` |
| Updated | `2026-09-07T17:00:35+04:00` |
| Owner | Пользователь проекта `imgread_benchmark` |

## Goal

Новая команда `imgread_benchmark dataloader` и Python API сравнивают четыре пути чтения одного snapshot через настоящий CPU DataLoader, публикуют проверяемые метрики каждой эпохи и по запросу CPU/RSS с явно указанной полнотой наблюдений, сохраняя старый CLI.

## Architecture and approach

Работа выполняется в указанном отдельном worktree от полного `Planning base SHA`, созданном по указанию пользователя. Исходное рабочее дерево остаётся источником принятых документов, но не исполняемого кода. Материальное отличие от свидетельств задачи: в этом коммите старый `BenchmarkImgRead` ещё не содержит незакоммиченных shuffle/byte warmup, а `tests/test_benchmark_file_order.py` отсутствует. Эти изменения не переносятся и не отменяются; новый режим реализует собственные необходимые sampler/warmup. При последующем объединении веток требуется сохранить обе функциональности и повторить регрессии CLI. Остальные проверенные контракты совпадают: несортированный discovery со срезом N, три прежние команды, старое `--nw 0` → все CPU, ленивые импорты и optional torchvision.

Новый пакет `imgread_benchmark.dataloader` отделяет snapshot/config/results, адаптеры, общий transform, исполнение и наблюдение ресурсов. Координатор проверяет конфиг и исходные форматы всех выбранных файлов, затем запускает отдельный чистый Python-процесс на каждый вызов конфигурации. Это одинаковая механика при monitoring on/off; при off psutil и sampler вообще не создаются. Процесс-потребитель импортирует только выбранный backend и необходимые torch/torchvision, читает байты для warmup без хранения датасета, создаёт Dataset/DataLoader и сообщает готовность. Один DataLoader живёт все эпохи конфигурации. Workers используют `spawn`; результаты эпох и ресурсные samples собирает внешний координатор.

Адаптеры возвращают RGB ndarray HWC или CPU RGB Tensor CHW. Один общий transform обеспечивает безопасную границу ndarray→Tensor, затем вызывает Tensor-функции torchvision с фиксированными параметрами. Прежние `read_img.py`, fallback-адаптеры и registry плагинов в новом пути не используются. Сохранённый manifest восстанавливает точный упорядоченный список; throughput использует фактические B/N. Наблюдение хранит сырые CPU counters и RSS вне потребителя, сопоставляет интервалы с опубликованными точными окнами эпох после их завершения и сохраняет partial/missing отдельно от успешности выдачи батчей.

Точные рабочие копии task, task review и текущего draft plan сохранены также в `docs/plans/` нового worktree. Пользовательский путь плана остаётся исходным; при принятии плана эти копии обновляются до полного совпадения перед коммитом.

**Состояние входа:** перед синхронизацией метаданных проверено точное совпадение task SHA-256 `ae51a064270fe902562e4dba862d7529c4306ccbb9d87170b55e244da6e48e61` с Bound artifact SHA-256 ревью. В `2026-09-07T15:33:16+04:00` синхронизированы только `Updated`, устаревшие записи готовности/ожидания ревью и bound-hash метаданные sidecar. Требования, AC, ревизия 2 и выводы ревью не изменены. Текущий task SHA-256: `c4c316e2538ef2759418bf04fb238656dd02c3d51f2ffec7f80f61d7bc1dfd33`; `Status=accepted`, `Review=approved`, `Plan readiness=ready`, содержательных блокирующих вопросов нет. Это не пропуск ревью.

## Technology and conventions

- Python 3.12–3.13; существующие `argparsecfg.App`, `field_argument`, dataclasses, Rich, NumPy и Pillow. Все Python-команды и зависимости — через `uv`.
- Поддерживаемая пара первой версии: `torch==2.10.0`, `torchvision==0.25.0`; локальный METADATA torchvision требует именно torch 2.10.0, и оба номера уже находятся в базовом `uv.lock`. Геометрия проверена по исходникам torchvision 0.25.0. Суффикс сборки (`+cpu`, `+cu128`) сохраняется в результате. Другую пару новый режим отклоняет до таймера; базовая установка не ограничивается этой проверкой.
- Новые extras: `dataloader = ["torch==2.10.0", "torchvision==0.25.0", "opencv-python-headless==4.13.0.90"]`, `monitor = ["psutil>=7.2,<8"]`. Точный psutil фиксирует обновлённый `uv.lock`; базовые зависимости и существующие extras сохраняются. Даже без OpenCV установленный через иной способ torch/torchvision позволяет выбрать PIL/torchvision; недоступные cv2-варианты показываются с причиной.
- MVP квалифицируется на Linux с Python 3.12 и 3.13 и worker start method `spawn`. Новый subprocess runner использует POSIX process groups; на неподдерживаемой платформе выполнение нового режима заранее сообщает ограничение, listing/help и старые команды остаются доступны. Расширение платформ требует отдельной проверки process cleanup.
- PEP 8, 4 пробела, snake_case/CamelCase, тесты pytest. Бинарные диагностические изображения создаются тестовыми fixtures в tmp_path; текущие `tests/test_imgs/*` не изменяются.
- [torchvision decode_image 0.25](https://docs.pytorch.org/vision/0.25/generated/torchvision.io.decode_image.html) предоставляет RGB mode, CHW Tensor и отключение EXIF orientation; выбран один этот путь. [OpenCV flags 4.13](https://docs.opencv.org/4.13.0/d8/d6a/group__imgcodecs__flags.html) задают RGB/BGR и IGNORE_ORIENTATION. Для psutil используются собственные user/system counters, creation time и RSS из [API reference](https://psutil.io/api/); API 8.x не входит в выбранный диапазон.

## Global constraints

- **C1 — совместимость:** существующие `benchmark`, default-вызов, `libs`, `data`, форматные адаптеры и значение старого `--nw 0` сохраняются. Новый режим не делает torch/torchvision обязательными для базовой установки; его зависимости и отсутствие этих зависимостей описаны явно, импорты ленивые.
- **C2 — жизненный цикл:** Dataset, ридеры и трансформы совместимы с `spawn`; workers предыдущей конфигурации освобождаются перед следующей, в том числе при исключении. `persistent_workers=True` при нуле workers и явно заданный prefetch при нуле workers отклоняются. Batch size, число эпох и применяемый prefetch положительные; число workers неотрицательное. Недопустимые сочетания не исправляются молча.
- **C3 — семантика изображений:** начальный гарантированный вход — обычные 8-bit RGB JPEG/PNG. Grayscale приводится к трём RGB-каналам; alpha отбрасывается без композиции; EXIF orientation не применяется; не выполнять ICC-преобразование. Вариант обязан соблюдать эти правила либо явно отказать в неподдерживаемом случае. Ограничения относятся к исходному файлу, а не только к dtype после decode: для исходных 16-bit, CMYK и многостраничных изображений MVP даёт явный отказ, даже если выбранный декодер умеет автоматически превратить их в RGB uint8. Проверка исходного режима не должна скрыто менять snapshot или включать альтернативный декодер в успешную строку. Побитовое совпадение разных JPEG-декодеров не является универсальным критерием.
- **C4 — качество измерения:** монотонный wall-clock таймер, одинаковое тело потребления батчей, без вычислений модели и подробного progress-вывода на каждый batch. Проверки содержания и индексов выполняются в тестах/проверочных запусках, а не как полное сравнение каждого изображения в основном замере. Не вводить порог ускорения как доказательство корректности.
- **C5 — разработка:** соблюдать `AGENTS.md`, Python 3.12–3.13, использовать `uv` для Python и зависимостей. Проверять поведение относительно поддерживаемой пары torch/torchvision. Существующие незакоммиченные изменения учитывать, не отменять.
- **C6 — достоверность наблюдения:** default off не запускает sampler и не требует его optional dependency. При явном включении недоступная зависимость диагностируется до замера; ошибка сбора во время замера отдельно помечает ресурсную часть как неполную, не превращая успешную итерацию в полные измерения ресурсов. Накладные расходы наблюдения не вычитаются из wall time; monitored/off и разный интервал различимы в результатах. Sampled peak не обещает поймать пик между наблюдениями; короткоживущие workers могут быть пропущены. Метод наблюдения и его известные ограничения описываются вместе с метриками.
- **C7 — сопоставимость памяти:** при включённом наблюдении каждая конфигурация получает свежий процесс-потребитель с выбранным backend, без состояния прежних конфигураций; один и тот же процесс сохраняется между её эпохами. Запуск этого процесса, его начальные импорты, создание Dataset/DataLoader и начальный RSS после этой подготовки находятся до таймера эпохи. Начальный RSS публикуется рядом с абсолютными наблюдаемыми уровнями; разность с baseline можно дать дополнительно, но она не обозначает точный объём выделений. Сэмплы мониторинга и результаты предыдущих конфигураций хранятся вне измеряемого процесса, чтобы их накопление не имитировало рост памяти DataLoader. Достаточны отдельные скалярные запуски CLI или эквивалентная изоляция вызова API; платформа экспериментов не требуется. Изоляция процесса не сбрасывает OS cache и не гарантирует одинаковую системную нагрузку.

- **G1:** Сохранить R1–R13 и AC1–AC12 исходной задачи. Нет обучения, GPU decode/transfer, DDP, optimizer/search, dashboard, всех старых backends, decoded-cache, float pipeline, performance threshold или универсального рейтинга библиотек.
- **G2:** Во время реализации менять только перечисленные файлы; task/plan/approval inputs внутри репозитория принудительно добавить в Git, поскольку `docs/` игнорируется. Не редактировать старые benchmark/cl_app/read_img/adapters ради нового режима.
- **G3:** Runnable root и cwd всех команд — `.` относительно canonical repository; собственный код — только `src/imgread_benchmark`, `tests`, `tests/dataloader_smoke.py`. Ни один путь не содержит `..`. Сейчас `.`/`src`/`tests` проверены внутри worktree; после создания новых путей implementation review повторяет `realpath`-проверку, включая symlink-цепочки и фактический `imgread_benchmark.__file__`. Не использовать editable install или PYTHONPATH из исходного checkout.
- **G4:** Оба абсолютных record root канонизированы через существующие предки и находятся вне worktree и вне исходного checkout. Создать их при handoff с mode 0700, перед каждым использованием повторять containment check; запретить symlink/path-alias возврат внутрь любого checkout. Records не помещаются в runnable/import/config/data trees. Принятые task/plan сохраняют пользовательские абсолютные пути.

## Requirement traceability

| Source | Implemented by | Verified by |
|---|---|---|
| R1, R6, AC1 | Tasks 1, 3 | config/manifest и фактические order tests; V1, V2, V6 |
| R2, R3, C3, AC2 | Tasks 1, 2 | source modes, adapter spies, tensor/array equivalence; V1, V2, V6 |
| R4, C4, AC3 | Task 2 | независимая геометрия, native-size collation; V2 |
| R5, R9, AC4 | Tasks 1, 3, 5 | N=10/B=4, drop_last, zero output; V1, V2, V3 |
| R7, R8, C4, AC5 | Task 3 | fake-clock boundary tests и spawn/persistence; V2 |
| R9, R11, AC6 | Tasks 1, 3, 5 | формулы, round-trip JSON, CLI/API equality; V1, V2, V3, V6 |
| R10, C3, AC7 | Tasks 1, 2, 3 | corrupt/unsupported/None/worker errors; V1, V2 |
| R11, C1, C2, C5, AC8 | Tasks 1, 3, 5, 6 | dependency isolation, spawn, pinning/cleanup; V2–V6 |
| R2, R5, R11, C1, C5, AC9 | Tasks 5, 6 | help/examples, legacy regressions, uv; V3–V6 |
| R12, R13, C6, AC10 | Task 4 | детерминированная CPU/RSS трасса и real monitor; V1, V2, V6 |
| R12, R13, C6, AC11 | Task 4 | boundary/missing/PID reuse/sampler failure tests; V1, V2 |
| R7, R11, R12, C7, AC12 | Tasks 3, 4, 6 | fresh consumer IDs, baseline и A/B/B/A; V2, V6 |

## File map

Все пути таблицы относятся к canonical repository. Это полный набор product/test/doc изменений реализации; переносов и удалений нет.

| Path | Operation | Responsibility after this change |
|---|---|---|
| `pyproject.toml` | modify | optional dataloader/monitor dependencies |
| `uv.lock` | modify | воспроизводимое разрешение extras |
| `src/imgread_benchmark/dataloader/__init__.py` | create | ленивый публичный API нового режима |
| `src/imgread_benchmark/dataloader/models.py` | create | конфиг, manifest/result/event schemas, ошибки и чистые расчёты |
| `src/imgread_benchmark/dataloader/manifest.py` | create | выборка, header-only source validation, stat identities, JSON manifest |
| `src/imgread_benchmark/dataloader/readers/__init__.py` | create | закрытый registry четырёх IDs, availability и lazy loading |
| `src/imgread_benchmark/dataloader/readers/pillow.py` | create | Pillow → RGB ndarray |
| `src/imgread_benchmark/dataloader/readers/torchvision.py` | create | decode_image → RGB CPU CHW Tensor |
| `src/imgread_benchmark/dataloader/readers/opencv.py` | create | два самостоятельных RGB decode paths |
| `src/imgread_benchmark/dataloader/dataset.py` | create | map Dataset, общий transform, строгий collate, epoch sampler |
| `src/imgread_benchmark/dataloader/engine.py` | create | подготовка, warmup, DataLoader, timer, counts, teardown adapter |
| `src/imgread_benchmark/dataloader/runner.py` | create | Python API, отдельный subprocess, IPC, cancellation и сбор результата |
| `src/imgread_benchmark/dataloader/_child.py` | create | закрытый `-m` entry point потребителя и IPC event writer |
| `src/imgread_benchmark/dataloader/resources.py` | create | optional внешний sampler и чистое сведение CPU/RSS |
| `src/imgread_benchmark/dataloader/report.py` | create | таблица/JSON одного результата |
| `src/imgread_benchmark/dataloader/cli.py` | create | конфиг и обработчик dataloader, лёгкая регистрация команды |
| `src/imgread_benchmark/cli.py` | modify | новый known command и его регистрация без тяжёлых импортов |
| `tests/dataloader_helpers.py` | create | детерминированные fixtures, picklable test helpers, fake clocks/process traces |
| `tests/test_dataloader_contracts.py` | create | config, manifest, headers, metrics, schema tests |
| `tests/test_dataloader_dataset.py` | create | adapters, transform, geometry, collation, actual index order |
| `tests/test_dataloader_runner.py` | create | timing, isolated execution, spawn, errors, pinning и cleanup |
| `tests/test_dataloader_resources.py` | create | sampler arithmetic, scope, missing и lifecycle |
| `tests/test_dataloader_cli.py` | create | новый CLI, JSON, ошибки и optional deps |
| `tests/test_cli_unified.py` | modify | passthrough dataloader и legacy routing regressions |
| `tests/test_lazy_imports.py` | modify | base-only import/old commands/new dependency diagnostic |
| `tests/dataloader_smoke.py` | create | ограниченный post-review acceptance run contract и проверка artifacts |
| `README.md` | modify | запуск/зависимости/сравнение/ограничения нового режима |
| `docs/dataloader-benchmark.md` | create | полный API/CLI/result/resource contract и воспроизведение |
| `docs/plans/20260907_143308_dataloader-benchmark.task.md` | create exact copy | принятый workflow input из Source task |
| `docs/plans/20260907_143308_dataloader-benchmark.task.review-01.md` | create exact copy | одобренное task review |
| `docs/plans/20260907_143308_dataloader-benchmark.plan.md` | create exact copy | принятый plan revision при implementation handoff |

План-review sidecar, когда появится, также копируется по его фактическому имени в `docs/plans/` и коммитится; имя и SHA записываются в handoff record. Это workflow input, не дополнительная product-задача.

## Interfaces and contracts

### Public API and CLI

| Interface | Consumed by | Produced or changed by | Contract |
|---|---|---|---|
| `DataLoaderConfig` frozen dataclass | Tasks 2–6 | Task 1 | `reader="pil-rgb"`, `num_workers=0`, `batch_size=32`, `shuffle=False`, `seed=0`, `epochs=5`, `prefetch_factor=None`, `persistent_workers=False`, `pin_memory=False`, `drop_last=False`, `geometry=True`, `warmup=True`, `monitor_resources=False`, `sample_interval_ms=100.0` |
| `snapshot_files(filenames: Sequence[str | Path], *, num_samples: int = 0) -> FileManifest` | runner/API/CLI | Task 1 | копирует список, срез N без сортировки/дедупликации; N=0 все, N<0/empty — ошибка |
| `discover_manifest(img_path: str | Path, *, num_samples: int = 0) -> FileManifest` | CLI | Task 1 | один вызов существующего `get_img_filenames`, затем snapshot; нет повторной фильтрации BMP/TIFF |
| `list_readers() -> tuple[ReaderInfo, ...]` | CLI/API | Task 2 | все четыре ID в фиксированном порядке, description/available/reason/version; проверки в отдельных короткоживущих probe subprocess, без загрязнения будущего consumer |
| `ImageListDataset(manifest: FileManifest, reader_id: str, geometry: bool)` | engine/tests | Task 2 | picklable map-style Dataset; `__len__()`; `__getitem__(index: int) -> tuple[torch.Tensor, int]`, target всегда 0 |
| `ImageTransform(geometry: bool = True).__call__(image: np.ndarray | torch.Tensor) -> torch.Tensor` | Dataset | Task 2 | один RGB/Tensor контракт, описанный ниже |
| `EpochSampler(length: int, seed: int, shuffle: bool)` / `set_epoch(epoch: int)` | engine | Task 2 | epoch начинается с 0; при shuffle `torch.randperm(length, generator=локальный CPU generator с seed=(seed+epoch) mod 2**63)`; иначе range; независим от generator DataLoader |
| `run_benchmark(manifest: FileManifest, config: DataLoaderConfig) -> BenchmarkResult` | CLI/Python | Task 3 | один fresh subprocess на вызов; сохраняет snapshot; ошибки конфигурации → ValueError, исполнения → BenchmarkRunError с `.result`, содержащим неуспех/частичные завершённые эпохи |
| `write_result(result: BenchmarkResult, output_dir: Path) -> None` / `render_result(result) -> rich.table.Table` | CLI/API | Task 5 | один schema source; export в новый каталог, без overwrite существующего запуска |

Публичные символы экспортируются только из `imgread_benchmark.dataloader`; прежний верхний `imgread_benchmark.__init__` не расширяется. Модели/manifest/registry descriptors не импортируют torch/cv2/psutil на уровне модуля; runtime typing через postponed annotations/TYPE_CHECKING.

CLI: `imgread_benchmark dataloader [img_path]` имеет `--reader`, `-n/--num-samples` (default 0), `--num-workers`, `--batch-size`, `--shuffle`, `--seed`, `--epochs`, `--prefetch-factor`, `--persistent-workers`, `--pin-memory`, `--drop-last`, `--no-geometry`, `--no-warmup`, `--monitor-resources`, `--sample-interval-ms`, `--output`, `--manifest`, `--list-readers`. Без listing требуется ровно один источник: каталог либо `--manifest manifest.json`; N применяется после загрузки полного manifest, по прежней slice-семантике. `--list-readers` не начинает benchmark. Новые флаги не входят в `_BENCHMARK_FLAG_ALLOWLIST`. Новый режим не добавляет alias `--nw`, чтобы две семантики не смешивались. Exit codes: 0 success/listing, 2 неверный ввод/недоступные зависимости/backend/platform, 1 decode/worker/export/run failure, 130 отмена.

Валидация: batch/epochs >0, workers≥0, seed целый в `[0, 2**63)`, sample interval конечный и >0; явный prefetch положительный и запрещён с workers=0; effective prefetch None при workers=0, иначе заданный либо 2. Persistence при workers=0 — ошибка. `drop_last` с N<batch size отклоняется как zero-output до замера. Requested/effective prefetch сохраняются раздельно. Изображения читаются только CPU; pin_memory не означает перенос на GPU.

### Source and image semantics

Manifest version 1 содержит упорядоченный массив абсолютных нормализованных путей с сохранением повторов, requested/selected N, source header (`format`, `mode`, `bit_depth`, `width`, `height`, `frames`), stat identity (`size`, `mtime_ns`, `dev`, `ino`) и `selection_id = sha256(UTF-8 canonical JSON массива путей, ensure_ascii=False, separators=(",", ":")))`. Hash списка не утверждает неизменность содержимого. Перед каждым decode Dataset сверяет stat identity; изменение означает явную ошибку, а не пересбор snapshot. Сохранённые файлы не переписываются; adversarial rewrite с сохранением stat не считается гарантированно обнаруживаемым.

Общий preflight до запуска конфигурации проверяет **весь** snapshot, включая будущие drop_last элементы. Разрешены только JPEG/PNG с исходными 8-bit RGB/L и PNG RGBA/LA; другие режимы явно unsupported. Header-only Pillow `Image.open` определяет format/mode/frame count, без `load`/`convert`/alternate decode; PNG IHDR дополнительно проверяет bit depth и color type, поскольку RGB 16-bit нельзя определять по dtype декодированного изображения. JPEG SOF проверяет sample precision. Неправильный/обрезанный header, исходные 16-bit, CMYK, palette/1-bit, animated/multipage, BMP/TIFF дают reader/path/reason и неуспех. При Pillow-адаптере decode всё равно выполняется заново внутри эпохи. Это общая подготовительная проверка, не запасной декодер. Тест с JPEG/PNG под другим расширением фиксирует проверку реального формата, а не суффикса.

ID и точный decode path:

- `pil-rgb`: `Image.open(path)` в context manager → `.convert("RGB")` → `np.asarray`; не применять EXIF transpose или ImageCms. Возвращать ndarray непосредственно, даже read-only; lifetime проверяет тест.
- `torchvision-rgb`: `torchvision.io.decode_image(path, mode=ImageReadMode.RGB, apply_exif_orientation=False)`; Tensor напрямую, без NumPy. Это единственный torchvision ID.
- `cv2-rgb`: `cv2.imread(path, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)`; ndarray напрямую. Отсутствие любого нужного флага делает вариант unavailable.
- `cv2-bgr-cvtcolor`: `cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)` → `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`. `None` проверяется до cvtColor. Замены этой реализации срезом в MVP нет.

У всех путей grayscale становится RGB, alpha отбрасывается без композиции, EXIF/ICC не применяются. Общая граница валидирует rank/каналы/uint8/CPU. Writable positive-strided NumPy допускает `torch.from_numpy(...).permute(2,0,1)`; read-only, отрицательные/нулевые strides или выравнивание, несовместимое с from_numpy, получают одну общую C-order writable copy. Положительный non-contiguous вход допустим без обязательной копии массива. После получения CHW у **обоих** входных типов применяется одна политика `contiguous()`, если это требуется; готовый совместимый contiguous Tensor сохраняется. Нет ветвлений по reader ID, изменения исходника, float cast, `/255` или Normalize.

Далее одна реализация `torchvision.transforms.v2.functional.resize(tensor, [256], interpolation=InterpolationMode.BILINEAR, antialias=True)` → `center_crop(tensor, [224,224])`. Выход обычный Tensor, не `tv_tensors.Image`. Выбранная версия определяет длинную сторону как `int(256*long/short)`, anchor crop — `int(round((dimension-224)/2.0))` с Python ties-to-even. Внутренний kernel может считать float; выход остаётся uint8. При geometry=False только общая входная граница, исходные H/W сохраняются. `ContextDataset(dataset: ImageListDataset).__getitem__(index)` — внутренняя обёртка engine — добавляет путь к паре Dataset и возвращает `(image, 0, path)`. Module-level `collate_images(samples)` сравнивает shapes, при mismatch сообщает reader, shapes и пути, иначе передаёт только пары `(image,0)` в обычный `default_collate`, возвращая `(images, targets)`. Публичный ImageListDataset и его прямой обычный DataLoader сохраняют `(image,0)`; контекстные пути не попадают в выдаваемый engine batch.

### Timing, isolation and lifecycle

Координатор `Popen([sys.executable, "-m", "imgread_benchmark.dataloader._child"], start_new_session=True)` запускает consumer с cwd `.` canonical worktree; subprocess не переисполняет пользовательский `__main__`. Запрос — JSON stdin, события — JSON-lines stdout; диагностический вывод/предупреждения — stderr, постоянно дренируемый координатором. `_child` резервирует stdout для одного синхронизированного event writer. Protocol version 1, `config_id` во всех сообщениях: `ready`, `worker_started`, `epoch_result`, `run_error`, `done`; stdin: `run_request`, `go`, `cancel`. EOF, ненулевой exit или отсутствие `done` — неуспех. IPC не передаёт изображения.

В child до ready: импорт выбранного backend, thread policy, config validation, byte warmup чанками 1 MiB, Dataset/loader creation. В дочернем env OMP/MKL/OPENBLAS_NUM_THREADS=1; `torch.set_num_threads(1)` и interop=1, для cv2 `setNumThreads(1)`; те же настройки внутри module-level worker initializer. Значения, effective getters и неизвестные настройки Pillow записываются. Эти изменения относятся только к новому consumer/workers и не меняют среду вызывающего Python-процесса. Данные worker RNG seed задаются отдельно от sampler, и DataLoader получает отдельный generator. Явно `in_order=True`, multiprocessing_context=`spawn` только при workers>0, timeout=0. Один объект loader на конфигурацию.

После ready внешний sampler снимает baseline, если monitoring on, и координатор отправляет go. Для каждой эпохи sampler.set_epoch выполняется до `t0=perf_counter_ns()`. Следующая операция — `iter(loader)`; полностью потребить iterator тем же циклом во всех конфигурациях, считать `batch[0].shape[0]` и batch count, не накапливать батчи. `t1` берётся сразу после StopIteration. Normal non-persistent worker shutdown при exhaustion остаётся внутри этого окна. Затем вне таймера проверяется `last_images.is_pinned()`, удаляется последний batch, формируется epoch_result, отправляются timing window/метрики/предупреждения. Это наблюдение последнего реального батча, не гарантия pinning всех батчей и не дополнительная прогревочная итерация.

Если ошибка возникает в эпохе, её статус failed, elapsed/counts могут сохраняться как partial, но successful rates и поздняя медиана её не используют. Отдельно публикуются первая эпоха и median секунд **завершённых последующих** эпох; отсутствие поздних эпох → null. Вся конфигурация failed при любой ошибке эпохи, даже если есть предыдущие успешные. `pin_memory` содержит requested, last_batch_observed, accelerator_available, warnings и status (`observed_pinned`, `observed_unpinned`, `not_requested`, `not_observed`); никаких обещаний ускорения.

`engine.close_loader` — единственная граница приватного torch API `_shutdown_workers`/`_iterator`; она привязана к 2.10.0, идемпотентна, вызывает shutdown для сохранённого iterator, освобождает pin thread/queues и ссылки в finally. Координатор ждёт orderly завершения 10 s, затем SIGTERM группе consumer, ещё 5 s, затем SIGKILL; проверяет выход группы перед следующей конфигурацией. Ctrl-C/SIGTERM обрабатываются тем же finally. Никакого psutil для cleanup off. В тестах с умершими/зависшими workers проверяется отсутствие живых PIDs, а не только завершение consumer. После неудачного принудительного cleanup следующая конфигурация не запускается.

### Resource observation and result schema

`ResourceSampler(consumer_identity, interval_seconds)` существует только в координаторе с явно включённым monitoring. Сначала import psutil/preflight его process API, затем запуск child; отсутствие зависимости — ошибка до измерения. Sampler стартует после ready до go, baseline — отдельный обход после подготовки, до epoch 0. Worker initializer при monitoring on однократно получает собственную identity через psutil и отправляет `worker_started(worker_id,pid,create_time,registered_monotonic_ns)` через multiprocessing Queue в служебный relay child; relay пересылает событие координатору, не хранит историю. Identity сверяется с sampled create_time, поэтому запоздалая регистрация не присваивается новому владельцу PID. Это только worker startup overhead (включая optional psutil import), без batch hooks и без sampler в worker. Samples и прошлые результаты никогда не хранятся в child. При off не создаются sampler, psutil import, registry Queue или relay.

Группа — consumer `(pid,create_time)` и зарегистрированные worker roots с их потомками. Observer/coordinator, его посторонние дети и multiprocessing resource_tracker вне worker roots исключаются. Sampler может предварительно записывать кандидатов-потомков consumer, но включает их в сводки только после установления принадлежности зарегистрированному worker; неподтверждённые помечаются excluded/unknown. Worker registration delay, неуспевший наблюдаться worker, AccessDenied, исчезновение и несовпадение create_time видны как причины неполноты. PID reuse начинает новую identity и не продолжает предыдущие counters.

`ResourceSample` version 1: config_id, sweep_start_ns/end_ns, assigned epoch или null, processes (`pid`, `create_time`, `parent_identity`, `role`, counter timestamp, user_seconds/system_seconds/rss_bytes либо null, status/reason), expected/observed worker identities, discovery status, totals и completeness. Обходы не перекрываются; после задержки sampler не делает burst catch-up. Первый обход нужен для baseline counters, а не нулевой CPU. Привязка к эпохам выполняется по windows из epoch_result; pending samples не угадывают эпоху по моменту получения IPC.

CPU-интервал между `sweep_end_ns` двух последовательных обходов использует разности только собственных user+system counters одной identity, никогда children_user/children_system. Для включения оба полных обхода и промежуток между ними должны лежать внутри одного `[t0,t1]`; отрицательная delta/неположительная длительность → missing. `cpu_percent=100*sum(valid_process_deltas)/interval_seconds`; timestamps отдельных чтений также сохраняются, интервал обходов — явное приближение. При missing процесса известная частичная сумма допустима с partial, а при отсутствии всех counters — null. Mean = сумма(percent*dt)/сумма(dt) валидных неперекрывающихся интервалов; max = максимум включённых percent. Cross-boundary интервалы сохраняются с причиной и исключаются из mean/max/coverage, без интерполяции. Coverage = сумма включённых dt / epoch_seconds; отдельно complete_process_interval_count/partial count и reasons. Интервал с известным только consumer может дать partial CPU и временное coverage; это не доказательство полноты workers.

RSS включается, только если весь sweep находится внутри эпохи. Суммы в bytes: consumer, workers+descendants, total; missing не превращается в zero. Известная неполная сумма помечается partial. Worker RSS=0 допустим, когда config workers=0 и отсутствуют ожидаемые worker descendants. Для каждой серии выводятся арифметическое среднее наблюдений и max в MiB (bytes/2**20); total max вычисляется по per-sweep total, не суммой индивидуальных максимумов. Baseline публикуется отдельно, baseline вычитание не вводится. В summaries counts полных/частичных наблюдений раздельны; выборка без валидного RSS или CPU имеет null, отсутствие ожидаемых частичных пиков не считается нулём. Ошибка sampler останавливает только наблюдение, сохраняет exception/partial resource status и не переписывает успешное wall-time измерение.

`BenchmarkResult` version 1: status/error, config_id (hash канонического effective config + selection_id), execution_id (новый UUID), manifest, config requested/effective, reader info, environment, consumer identity (`pid`, `execution_id`, `started_wall_time`, `os_create_time`; последнее null с reason при off, psutil creation time при on), isolation=`fresh_subprocess`, preparation/baseline, ordered epochs, late_epoch_seconds_median, pinning, resource_status, resource_samples, resource_intervals и warnings. Каждый `EpochResult`: index/kind(first|subsequent), status, start/end ns, epoch_seconds, images_delivered, batches_delivered, images_dropped, `order_id` и `delivered_order_id` (хэши индексов sampler/выданного prefix, вычисляются вне timer), rates либо null, resource summary. Эти IDs отличают несовпадающие drop_last подмножества. Environment: Python/package/backend/torch/torchvision/psutil versions, OS/kernel/machine, logical/available CPU и affinity при доступности, start method, thread env/effective getters, geometry policy и CPU capability torchvision; недоступное — null с reason.

Export в `--output DIR`: `manifest.json`, `result.json`, `samples.jsonl`, `intervals.jsonl`; resource streams пусты при off, result ссылается на них и не дублирует их массивы. Python result содержит сами структуры; `write_result` и parser обеспечивают без потерь round-trip. JSON — UTF-8, schema_version=1, allow_nan=False, отсутствующие метрики null. Таблица показывает config/reader, epoch, elapsed, N/B/dropped, ms/image, images/s, first/late median, requested/effective pinning; при monitoring также baseline, CPU mean/max/coverage, RSS mean/observed peak, interval/sample count и partial labels. Описание RSS shared pages/non-atomic sweep/missed peaks/observer overhead обязательно. Ни время эпохи, ни samples не публикуются вовне автоматически.

## Task dependency graph

- Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6.
- Один исполнитель последовательно владеет пересекающимися models/helpers/runner; отдельные параллельные редакторы не нужны. Изменение зафиксированного интерфейса требует согласованного обновления всех потребителей и тестов до следующей задачи.

## Implementation tasks

### Task 1: валидируемый snapshot и структурированный контракт

**Outcome:** без запуска DataLoader можно получить воспроизводимый manifest, отклонить неподдерживаемый исходник/конфиг и сериализовать результат с корректными метриками.

**Depends on:** None. **Requirements:** R1, R5, R9–R11, C1, C3, C5, AC1, AC4, AC6, AC7.

**Files:** create `src/imgread_benchmark/dataloader/__init__.py`, `src/imgread_benchmark/dataloader/models.py`, `src/imgread_benchmark/dataloader/manifest.py`, `tests/dataloader_helpers.py`, `tests/test_dataloader_contracts.py`; modify `pyproject.toml`, `uv.lock`.

**Interfaces:** consumes `get_img_filenames`; produces DataLoaderConfig, FileManifest, ReaderInfo, BenchmarkResult/error/event dataclasses, snapshot/discover functions и schema version 1.

**Steps:**

- [ ] Добавить тесты неизменности списка, порядка/повторов/N=0/N<0/empty, одинаковых selection IDs и manifest round-trip с Unicode/пробелами. Написать параметрические проверки всех недопустимых config combinations.
- [ ] Добавить fixtures RGB/L/RGBA/LA JPEG/PNG, 16-bit grayscale и RGB PNG (IHDR 16), CMYK JPEG, multipage TIFF, animated PNG, BMP, corrupt header. Проверить отказ до decoder call и отсутствие фильтрации неподходящего элемента snapshot.
- [ ] Запустить `uv run --extra test pytest -q tests/test_dataloader_contracts.py`; ожидаемый начальный сбой — отсутствие нового модуля/контрактов, не проблема fixture.
- [ ] Реализовать manifest/header validation/models и формулы на заданных счётчиках. Зафиксировать null на zero/failed, вычисления 2 s/10 images → 200 ms/image и 5 images/s; 1 эпоха → late median null.
- [ ] Добавить extras, выполнить `uv lock` без upgrade несвязанных зависимостей, проверить lock diff. Импорты новых моделей/manifest должны работать без torch/torchvision/psutil.

**Task verification:** `uv run --locked --extra test pytest -q tests/test_dataloader_contracts.py`. Expected: все перечисленные cases pass; parser manifest отвергает неверную schema/hash и не изменяет исходные файлы. Optional binary backends не требуются для этого набора.

**Risks / rollback:** header после преобразования может скрывать глубину — проверять исходные IHDR/SOF. Откат — убрать новый пакет/добавленные extras; старый discovery не меняется.

### Task 2: четыре decoder paths и одна Dataset-геометрия

**Outcome:** каждый поддерживаемый reader выдаёт эквивалентный по семантике CPU uint8 batch через общий transform; реальные индексы воспроизводятся.

**Depends on:** Task 1. **Requirements:** R2–R4, R6, R10, C2–C4, AC1–AC3, AC7.

**Files:** create `src/imgread_benchmark/dataloader/readers/__init__.py`, `src/imgread_benchmark/dataloader/readers/pillow.py`, `src/imgread_benchmark/dataloader/readers/torchvision.py`, `src/imgread_benchmark/dataloader/readers/opencv.py`, `src/imgread_benchmark/dataloader/dataset.py`, `tests/test_dataloader_dataset.py`; extend `tests/dataloader_helpers.py`.

**Interfaces:** consumes FileManifest/DataLoaderConfig/ReaderInfo; produces list_readers, lazy reader callables, ImageTransform, ImageListDataset, EpochSampler и internal contextual collate.

**Steps:**

- [ ] Добавить spies, подтверждающие неизменный объект RGB ndarray на входе transform у cv2 и отсутствие Tensor→NumPy у torchvision, отдельные IDs и unavailable прямого RGB при удалённом флаге.
- [ ] Зафиксировать failing tests dtype/range/shape, RGB/grayscale/alpha/EXIF, исходных 0/128/255 и shared transform. Тестовые заранее декодированные ndarray/Tensor дают точно одинаковый выход без изменения источника; проверить read-only/positive non-contiguous/negative strides, buffer lifetime после GC.
- [ ] Добавить независимые геометрические ожидания: 100×200 → 256×512 → crop anchor (16,144), 128² → 256² → anchor (16,16); 101×203 → 256×514 → (16,145); 256×257 → crop left=16 и 256×259 → left=18 для ties-to-even. Координатные каналы при short side=256 проверяют точную центральную область без resize; отдельные ramp/constant/impulse случаи проверяют bilinear и antialias с независимым аналитическим эталоном, допуском ≤1 intensity для kernel rounding. Shape-only теста недостаточно.
- [ ] Выполнить `uv run --locked --extra test --extra dataloader pytest -q tests/test_dataloader_dataset.py`; подтвердить failures будущих интерфейсов, затем реализовать adapters/common transform/Dataset/collate/sampler.
- [ ] Проверить geometry off: batch равных размеров stack, mismatch сообщает context и падает, batch_size=1 принимает разные размеры. Ошибка/None ридера не заменяется изображением. Проверить stat change до decode.
- [ ] С помощью отдельного picklable test Dataset, кодирующего исходный index в значениях (не в target production), собрать фактически полученные последовательности в 3 эпохах: workers 0/2, persistence off/on где допустимо, shuffle false/true, два независимых процесса, все четыре читателя на identity-coded PNG. Сравнить последовательности, а не только seeds/sampler hashes.

**Task verification:** тот же `uv run --locked --extra test --extra dataloader pytest -q tests/test_dataloader_dataset.py`. Expected: четыре reader paths проходят на квалифицируемом окружении; имитированная unavailable API диагностируется; фактические orders совпадают для одинаковых seed/epoch, без принудительного отличия соседних перестановок.

**Risks / rollback:** JPEG не сравнивать побитово между библиотеками; использовать контрольные цвета/ориентацию и lossless PNG для exact equality. Не менять пространственный эталон при различиях native kernel.

### Task 3: измеряемый проход и жизненный цикл изолированной конфигурации

**Outcome:** Python API полностью потребляет эпохи в новом процессе, корректно считает throughput и освобождает workers при успехе/ошибке/отмене.

**Depends on:** Task 2. **Requirements:** R5–R11, C2, C4, C7, AC1, AC4–AC8, AC12.

**Files:** create `src/imgread_benchmark/dataloader/engine.py`, `src/imgread_benchmark/dataloader/runner.py`, `src/imgread_benchmark/dataloader/_child.py`, `tests/test_dataloader_runner.py`; extend `src/imgread_benchmark/dataloader/models.py`, `src/imgread_benchmark/dataloader/__init__.py`, `tests/dataloader_helpers.py`.

**Interfaces:** consumes Dataset/sampler/config/manifest; produces run_benchmark, protocol v1, close_loader, epoch windows, counts, pinning/env/isolation metadata. `monitor_resources=True` до Task 4 даёт явный NotImplementedError в development, не молчаливое off; этот промежуточный отказ удаляется Task 4.

**Steps:**

- [ ] Написать fake clock/event-log tests с различными затратами imports/discovery/preflight/warmup/loader constructor/iter/next/StopIteration/report. Ожидаемый интервал включает iter и полное exhaustion, исключает подготовку/отчёт. Проверить исключение посреди эпохи и отсутствие её rates в summary.
- [ ] Добавить N=10, batch=4 → 4/4/2 и N=10; drop_last → 4/4, N=8, dropped=2; N<batch с drop_last → неуспех; не вычислять успешные zero rates.
- [ ] Запустить focused тесты, затем реализовать engine и JSON subprocess protocol. Импорты/подготовка до ready; sampling order не зависит от loader generator и persistence. Сериализованный request не принимает произвольные callables/код.
- [ ] Реализовать thread policy, chunked byte warmup, счётчики, first/subsequent aggregation, order IDs и post-timer last-batch pinning probe. Проверить отсутствие hidden extra epoch и хранения batch history.
- [ ] Реализовать pinned-version cleanup adapter и parent process-group fallback. Spawn error, corrupt file в worker, invalid batch, KeyboardInterrupt, EOF и принудительно зависший worker должны завершаться явным failure с reader/path где они известны. Errors без известного file path честно содержат null и stage.
- [ ] Выполнить success/failure/forced cancellation в subprocess тестах и проверить прекращение всех worker identities до второго run. С persistence PID работников сохраняются между эпохами одной конфигурации; новые run имеют новые consumer process identities. PID сам по себе недостаточен при reuse.
- [ ] Проверить pin requested/effective без accelerator и замоканный supported-pinned путь; реальные batches проверяются только после таймера. Python caller, уже импортировавший другие readers и хранящий прежний результат, не передаёт это состояние новому consumer.

**Task verification:** `uv run --locked --extra test --extra dataloader pytest -q tests/test_dataloader_runner.py`. Expected: успешные real workers=0/2 spawn, детерминированные timer/count assertions, отсутствие живых дочерних групп после всех exits; no psutil import при off.

**Risks / rollback:** приватный teardown ограничен одной функцией и одной torch-парой. Если bounded cleanup не доказан, не выпускать runner и не запускать следующую конфигурацию; не скрывать failure принудительной сменой workers.

### Task 4: внешнее наблюдение CPU/RSS с явной полнотой

**Outcome:** monitoring on добавляет baseline/raw samples/epoch summaries, а off остаётся независим от psutil; CPU/RSS арифметика не завышает достоверность.

**Depends on:** Task 3. **Requirements:** R11–R13, C6, C7, AC10–AC12.

**Files:** create `src/imgread_benchmark/dataloader/resources.py`, `tests/test_dataloader_resources.py`; extend `src/imgread_benchmark/dataloader/runner.py`, `src/imgread_benchmark/dataloader/_child.py`, `src/imgread_benchmark/dataloader/engine.py`, `src/imgread_benchmark/dataloader/models.py`, `tests/dataloader_helpers.py`.

**Interfaces:** consumes ready/worker_started/epoch_result windows; produces ResourceSampler, ResourceSample, ResourceInterval, per-epoch ResourceSummary and baseline; ресурсная ошибка отделена от decode failure.

**Steps:**

- [ ] Добавить fake psutil process-tree/clock trace: consumer+двое workers и потомок, внешний observer/посторонний процесс исключены; разные dt 1 s и 3 s, собственные суммы delta 1 s и 6 s дают 100%/200%, weighted mean=175%, max=200%, total valid dt=4 s. Искусственно большие children_user/system не меняют результат.
- [ ] RSS трасса с (consumer,workers)=(100,20) и (80,70) MiB даёт total mean=135 и observed peak=150, а не 170. Проверить метки non-atomic sweep и отсутствие RSS за пределами window.
- [ ] Добавить first sample, исчезновение worker, PID reuse, AccessDenied, counter rollback, отрицательный/NaN interval, no samples, короткую эпоху, sampler exception и cross-boundary interval. Последний сохраняется, но не влияет на CPU mean/max/coverage. Пример 4 s валидных интервалов в эпохе 8 s → coverage=0.5 даже при partial process scope.
- [ ] Запустить `uv run --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_resources.py`; реализовать чистую агрегацию до real polling, затем внешний sampler и worker registration relay. Ни одного per-batch сообщения/ожидания.
- [ ] Интегрировать baseline ready→sample→go, корректную epoch association после завершения, bounded sampler stop/join и сериализацию failure. Проверить быстрый worker, умерший до регистрации/первого sweep: missing/partial, не zero. Проверить count expected workers в каждой эпохе с/без persistence.
- [ ] Real spawn tests с 0 и 2 workers проверяют структуру/идентичности/окна и cleanup; synthetic slow Dataset только в тесте гарантирует ≥2 samples, без искусственного delay в production loop. Два reader/config A/B и затем B/A запускаются в одном API-координаторе, имеют четыре свежих consumers и отдельные baselines; абсолютное равенство RSS не требуется.
- [ ] Заблокировать import psutil и убедиться: off проходит без создания sampler/relay; on падает до timer с dependency diagnostic. Во время sampler error успешная итерация остаётся success, resource_status=partial с error и сохранёнными observations; overhead не вычитается.

**Task verification:** тот же focused pytest command плюс `uv run --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_runner.py`. Expected: точные детерминированные числа, явные null/partial, runtime samples хранятся только в координаторе, sampler/relay/worker cleanup после success/error.

**Risks / rollback:** sampled RSS не уникальная RAM; PID/process discovery неполон. При regression пользователь может отключить monitoring; результат on никогда не подменяется off под тем же config_id.

### Task 5: CLI, экспорт и документация одного результата

**Outcome:** режим доступен из прежнего unified CLI, API/table/export согласованы, установка без heavy deps сохраняет старые сценарии.

**Depends on:** Task 4. **Requirements:** R2, R5, R9–R13, C1, C5, C6, AC6, AC8–AC11.

**Files:** create `src/imgread_benchmark/dataloader/cli.py`, `src/imgread_benchmark/dataloader/report.py`, `tests/test_dataloader_cli.py`, `docs/dataloader-benchmark.md`; modify `src/imgread_benchmark/cli.py`, `tests/test_cli_unified.py`, `tests/test_lazy_imports.py`, `README.md`.

**Interfaces:** consumes run_benchmark/list_readers/result schema; produces exact CLI flags, exit codes, write_result/render_result and documented API.

**Steps:**

- [ ] Добавить routing tests нового known command, help без torch, отсутствие новых флагов в legacy allowlist и прежнее `benchmark --nw 0` → все CPU. Существующие тесты unified CLI/lazy imports остаются регрессиями.
- [ ] Написать CLI tests с контролируемым BenchmarkResult для table/JSON equality, null/N/A, failure exit и export partial результата без successful failed-epoch rates. Test parse saved manifest повторяет ровно список исходного запуска, включая порядок/повторы.
- [ ] Зарегистрировать лёгкий dataloader command через существующий argparsecfg, грузить runner лишь после разбора/валидации. Listing проверяет нужные API и сообщает версии/reasons без fallback.
- [ ] Реализовать export в новый private directory (0700, файлы 0600, при существующем каталоге отказ), schema round-trip и coherent resource streams. Файл error сохраняет полный stage/reason; не выдавать неполный JSON за успешный результат.
- [ ] Обновить README коротким запуском, extras install и ссылкой на подробный документ. В подробном документе дать все defaults/флаги, Python API с сохранением manifest, повтор с другим workers/batch/shuffle, no-geometry/batch_size=1, drop_last, единицы, cold-cache limitation, first/late median, supported pair/platform, source restrictions, pin requested/effective и CPU/RSS caveats.
- [ ] Проверить `--help`, `--list-readers`, invalid config, missing extras, старые `--version`, `libs`, mocked `data`, default benchmark и отсутствие torch/torchvision/psutil в base import через отдельный base-only uv environment, без `--extra dev` (он уже включает torchvision).

**Task verification:** `uv run --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_cli.py tests/test_cli_unified.py tests/test_lazy_imports.py`. Expected: точные mapping/exit/status assertions; README команды используют реальные имена и API.

**Risks / rollback:** основной риск — нормализация CLI или eager imports. Откат только регистрации dataloader возвращает прежний CLI; старые функции не переписываются. Документировать merge overlap с незакоммиченным CLI исходного checkout.

### Task 6: воспроизводимая приёмка и чистый handoff

**Outcome:** реализация имеет проверенное покрытие AC, ограниченный сценарий post-review запуска и все inputs/code привязаны к чистому коммиту.

**Depends on:** Task 5. **Requirements:** все R/C/AC, в особенности AC8, AC9, AC12.

**Files:** create `tests/dataloader_smoke.py`; finish documentation in `README.md`, `docs/dataloader-benchmark.md`; copy workflow inputs from unchanged user-visible paths into `docs/plans/` at handoff, without substantive edits.

**Interfaces:** consumes public API/export; produces `prepare`, `run`, `verify` guarded script commands и artifacts run contract ниже.

**Steps:**

- [ ] Реализовать небольшой локальный acceptance script строго по матрице ниже; это test harness, без поиска оптимума и произвольного scheduler. Manifest создаётся один раз; все строки используют его.
- [ ] Запустить V1–V5 в Python 3.13, затем функциональный набор V2/V3 в чистом Python 3.12 env. Проверить shape/geometry для обеих версий Python, без изменения torch/torchvision pair. Для реальных 0/2-worker tests пропуск из-за окружения означает незавершённую квалификацию, а не выполненный AC.
- [ ] Сверить все R/C/AC с traceability, обновить документацию итоговыми подтверждёнными install/runtime details. Никаких numeric speed claims до измерений; acceptance не требует победителя.
- [ ] Скопировать принятые workflow inputs побайтно из исходного `docs/plans/` в worktree, добавить их и подробный документ через `git add -f` (docs игнорируется), остальные изменения обычным `git add` по file map. Не включать чужие файлы исходного checkout и runtime outputs.
- [ ] Повторить containment/import origin, проверить lock/working tree и закоммитить всю реализацию и in-repository workflow inputs. Не требуются промежуточные ceremonial commits. `git status --porcelain=v1 --untracked-files=all` должен быть пустым.
- [ ] Вызвать `$review-implementation /home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.plan.md` из нового implementation session с указанным canonical worktree. Только после approved exact SHA выполнить V6 run contract. Исправление кода после review требует нового коммита/implementation review и новых run IDs; документационные control records писать внешне.

**Task verification:** V1–V7 и external implementation review. Expected: clean committed SHA, immutable review sidecar с approved, локальные acceptance artifacts совпадают с code/manifest hashes, processes освобождены; отсутствуют speed thresholds и изменённые исходные изображения.

**Risks / rollback:** docs игнорируются и легко остаются вне коммита; `git ls-files --error-unmatch` проверяет каждую обязательную копию. Проблема окружения записывается как incomplete/blocked в handoff, без смены нагрузки/reader под прежним ID.

## Cross-cutting verification

Все команды запускаются из `.` canonical repository; implementation env сначала готовится командой `uv sync --extra test --extra dataloader --extra monitor`. Дополнительные проверки запускаются только при новом изменении/ошибке. Здесь команды планируемые: планирование не запускало product tests и не устанавливало зависимости в worktree.

| ID | Verification | Command or procedure | Expected evidence |
|---|---|---|---|
| V1 | Чистые контракты/арифметика | `uv run --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_contracts.py tests/test_dataloader_resources.py` | config/header/manifest tests pass; CPU=175%/200%, RSS peak=150 MiB и все missing/boundary cases pass |
| V2 | Реальные Dataset/DataLoader и lifecycle | `uv run --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_dataset.py tests/test_dataloader_runner.py tests/test_dataloader_resources.py` | все 4 readers, 0/2 workers, spawn/persistence, independent orders, geometry, timer и cleanup pass |
| V3 | CLI/base compatibility | `uv run --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_cli.py tests/test_cli_unified.py tests/test_lazy_imports.py` | flags/exports/errors/legacy и blocked-import checks pass |
| V4 | Полные regression tests/сборка/стиль | `uv run --locked --extra test --extra dataloader --extra monitor pytest --cov=imgread_benchmark`; `uv run --with ruff ruff check src/imgread_benchmark/dataloader tests/test_dataloader_contracts.py tests/test_dataloader_dataset.py tests/test_dataloader_runner.py tests/test_dataloader_resources.py tests/test_dataloader_cli.py tests/dataloader_helpers.py tests/dataloader_smoke.py`; `uv run --with build python -m build` | нет новых failures; отсутствующие unrelated optional libs допускают только существующие skips; wheel/sdist содержат новый пакет; lint нового кода pass, для изменённого legacy cli проверить отсутствие новых lint diagnostics относительно базы |
| V5 | Чистые environments Python 3.12/3.13 и base-only | Три точные команды в блоке V5 ниже | обе Python-версии проходят функциональные проверки; ни одна среда не импортирует код другого checkout; base_install проверяет реальное отсутствие torch/torchvision/psutil и старые команды, новый execution даёт exit 2 |
| V6 | Post-review bounded smoke | `prepare` → `run` → `verify` из run contract | 20 успешных config executions, одинаковая selection/order workload, fresh identities/baselines, корректный экспорт; короткие resource окна имеют N/A/partial, не придуманные числа |
| V7 | Code/input binding и containment | `git rev-parse HEAD`; `git status --porcelain=v1 --untracked-files=all`; `git ls-files --error-unmatch pyproject.toml uv.lock docs/dataloader-benchmark.md docs/plans/20260907_143308_dataloader-benchmark.task.md docs/plans/20260907_143308_dataloader-benchmark.task.review-01.md docs/plans/20260907_143308_dataloader-benchmark.plan.md`; realpath/import-origin procedure | исходные SHA/inputs сохранены, все paths существуют внутри worktree, рабочее дерево чисто; внешние records не входят в Git |

Команды V5 (каждая `uv run` подготавливает свою отдельную среду из lock):

```bash
UV_PROJECT_ENVIRONMENT=/tmp/imgread-dataloader-venv-312 uv run --python 3.12 --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_dataset.py tests/test_dataloader_runner.py tests/test_dataloader_resources.py tests/test_dataloader_cli.py tests/test_cli_unified.py tests/test_lazy_imports.py
UV_PROJECT_ENVIRONMENT=/tmp/imgread-dataloader-venv-313 uv run --python 3.13 --locked --extra test --extra dataloader --extra monitor pytest -q tests/test_dataloader_dataset.py tests/test_dataloader_runner.py tests/test_dataloader_resources.py tests/test_dataloader_cli.py tests/test_cli_unified.py tests/test_lazy_imports.py
UV_PROJECT_ENVIRONMENT=/tmp/imgread-dataloader-base uv run --python 3.13 --locked --extra test pytest -q tests/test_dataloader_cli.py -k base_install
```

Для V5 test `base_install` не должен на collection импортировать heavy modules: отдельные imports только внутри runtime tests, с явным skip остальных heavy tests в base env. Сам base_install test запускает свежие интерпретаторы с blocked/absent deps и мокирует download, а не обращается к сети. Общий функциональный прогон, где heavy deps установлены, помечает только этот specifically base-only case skipped с причиной; его обязательный отдельный V5 запуск должен pass.

## Migration, rollout, and rollback

Миграции пользовательских данных нет: новый пакет, подкоманда и versioned local output. Существующие результаты старого benchmark не конвертируются. Rollout — optional extras и opt-in dataloader, monitoring default off. Пользователь может продолжать старую команду без новых зависимостей. Откат нового режима — revert его реализации и optional extras/lock changes; изображения и чужие результаты не затрагиваются. Output schema имеет version 1, future readers не маскируются под существующие IDs. Ветка не сливается и не публикуется во время планирования/приёмочных прогонов; возможный merge с исходными CLI/README edits делается отдельно с сохранением обеих сторон и V3/V4.

## Observability and operations

Наблюдаемость — сам локальный результат: epoch durations/counts, per-process resource samples, env/config/manifest и ошибки стадий preflight/prepare/iterate/observe/export/cleanup. Consumer stderr сохраняется отдельно координатором; подробного batch progress нет. Нет telemetry/service/network upload/alerts. README показывает интерпретацию resource partial и CPU coverage, limitations RSS shared pages, overhead и неполного обнаружения процессов. Экспорт может содержать абсолютные пути пользователя; его внешняя публикация в scope не входит.

## Experiment run contract

**Applicability:** `required` — только ограниченная функциональная квалификация нового benchmark после implementation review. Полномасштабный Imagenette/ImageNet ranking не является принятым outcome. Development unit/integration tests из Tasks 1–5 выполняются до review как проверки разработки; результаты performance smoke привязываются лишь к approved clean SHA.

| Contract item | Frozen value |
|---|---|
| Runnable root and working directory | `.` для каждого узла; harness `tests/dataloader_smoke.py`, product `src/imgread_benchmark`; existing/final realpaths внутри canonical worktree, без `..`/symlink escape. Данные/outputs внешние и не попадают в PYTHONPATH |
| Code binding | Только exact clean committed SHA, approved `$review-implementation`. Harness читает HEAD/clean status и `imgread_benchmark.__file__`, отказывается при несовпадении с external review. Изменение кода аннулирует reuse |
| DAG and run matrix | S0 preflight → S1 prepare → S2 run → S3 verify. Обязательная матрица в лексическом фиксированном порядке: reader `pil-rgb`, `torchvision-rgb`, `cv2-rgb`, `cv2-bgr-cvtcolor`; для каждого workers/persistence `(0,false)`, `(2,true)`; внутри `monitor=false,true`. 16 configurations. Далее в одном coordinator API процессе 4 вызова A,B,B,A: A=`pil-rgb,0,false,monitor=true`, B=`cv2-bgr-cvtcolor,2,true,monitor=true`. Всего 20 executions; epochs=3, N=10, batch=4, shuffle=true, seed=0, drop_last=false, geometry=true, warmup=true, pin=false, effective prefetch None/2, interval=100 ms. Другие edge cases покрываются V1–V5, optional nodes отсутствуют |
| Commands | S0: V7 и проверка ресурсов ниже. S1: `uv run --locked --extra test --extra dataloader --extra monitor python tests/dataloader_smoke.py prepare --root /tmp/imgread-dataloader-acceptance/20260907_143308_dataloader-benchmark`. S2: `timeout --signal=INT --kill-after=15s 900s uv run --locked --extra test --extra dataloader --extra monitor python tests/dataloader_smoke.py run --root /tmp/imgread-dataloader-acceptance/20260907_143308_dataloader-benchmark`. S3: `uv run --locked --extra test --extra dataloader --extra monitor python tests/dataloader_smoke.py verify --root /tmp/imgread-dataloader-acceptance/20260907_143308_dataloader-benchmark`. Ни shell glob, ни автоматическое изменение matrix |
| Environment and data provenance | Linux x86_64, Python 3.13, locked extras и указанные pair/OpenCV; GPU не используется, в smoke subprocess env `CUDA_VISIBLE_DEVICES=""`, OMP/MKL/OPENBLAS=1. S1 генерирует ровно 10 RGB PNG 320×480, index i=0…9, deterministic channels R=(x+i)%256, G=(y+3*i)%256, B=(x+y+7*i)%256, имена `000.png`…`009.png`; Pillow save PNG без metadata. Сохраняет SHA-256 каждого encoded файла, manifest, generator source SHA и lock SHA. Каждый run использует сохранённый manifest, S3 сверяет hashes; скачивания datasets нет |
| Resource envelope and concurrency | Только local host: доступно ≥4 logical CPUs, ≥8 GiB available RAM, ≥2 GiB свободного artifact storage, ≥256 MiB свободной `/dev/shm`; CPU threads=1 у consumer/worker, 2 workers maximum. Maximum active attempts=1, никаких параллельных benchmark процессов/конфигураций. До запуска проверить headroom; при нехватке остановить квалификацию, не уменьшать workload. PSS/GPU resources отсутствуют |
| Budgets and stop conditions | S1 ≤60 s, S2 ≤900 s total, каждая конфигурация ≤60 s с учётом подготовки; S3 ≤60 s; общий лимит 17 min плюс до 15 s cleanup. 20 executions, максимум одна попытка на logical ID за invocation, ноль платных ресурсов/GPU-hours. Таймаут, dependency/platform mismatch, OOM, resource exhaustion, decode/cleanup failure прекращают S2; нет продолжения ради хорошей строки |
| Artifact and logging contract | ROOT=`/tmp/imgread-dataloader-acceptance/20260907_143308_dataloader-benchmark`; внутри `<approved-full-SHA>/data`, `manifest.json`, `environment.json`, `matrix.json`, `attempt-<UUID>/` с command/env, per-config result/manifest/samples/intervals/stdout/stderr и `verification.json`. `logical_id=sha256(code_sha + selection_id + effective_config_canonical_json + matrix_position)`; runtime UUID отдельный. Mode 0700 dirs/0600 files, не коммитить. Сохранять до решения пользователя; `/tmp` не обещает retention через системную очистку, внешний workflow record хранит hashes и outcome |
| Metrics and frozen decision rule | S3 — authoritative structural evaluator: все 20 runs success; по 3 завершённые эпохи N=10/B=3/dropped=0, rates равны формулам с относительным допуском 1e-9, late median двух subsequent seconds; selection/order IDs и фактические order tests согласованы; fresh consumer identities для всех 20 и A/B/B/A, pin not_requested, monitored baseline и methods присутствуют. Off не создаёт sampler, on допускает честные null/partial для коротких windows. Валидные samples/intervals лежат в своих epochs, cross-boundary не входят в CPU summaries. Никакой минимальной скорости/рейтинга; недоступный reader в этом квалифицируемом matrix — incomplete, не success |
| Retry, resume, reuse, and fallback | Нет автоматических retries/epoch resume или reuse старых performance results. S1 может переиспользовать только точный тот же data manifest после всех content hashes. После устранения внешней transient причины возможен новый полный invocation с новым attempt UUID на неизменном approved SHA, старый outcome сохраняется. Code fix → новый SHA/review и отдельный root. Запрещены reader fallback, смена pair/матрицы/seed/workers/интервала и выдача partial epoch за success |
| Durable launch, monitoring, and cancellation | Один foreground supervisor GNU `timeout` для S2, harness контролирует 60 s per-config, coordinator всегда дренирует IPC/stderr и сохраняет завершённые records вне измерений. Состояние процесса/выхода проверяется после каждой конфигурации, без печати per-batch. Для этого короткого smoke нет detached scheduler/job. Ctrl-C/timeout → cancel, cleanup consumer process groups (10 s graceful, 5 s TERM, затем KILL), остановка sampler/relay, failure record. После потери supervisor не запускать повтор, пока PID/group прежней попытки не проверены и не остановлены |
| Privacy, permissions, and external effects | Только генерируемые локальные картинки; пользовательские изображения не меняются. Установка pinned deps через uv допускается на implementation setup, S1–S3 сеть не используют. Никаких upload, email, deployment, root/sysctl/cache-drop, paid service или системных thread settings |
| Workflow/control records | Только абсолютный Workflow record root из metadata: immutable launch/result summaries, checked SHA/root paths, approval reference, artifact hashes и outcome. Implementation sidecar только Implementation review record root; оба вне repo и вне data/artifact tree. Их создание/запись не загрязняет измеряемый commit |

Harness не вводит универсальную платформу запусков: три фиксированных команды и небольшая матрица данного acceptance contract. Порог ресурсов относится к квалификации, не к произвольным пользовательским запускам CLI. Внешняя installation/setup failure фиксируется отдельно от benchmark performance.

## Documentation updates

- README: extras, `dataloader --list-readers`, один пример scalar run, ссылка на подробный контракт и отличие нового workers=0 от старого --nw=0.
- `docs/dataloader-benchmark.md`: API/flags/defaults/все IDs, source-mode limits, transform uint8/no scaling, warmup и timer, manifest restore, resource units/partial/isolation/baseline, supported Linux/pair, errors/cleanup/export и полный run contract reference.
- Примеры используют `uv run --extra dataloader --extra monitor imgread_benchmark dataloader tests/test_imgs --reader pil-rgb --num-workers 0 --batch-size 2 --epochs 3 --output /tmp/imgread-dataloader-example-a`; повтор с `--num-workers 2 --persistent-workers --shuffle --monitor-resources` и новым output `...-b`; воспроизведение того же списка через `--manifest /tmp/imgread-dataloader-example-a/manifest.json`. Каталоги outputs должны быть новыми. Эти примеры — документация, не дополнительная обязательная performance-матрица.

## Implementation handoff

- **Session boundary:** после acceptance плана или явного user review skip реализация начинается в новой Codex session в `/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark`, ветка `aya/dataloader-benchmark`. В planning/review context продукт не реализуется.
- **Workflow records:** user-visible task/plan/review остаются по исходным абсолютным путям. Это внешние inputs по отношению к новому worktree; при handoff сделать exact copies в его `docs/plans/`, включая появившееся plan review. После любого принятого изменения обновлять copies до byte equality. Копии, code и подробные docs коммитятся через явное добавление игнорируемого docs. Абсолютный план для `$review-implementation` остаётся исходным; reviewer использует explicit Canonical repository и сравнивает committed copies. Никакие records не записываются в исходный mutable checkout после handoff, кроме user-approved правок самих task/plan/review inputs.
- **Execution prerequisites:** текущая ревизия плана accepted и approved/skipped; clean planning-base worktree; доступные Linux/Python3.12–3.13/runtime dependencies; оба canonical record roots вне обоих checkout, ещё раз проверенные после создания. Source task остается revision 2 approved. Изменившийся HEAD не принимать молча: сверить drift, при материальном изменении перепланировать.
- **User-run or external steps:** dependency downloads и квалификация на требуемом Linux host выполняются implementation session по указанным командам; доступа к private dataset/GPU/service не требуется. Публикация, merge и дополнительные бенчмарки сюда не входят. При недоступности qualified host оставить это внешним невыполненным условием, не объявлять AC8/V6 выполненными.
- **Safe stopping points:** после каждого focused deliverable/test pass; во время benchmark — cancel и подтверждённый cleanup перед остановкой. Падение sampler позволяет завершить timing с partial resource status, падение decoder/worker прекращает конфигурацию.
- **Decisions requiring escalation:** изменение dtype/геометрии/общей границы, обход исходных mode restrictions, mandatory NumPy round-trip Tensor, другой batch контракт, отказ от resource isolation, автоматическая фильтрация/fallback, смена frozen validation matrix/ресурсов, расширение до training/GPU/all plugins. Patch-version/pair change затрагивает pinned cleanup/geometry и требует plan correction/review.
- **Out of scope during implementation:** исправление/коммит чужого грязного исходного дерева, полный performance рейтинг, перенос legacy shuffle/warmup, rearchitecture старого CLI/adapters, новые внешние сервисы.
- **Implementation review:** закоммитить завершённый код и все in-repository workflow inputs, обеспечить абсолютно чистое стабильное рабочее дерево. Затем `$review-implementation /home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.plan.md`; sidecar `20260907_143308_dataloader-benchmark.implementation.review-<NN>.md` создаётся только под external Implementation review record root. Approval относится только к exact SHA; reviewer заново доказывает realpath containment созданных файлов и import origin. Post-review performance runs стартуют только после этого approval.

## Review request

Проверьте этот план revision `1` против source task revision `2` в свежем изолированном контексте, без истории авторского диалога и без содержательных правок файла. Проверьте покрытие R1–R13/C1–C7/AC1–AC12, выбор planning base и явный drift рабочего дерева, файлы/API/schema, task ordering, воспроизводимый sampler, отсутствие source-mode обхода, общую геометрию, timing/cleanup/pinning, CPU/RSS scope и partial arithmetic, CLI/lazy compatibility, команды/риски/документацию, V1–V7 и bounded run contract. Проверьте, что оба абсолютных control record roots канонизируются вне canonical worktree и исходного checkout, а runnable roots не выходят из reviewed worktree; отвергайте symlink/path aliases внутрь repo для records или наружу для runnable code. Проверьте handoff exact copies и обязательный clean committed implementation review. Для каждого finding укажите section/source ID/task. Вердикт `approved` допустим только без blocker/major; иначе `changes-requested`. Если сохраняете review, используйте `20260907_143308_dataloader-benchmark.plan.review-<NN>.md` рядом с user-visible планом и укажите точную ревизию.

## Review history

| Round | Plan revision | Reviewer | Decision | Review file or reference |
|---|---:|---|---|---|
| — | 1 | — | `pending` | Самопроверка автора выполнена; независимое ревью плана не проводилось |
| 01 | 1 | independent-agent; `/root/dataloader_plan_review_01` | `approved` | [20260907_143308_dataloader-benchmark.plan.review-01.md](/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.plan.review-01.md) |
